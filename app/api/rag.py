"""HTTP adapter for the existing RAGEngine. No model initialization in routes."""
import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import Field, FiniteFloat

from app.api.errors import error_response
from app.llm.exceptions import LLMConfigurationError, LLMRequestError, LLMResponseError
from app.llm.schemas import TokenUsage
from app.rag.chain import RAGEngine
from app.rag.citations import RAGCitationError
from app.rag.prompts import PromptBudgetError
from app.rag.schemas import RAGAnswer
from app.schemas import ErrorResponse, ResponseModel, SearchHitResponse, SearchRequest

router = APIRouter(tags=["rag"])
NonNegative = Annotated[int, Field(ge=0)]
Duration = Annotated[FiniteFloat, Field(ge=0)]


class EvidenceResponse(SearchHitResponse):
    evidence_id: Annotated[int, Field(ge=1)]
    truncated: bool


class CitationReportResponse(ResponseModel):
    structural_status: Literal["passed", "not_applicable"]
    semantic_status: Literal["not_run", "supported", "unsupported", "uncertain"]
    inline_ids: list[int]
    reason: str | None


class RAGResponse(ResponseModel):
    query: str
    answer: RAGAnswer
    evidence: list[EvidenceResponse]
    cited_evidence: list[EvidenceResponse]
    candidate_count: NonNegative
    retrieved_count: NonNegative
    dropped: dict[str, NonNegative]
    context_budget_units: NonNegative
    context_count_method: str
    prompt_bytes: NonNegative
    model: str | None
    usage: TokenUsage | None
    attempts: NonNegative
    timings_ms: dict[str, Duration]
    retrieval_timings_ms: dict[str, Duration]
    refusal_reason: str | None
    citation_report: CitationReportResponse
    judge_model: str | None
    judge_usage: TokenUsage | None
    judge_attempts: NonNegative


async def get_rag_engine(request: Request) -> RAGEngine | None:
    state = request.app.state
    if not getattr(state, "ready", False):
        return None
    return getattr(state, "rag_engine", None)


RAGDep = Annotated[RAGEngine | None, Depends(get_rag_engine)]


@router.get("/rag/health")
async def rag_health(request: Request, engine: RAGDep):
    """Local readiness only; does not send a paid upstream request."""
    if engine is None:
        return error_response(request, 503, "RAG_UNAVAILABLE", "RAG service unavailable or disabled")
    return {"status": "ok", "ready": True,
            "semantic_validation": engine.settings.rag_semantic_validation}


@router.post("/rag", response_model=RAGResponse, responses={
    422: {"model": ErrorResponse}, 429: {"model": ErrorResponse},
    502: {"model": ErrorResponse}, 503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
})
async def rag(body: SearchRequest, request: Request, engine: RAGDep):
    if engine is None:
        return error_response(request, 503, "RAG_UNAVAILABLE", "RAG service unavailable or disabled")
    settings = engine.settings
    k = settings.retrieve_top_k if body.retrieve_top_k is None else body.retrieve_top_k
    n = min(settings.rerank_top_k, k) if body.rerank_top_k is None else body.rerank_top_k
    if n > k:
        return error_response(request, 422, "VALIDATION_ERROR", "rerank_top_k must not exceed retrieve_top_k")

    slots = request.app.state.rag_slots
    # Reject overload instead of forming an unbounded queue. No await between
    # locked() and acquire() on the same event loop when a slot is available.
    if slots.locked():
        return error_response(request, 429, "RAG_BUSY", "RAG concurrency limit reached",
                              headers={"Retry-After": "1"})
    await slots.acquire()
    task = None

    def release_when_finished(done):
        slots.release()
        # Retrieve an exception from a task that outlived the HTTP request.
        if not done.cancelled():
            done.exception()

    try:
        # Shield work so a timed-out request does not release the concurrency
        # slot while a synchronous FAISS/model operation is still executing.
        task = asyncio.create_task(engine.answer(
            body.query, retrieve_top_k=k, rerank_top_k=n, use_reranker=body.use_reranker))
        task.add_done_callback(release_when_finished)
        request.app.state.rag_tasks.add(task)
        task.add_done_callback(request.app.state.rag_tasks.discard)
        result = await asyncio.wait_for(asyncio.shield(task), timeout=settings.rag_request_timeout)
        return RAGResponse.model_validate(result)
    except asyncio.TimeoutError:
        return error_response(request, 504, "RAG_TIMEOUT", "RAG request deadline exceeded")
    except PromptBudgetError:
        return error_response(request, 422, "RAG_PROMPT_BUDGET", "Question and evidence exceed prompt budget")
    except RAGCitationError:
        return error_response(request, 502, "RAG_CITATION_INVALID", "Generated answer failed citation validation")
    except LLMConfigurationError:
        return error_response(request, 503, "RAG_UNAVAILABLE", "RAG model configuration unavailable")
    except LLMRequestError as exc:
        if exc.status_code == 429 or exc.status_code == 503:
            return error_response(request, 503, "LLM_UNAVAILABLE", "Generation service temporarily unavailable")
        return error_response(request, 502, "LLM_REQUEST_FAILED", "Generation service request failed")
    except LLMResponseError:
        return error_response(request, 502, "LLM_RESPONSE_INVALID", "Generation service returned an invalid response")
    finally:
        if task is None:
            slots.release()
