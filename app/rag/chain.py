"""M3.4: asynchronous orchestration of the existing retrieval and generation stages."""
import asyncio
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Protocol

from app.config import Settings
from app.llm.schemas import ChatResult, TokenUsage
from app.rag.context import ContextBuilder, Evidence
from app.rag.prompts import PromptBuilder
from app.rag.schemas import RAGAnswer


class SearchBackend(Protocol):
    def search(self, query: str, *, retrieve_top_k: int, rerank_top_k: int,
               use_reranker: bool): ...


class GenerationBackend(Protocol):
    async def chat_json(self, messages, schema, *, max_message_bytes: int | None = None): ...


class RAGCitationError(ValueError):
    """A generated citation points outside the selected evidence."""


@dataclass(frozen=True)
class RAGResult:
    query: str
    answer: RAGAnswer
    evidence: tuple[Evidence, ...]
    cited_evidence: tuple[Evidence, ...]
    candidate_count: int
    retrieved_count: int
    dropped: dict[str, int]
    context_budget_units: int
    context_count_method: str
    prompt_bytes: int
    model: str | None
    usage: TokenUsage | None
    attempts: int
    timings_ms: dict[str, float]
    retrieval_timings_ms: dict[str, float]
    refusal_reason: str | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["answer"] = self.answer.model_dump(mode="json")
        data["usage"] = self.usage.model_dump(mode="json") if self.usage is not None else None
        return data


class RAGEngine:
    """Reuse caller-owned retrieval and LLM clients; never load models per question."""
    def __init__(self, settings: Settings, retrieval: SearchBackend, llm: GenerationBackend):
        self.settings = settings
        self.retrieval = retrieval
        self.llm = llm
        self.context_builder = ContextBuilder(settings)
        self.prompt_builder = PromptBuilder(settings)

    async def answer(self, query: str, *, retrieve_top_k: int | None = None,
                     rerank_top_k: int | None = None, use_reranker: bool = True) -> RAGResult:
        started = perf_counter()
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 2048:
            raise ValueError("Query must contain 1..2048 characters")
        query = query.strip()
        k = self.settings.retrieve_top_k if retrieve_top_k is None else retrieve_top_k
        if type(k) is not int or not 1 <= k <= 200:
            raise ValueError("retrieve_top_k must be an integer between 1 and 200")
        n = min(self.settings.rerank_top_k, k) if rerank_top_k is None else rerank_top_k
        if type(n) is not int or not 1 <= n <= k:
            raise ValueError("rerank_top_k must be an integer between 1 and retrieve_top_k")
        if type(use_reranker) is not bool:
            raise ValueError("use_reranker must be a boolean")

        # The engine is synchronous and already owns its concurrency semaphore.
        # Offload it so network waits and other async tasks can continue.
        t = perf_counter()
        search = await asyncio.to_thread(
            self.retrieval.search, query, retrieve_top_k=k,
            rerank_top_k=n, use_reranker=use_reranker,
        )
        retrieval_ms = (perf_counter()-t)*1000
        t = perf_counter()
        context = await asyncio.to_thread(self.context_builder.build, search.results)
        context_ms = (perf_counter()-t)*1000

        prompt_bytes = 0
        prompt_ms = generation_ms = citation_ms = 0.0
        llm_result: ChatResult | None = None
        refusal_reason = None
        if not context.evidence:
            # Bilingual fallback when no generation call is warranted.
            answer = RAGAnswer(answer="当前检索证据不足，无法回答。 / The retrieved evidence is insufficient.",
                               citations=[], insufficient_evidence=True)
            refusal_reason = "no_evidence"
            cited = ()
        else:
            t = perf_counter()
            prompt = self.prompt_builder.build(query, context)
            prompt_ms = (perf_counter()-t)*1000
            prompt_bytes = prompt.message_bytes
            t = perf_counter()
            answer, llm_result = await self.llm.chat_json(
                list(prompt.messages), RAGAnswer, max_message_bytes=prompt.budget,
            )
            generation_ms = (perf_counter()-t)*1000
            # A backend must honor the same output contract as LLMClient.
            answer = RAGAnswer.model_validate(answer.model_dump() if isinstance(answer, RAGAnswer) else answer,
                                             strict=True)
            t = perf_counter()
            by_id = {e.evidence_id: e for e in context.evidence}
            if any(i not in by_id for i in answer.citations):
                raise RAGCitationError("Generated citation ID is outside selected evidence")
            cited = tuple(by_id[i] for i in answer.citations)
            citation_ms = (perf_counter()-t)*1000
            if answer.insufficient_evidence:
                refusal_reason = "model_insufficient_evidence"

        return RAGResult(
            query=query, answer=answer, evidence=context.evidence, cited_evidence=cited,
            candidate_count=search.candidate_count, retrieved_count=context.input_count,
            dropped=dict(context.dropped), context_budget_units=context.estimated_tokens,
            context_count_method=context.count_method, prompt_bytes=prompt_bytes,
            model=llm_result.model if llm_result is not None else None,
            usage=llm_result.usage if llm_result is not None else None,
            attempts=llm_result.attempts if llm_result is not None else 0,
            timings_ms={"retrieval": retrieval_ms, "context": context_ms,
                        "prompt": prompt_ms, "generation": generation_ms,
                        "citation": citation_ms, "total": (perf_counter()-started)*1000},
            retrieval_timings_ms=dict(search.timings_ms), refusal_reason=refusal_reason,
        )
