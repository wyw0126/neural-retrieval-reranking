"""Per-invocation Agent state. No clients, credentials or private reasoning."""
from operator import add
from typing import Annotated, TypedDict
from typing_extensions import NotRequired

from app.schemas import SearchResponse
from app.rag.context import ContextResult, Evidence
from app.rag.schemas import RAGAnswer
from app.rag.citations import CitationReport
from app.llm.schemas import TokenUsage


class TraceEvent(TypedDict):
    node: str
    elapsed_ms: float
    status: str


class AgentState(TypedDict):
    query: str
    search_query: str
    retrieve_top_k: int
    rerank_top_k: int
    use_reranker: bool
    retry_count: int
    trace: Annotated[list[TraceEvent], add]
    search: NotRequired[SearchResponse]
    context: NotRequired[ContextResult]
    answer: NotRequired[RAGAnswer]
    cited_evidence: NotRequired[tuple[Evidence, ...]]
    citation_report: NotRequired[CitationReport]
    refusal_reason: NotRequired[str | None]
    model: NotRequired[str | None]
    usage: NotRequired[TokenUsage | None]
    attempts: NotRequired[int]
    judge_model: NotRequired[str | None]
    judge_usage: NotRequired[TokenUsage | None]
    judge_attempts: NotRequired[int]
    prompt_bytes: NotRequired[int]
