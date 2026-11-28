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


class ToolEvent(TypedDict):
    name: str
    retrieve_top_k: int
    top_k: int
    result_count: int


class AgentState(TypedDict):
    query: str
    search_query: str
    retrieve_top_k: int
    rerank_top_k: int
    use_reranker: bool
    retry_count: int
    trace: Annotated[list[TraceEvent], add]
    tool_calls: Annotated[list[ToolEvent], add]
    auto_route: bool
    route: NotRequired[str]
    route_source: NotRequired[str]
    router_model: NotRequired[str | None]
    router_usage: NotRequired[TokenUsage | None]
    router_attempts: NotRequired[int]
    direct_text: NotRequired[str]
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
