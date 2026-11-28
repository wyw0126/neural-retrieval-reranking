"""Allowlisted, schema-validated retrieval tools for graph nodes and future routers."""
import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.config import Settings
from app.rag.chain import SearchBackend
from app.schemas import QueryText, RequestModel, SearchResponse, TopK


class DenseSearchArgs(RequestModel):
    query: QueryText
    top_k: TopK | None = None
    retrieve_top_k: TopK | None = None

    @model_validator(mode="after")
    def validate_k(self):
        if self.top_k is not None and self.retrieve_top_k is not None and self.top_k > self.retrieve_top_k:
            raise ValueError("top_k must not exceed retrieve_top_k")
        return self


class RerankedSearchArgs(DenseSearchArgs):
    pass


class ToolCall(RequestModel):
    name: Literal["dense_search", "reranked_search"]
    arguments: dict


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: type[BaseModel]
    use_reranker: bool

    def definition(self) -> dict:
        return {"type":"function","function":{"name":self.name,
                "description":self.description,"parameters":self.schema.model_json_schema()}}


class ToolRegistry:
    """Only registered read-only tools may execute; no arbitrary dynamic dispatch."""
    def __init__(self, settings: Settings, retrieval: SearchBackend):
        self.settings = settings
        self.retrieval = retrieval
        self._specs = {
            "dense_search": ToolSpec("dense_search",
                "Retrieve passages by dense vector similarity without cross-encoder reranking. "
                "Use when lower retrieval latency is preferred.", DenseSearchArgs, False),
            "reranked_search": ToolSpec("reranked_search",
                "Retrieve candidate passages and rerank them with a cross-encoder. "
                "Use when relevance ordering and precise evidence selection matter.", RerankedSearchArgs, True),
        }

    def definitions(self) -> list[dict]:
        return [spec.definition() for spec in self._specs.values()]

    async def invoke(self, name: str, arguments: dict) -> SearchResponse:
        call = ToolCall(name=name, arguments=arguments)
        spec = self._specs[call.name]
        args = spec.schema.model_validate(call.arguments, strict=True)
        k = self.settings.retrieve_top_k if args.retrieve_top_k is None else args.retrieve_top_k
        n = min(self.settings.rerank_top_k,k) if args.top_k is None else args.top_k
        if n > k:
            raise ValueError("top_k must not exceed retrieve_top_k")
        raw = await asyncio.to_thread(self.retrieval.search,args.query,
            retrieve_top_k=k,rerank_top_k=n,use_reranker=spec.use_reranker)
        result = SearchResponse.model_validate(raw)
        if (result.query != args.query or result.use_reranker != spec.use_reranker
                or result.requested_retrieve_top_k != k or result.requested_result_top_k != n
                or len(result.results)>n or len(result.results)>result.candidate_count
                or result.candidate_count>k):
            raise ValueError("Retrieval tool returned inconsistent metadata")
        return result
