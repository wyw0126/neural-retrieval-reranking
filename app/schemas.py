"""HTTP request and response contracts for the retrieval service."""
from typing import Annotated, Literal

from pydantic import (
    BaseModel, ConfigDict, Field, StringConstraints, FiniteFloat,
    model_validator,
)

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]
DocumentText = Annotated[str, StringConstraints(min_length=1, max_length=20000)]
TopK = Annotated[int, Field(strict=True, ge=1, le=200)]
PassageID = Annotated[int, Field(strict=True, ge=0, le=2**63-1)]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RetrieveRequest(RequestModel):
    query: QueryText
    top_k: TopK | None = None


class SearchRequest(RequestModel):
    query: QueryText
    retrieve_top_k: TopK | None = None
    rerank_top_k: TopK | None = None
    use_reranker: bool = True

    @model_validator(mode="after")
    def check_top_k(self):
        if (self.retrieve_top_k is not None and self.rerank_top_k is not None
            and self.rerank_top_k > self.retrieve_top_k):
            raise ValueError("rerank_top_k must not exceed retrieve_top_k")
        return self


class DocumentInput(RequestModel):
    passage_id: PassageID
    text: DocumentText

    @model_validator(mode="after")
    def check_text(self):
        if not self.text.strip():
            raise ValueError("Document text must not be blank")
        return self


class RerankRequest(RequestModel):
    query: QueryText
    documents: Annotated[list[DocumentInput], Field(min_length=1, max_length=200)]
    top_k: TopK | None = None

    @model_validator(mode="after")
    def check_documents(self):
        ids = [doc.passage_id for doc in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate passage ID")
        return self


class ResponseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


class SearchHitResponse(ResponseModel):
    passage_id: PassageID
    text: str
    retrieval_score: FiniteFloat
    rerank_score: FiniteFloat | None
    retrieval_rank: Annotated[int, Field(ge=1)]
    rank: Annotated[int, Field(ge=1)]


class SearchResponse(ResponseModel):
    query: str
    use_reranker: bool
    requested_retrieve_top_k: TopK
    requested_result_top_k: TopK
    candidate_count: Annotated[int, Field(ge=0, le=200)]
    results: list[SearchHitResponse]
    timings_ms: dict[str, Annotated[FiniteFloat, Field(ge=0)]]


class RerankedDocumentResponse(ResponseModel):
    passage_id: PassageID
    text: str
    rerank_score: FiniteFloat
    rank: Annotated[int, Field(ge=1)]
    original_rank: Annotated[int, Field(ge=1)]


class RerankResponse(ResponseModel):
    query: str
    candidate_count: Annotated[int, Field(ge=1, le=200)]
    results: list[RerankedDocumentResponse]
    elapsed_ms: Annotated[FiniteFloat, Field(ge=0)]


class HealthResponse(ResponseModel):
    status: Literal["ok"]
    ready: bool


class ModelInfoResponse(ResponseModel):
    biencoder: str
    biencoder_hash: str
    reranker: str
    model_device: Literal["cpu", "cuda"]
    faiss_device: Literal["cpu", "gpu"]
    embedding_dim: Annotated[int, Field(gt=0)]
    document_count: Annotated[int, Field(gt=0)]
    collection_sha256: str
    reranker_max_length: Annotated[int, Field(gt=0)]


class ErrorResponse(ResponseModel):
    code: str
    message: str
