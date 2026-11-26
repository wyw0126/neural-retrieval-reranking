from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from app.schemas import (
    RetrieveRequest, SearchRequest, RerankRequest, SearchResponse,
    RerankedDocumentResponse, HealthResponse,
)


def test_defaults_and_trim():
    req = SearchRequest(query="  query  ")
    assert req.query == "query"
    assert req.retrieve_top_k is None
    assert req.rerank_top_k is None
    assert req.use_reranker is True


@pytest.mark.parametrize("kwargs", [
    {"query": " "}, {"query": "x"*2049}, {"query": 123},
    {"query": "a", "top_k": 0}, {"query": "a", "top_k": 201},
    {"query": "a", "top_k": True}, {"query": "a", "top_k": "10"},
    {"query": "a", "typo": 1},
])
def test_invalid_retrieve(kwargs):
    with pytest.raises(ValidationError):
        RetrieveRequest(**kwargs)


def test_search_k():
    with pytest.raises(ValidationError):
        SearchRequest(query="q", retrieve_top_k=5, rerank_top_k=10)
    assert SearchRequest(query="q", retrieve_top_k=5).rerank_top_k is None


def test_strict_boolean():
    with pytest.raises(ValidationError):
        SearchRequest(query="q", use_reranker="false")


def test_rerank_preserves_text():
    req = RerankRequest(query="q", documents=[{"passage_id": 1, "text": "  text  "}])
    assert req.documents[0].text == "  text  "


@pytest.mark.parametrize("docs", [
    [], [{"passage_id": 1, "text": " "}],
    [{"passage_id": -1, "text": "a"}],
    [{"passage_id": True, "text": "a"}],
    [{"passage_id": 1, "text": "a"}, {"passage_id": 1, "text": "b"}],
    [{"passage_id": 1, "text": "a"*20001}],
    [{"passage_id": i, "text": "a"} for i in range(201)],
])
def test_invalid_documents(docs):
    with pytest.raises(ValidationError):
        RerankRequest(query="q", documents=docs)


def test_response_from_dataclass():
    @dataclass
    class Hit:
        passage_id: int = 1
        text: str = "text"
        retrieval_score: float = .8
        rerank_score: float | None = None
        retrieval_rank: int = 1
        rank: int = 1

    @dataclass
    class Result:
        query: str = "q"
        use_reranker: bool = False
        requested_retrieve_top_k: int = 1
        requested_result_top_k: int = 1
        candidate_count: int = 1
        results: tuple = ( )
        timings_ms: dict | None = None

    model = SearchResponse.model_validate(Result(results=(Hit(),), timings_ms={"total": 1.2}))
    assert model.model_dump(mode="json")["results"][0]["rerank_score"] is None


def test_invalid_score():
    with pytest.raises(ValidationError):
        RerankedDocumentResponse(passage_id=1, text="a", rerank_score=float("nan"), rank=1, original_rank=1)


def test_health():
    assert HealthResponse(status="ok", ready=True).ready
