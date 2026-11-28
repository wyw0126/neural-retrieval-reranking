import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.exceptions import LLMRequestError, LLMResponseError
from app.llm.schemas import ChatResult, TokenUsage
from app.main import create_app
from app.rag.citations import GroundingVerdict
from app.rag.schemas import RAGAnswer


class FakeSearch:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return SimpleNamespace(results=[SimpleNamespace(
            passage_id=42, text="ML learns patterns from data.",
            retrieval_score=.5, rerank_score=2., retrieval_rank=1, rank=1)],
            candidate_count=1, timings_ms={"total": 1.})


class FakeLLM:
    def __init__(self, error=None, delay=0., invalid_citation=False, refusal=False):
        self.error, self.delay = error, delay
        self.invalid_citation, self.refusal = invalid_citation, refusal
        self.entered = self.closed = 0
        self.calls = []

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *exc):
        self.closed += 1

    async def chat_json(self, messages, schema, *, max_message_bytes=None):
        self.calls.append(schema)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        if schema is GroundingVerdict:
            parsed = schema(verdict="supported", reason="supported")
        elif self.refusal:
            parsed = RAGAnswer(answer="Insufficient evidence.", citations=[], insufficient_evidence=True)
        else:
            cid = 9 if self.invalid_citation else 1
            parsed = RAGAnswer(answer=f"ML learns patterns [{cid}].",
                               citations=[cid], insufficient_evidence=False)
        return parsed, ChatResult(content=parsed.model_dump_json(), model="fake",
            finish_reason="stop", elapsed_ms=1., attempts=1,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))


def make_app(llm=None, **kwargs):
    cfg = Settings(_env_file=None, rag_enabled=True, **kwargs)
    engine = FakeSearch()
    llm = llm or FakeLLM()
    return create_app(cfg, engine_factory=lambda _: engine, llm_factory=lambda _: llm), engine, llm


def test_lifespan_and_reuse():
    app, engine, llm = make_app(rag_semantic_validation=True)
    assert app.state.rag_engine is None and llm.entered == 0
    with TestClient(app) as client:
        assert app.state.rag_engine.retrieval is engine
        assert client.get("/rag/health").json()["semantic_validation"] is True
        for _ in range(2):
            r = client.post("/rag", json={"query": "ML?"})
            assert r.status_code == 200, r.text
            assert r.json()["cited_evidence"][0]["passage_id"] == 42
            assert r.json()["citation_report"]["semantic_status"] == "supported"
            assert r.json()["judge_usage"]["total_tokens"] == 15
            assert r.headers["x-request-id"]
        assert llm.entered == 1 and len(llm.calls) == 4
    assert llm.closed == 1 and app.state.rag_engine is None


def test_disabled():
    app = create_app(Settings(_env_file=None, rag_enabled=False),
                     engine_factory=lambda _: FakeSearch(),
                     llm_factory=lambda _: pytest.fail("LLM must not initialize"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/rag/health").status_code == 503
        r = client.post("/rag", json={"query": "ML?"})
        assert r.status_code == 503 and r.json()["code"] == "RAG_UNAVAILABLE"


@pytest.mark.parametrize("payload", [
    {"query":" "}, {"query":"ML?","retrieve_top_k":0},
    {"query":"ML?","retrieve_top_k":2,"rerank_top_k":3},
    {"query":"ML?","use_reranker":"false"}, {"query":"ML?","unknown":1},
    {"query":"ML?","rerank_top_k":101},
])
def test_bad_input(payload):
    app, engine, llm = make_app()
    with TestClient(app) as client:
        r = client.post("/rag", json=payload)
        assert r.status_code == 422
        assert r.json()["request_id"] == r.headers["x-request-id"]
    assert not engine.calls and not llm.calls


@pytest.mark.parametrize("exc,status,code", [
    (LLMRequestError(429),503,"LLM_UNAVAILABLE"),
    (LLMRequestError(401),502,"LLM_REQUEST_FAILED"),
    (LLMResponseError("private provider text"),502,"LLM_RESPONSE_INVALID"),
])
def test_upstream_failure(exc,status,code):
    app, _, _ = make_app(FakeLLM(error=exc))
    with TestClient(app) as client:
        r = client.post("/rag", json={"query":"ML?"})
        assert r.status_code == status and r.json()["code"] == code
        assert "private provider text" not in r.text
        assert r.json()["request_id"] == r.headers["x-request-id"]


def test_invalid_citation():
    app, _, _ = make_app(FakeLLM(invalid_citation=True))
    with TestClient(app) as client:
        r = client.post("/rag", json={"query":"ML?"})
        assert r.status_code == 502 and r.json()["code"] == "RAG_CITATION_INVALID"


def test_refusal_success():
    app, _, _ = make_app(FakeLLM(refusal=True))
    with TestClient(app) as client:
        r = client.post("/rag", json={"query":"ML?"})
        assert r.status_code == 200
        assert r.json()["answer"]["insufficient_evidence"] is True


def test_budget():
    app, _, llm = make_app(rag_prompt_budget_bytes=1024)
    with TestClient(app) as client:
        r = client.post("/rag", json={"query":"ML?"})
        assert r.status_code == 422 and r.json()["code"] == "RAG_PROMPT_BUDGET"
    assert not llm.calls


def test_timeout_and_drain():
    app, _, llm = make_app(FakeLLM(delay=.15), rag_request_timeout=.01,
                          rag_max_concurrent_requests=1)
    with TestClient(app) as client:
        r = client.post("/rag", json={"query":"ML?"})
        assert r.status_code == 504
        # Hold the slot until the shielded work completes.
        r = client.post("/rag", json={"query":"ML?"})
        assert r.status_code == 429 and r.headers["retry-after"] == "1"
    assert llm.closed == 1 and not app.state.rag_tasks


def test_openapi():
    app, _, _ = make_app()
    schema = app.openapi()["paths"]["/rag"]["post"]
    assert schema["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/RAGResponse")
    for status in ("422","429","502","503","504"):
        assert schema["responses"][status]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")
