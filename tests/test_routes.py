"""Route contract tests without model or index loading."""
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app


class FakeEngine:
    def __init__(self):
        self.calls = []

    def search(self, query, *, retrieve_top_k, rerank_top_k, use_reranker):
        self.calls.append(("search", query, retrieve_top_k, rerank_top_k, use_reranker))
        return dict(query=query, use_reranker=use_reranker,
                    requested_retrieve_top_k=retrieve_top_k,
                    requested_result_top_k=rerank_top_k, candidate_count=1,
                    results=[dict(passage_id=8, text="example", retrieval_score=.8,
                                  rerank_score=4.0 if use_reranker else None,
                                  retrieval_rank=1, rank=1)],
                    timings_ms={"total": 1.0})

    def retrieve(self, query, top_k=None):
        k = 100 if top_k is None else top_k
        return self.search(query, retrieve_top_k=k, rerank_top_k=k, use_reranker=False)

    def rerank(self, query, documents, top_k=None):
        self.calls.append(("rerank", query, documents, top_k))
        return [SimpleNamespace(passage_id=d.passage_id, text=d.text, rerank_score=4.0,
                                original_rank=i, rank=i)
                for i, d in enumerate(documents, 1)][:top_k]


@pytest.fixture
def client_engine():
    engine = FakeEngine()
    app = create_app(Settings(_env_file=None, retrieve_top_k=100, rerank_top_k=10),
                     engine_factory=lambda _: engine)
    with TestClient(app) as client:
        yield client, engine


def test_search(client_engine):
    client, engine = client_engine
    res = client.post("/search", json={"query":"q"})
    assert res.status_code == 200
    assert res.json()["results"][0]["rerank_score"] == 4.0
    assert engine.calls == [("search", "q", 100, 10, True)]


def test_search_defaults_clamp(client_engine):
    client, engine = client_engine
    res = client.post("/search", json={"query":"q","retrieve_top_k":3,"use_reranker":False})
    assert res.status_code == 200
    assert engine.calls[-1] == ("search","q",3,3,False)
    assert res.json()["results"][0]["rerank_score"] is None


def test_retrieve(client_engine):
    client, engine = client_engine
    res = client.post("/retrieve", json={"query":"q","top_k":5})
    assert res.status_code == 200
    assert engine.calls[-1] == ("search","q",5,5,False)
    assert res.json()["use_reranker"] is False


def test_rerank(client_engine):
    client, engine = client_engine
    res = client.post("/rerank", json={"query":"q","documents":[{"passage_id":8,"text":"example"}],"top_k":1})
    assert res.status_code == 200
    assert res.json()["results"][0]["passage_id"] == 8
    assert engine.calls[-1][2][0].text == "example"


@pytest.mark.parametrize("path,body", [
    ("/search", {"query":"q","rerank_top_k":101}),
    ("/search", {"query":"q","retrieve_top_k":3,"rerank_top_k":4}),
    ("/retrieve", {"query":" ","top_k":5}),
    ("/retrieve", {"query":"q","top_k":"5"}),
    ("/rerank", {"query":"q","documents":[]}),
])
def test_invalid_requests(client_engine, path, body):
    client, engine = client_engine
    assert client.post(path,json=body).status_code == 422
    assert engine.calls == []


def test_unready():
    app = create_app(engine_factory=lambda _: FakeEngine())
    client = TestClient(app)
    try:
        assert client.post("/search",json={"query":"q"}).status_code == 503
    finally:
        client.close()


def test_openapi(client_engine):
    client, _ = client_engine
    paths = client.get("/openapi.json").json()["paths"]
    assert all("post" in paths[p] for p in ("/retrieve","/search","/rerank"))
