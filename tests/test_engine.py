from types import SimpleNamespace
import pytest

from app.config import Settings
from app.retrieval import engine
from app.retrieval.dense import RetrievalHit
from app.retrieval.document_store import Document, MissingDocumentsError
from app.retrieval.reranker import RerankedDocument


@pytest.fixture
def service(monkeypatch):
    settings = Settings(_env_file=None, retrieve_top_k=3, rerank_top_k=2)
    monkeypatch.setattr(Settings, "validate_resources", lambda self: None)
    hits = [RetrievalHit(8, .9, 1), RetrievalHit(2, .8, 2), RetrievalHit(5, .7, 3)]
    dense = SimpleNamespace(document_count=3, model_hash="model", retrieve=lambda q, top_k: hits[:top_k])
    store = SimpleNamespace(document_count=3, collection_sha256=settings.collection_sha256,
                            get_many=lambda ids, strict: [Document(pid, str(pid)) for pid in ids])
    ranker = SimpleNamespace(rerank=lambda q, docs, top_k: [
        RerankedDocument(doc.passage_id, doc.text, float(10-i), i, len(docs)-i+1)
        for i, doc in enumerate(reversed(docs), 1)][:top_k])
    monkeypatch.setattr(engine, "DenseRetriever", lambda s: dense)
    monkeypatch.setattr(engine, "DocumentStore", lambda *a, **kw: store)
    monkeypatch.setattr(engine, "Reranker", lambda s: ranker)
    return engine.RetrievalEngine(settings)


def test_search(service):
    result = service.search("query")
    assert [h.passage_id for h in result.results] == [5, 2]
    assert [h.retrieval_rank for h in result.results] == [3, 2]
    assert [h.retrieval_score for h in result.results] == [.7, .8]
    assert [h.rerank_score for h in result.results] == [9., 8.]
    assert result.candidate_count == 3
    assert result.to_dict()["results"][0]["passage_id"] == 5
    assert all(v >= 0 for v in result.timings_ms.values())


def test_retrieve_only(service, monkeypatch):
    def fail(*a, **kw):
        pytest.fail("Reranker must not run")
    monkeypatch.setattr(service.reranker, "rerank", fail)
    result = service.retrieve("query", 3)
    assert [h.passage_id for h in result.results] == [8, 2, 5]
    assert all(h.rerank_score is None for h in result.results)
    assert result.timings_ms["reranking"] == 0


@pytest.mark.parametrize("kwargs", [{"retrieve_top_k":0}, {"retrieve_top_k":1,"rerank_top_k":2},
                                  {"rerank_top_k":True}, {"use_reranker":"false"}])
def test_invalid_parameters(service, kwargs):
    with pytest.raises(ValueError):
        service.search("query", **kwargs)


def test_invalid_query(service):
    with pytest.raises(ValueError):
        service.search(" ")


def test_empty(service, monkeypatch):
    monkeypatch.setattr(service.dense, "retrieve", lambda *a, **k: [])
    assert service.search("query").results == ()


def test_missing(service, monkeypatch):
    def fail(*a, **kw):
        raise MissingDocumentsError([8])
    monkeypatch.setattr(service.store, "get_many", fail)
    with pytest.raises(MissingDocumentsError):
        service.search("query")


def test_wrong_order(service, monkeypatch):
    monkeypatch.setattr(service.store, "get_many", lambda ids, strict: [Document(p,str(p)) for p in reversed(ids)])
    with pytest.raises(RuntimeError, match="order"):
        service.search("query")


def test_invalid_rerank(service, monkeypatch):
    monkeypatch.setattr(service.reranker, "rerank", lambda *a, **kw: [
        RerankedDocument(99, "bad", 5., 1, 1), RerankedDocument(2, "2", 3., 2, 2)])
    with pytest.raises(RuntimeError, match="identity"):
        service.search("query")


def test_info(service):
    assert service.model_info()["document_count"] == 3


def test_standalone_rerank(service):
    assert service.rerank("q", [Document(8, "8")], 1)[0].passage_id == 8


def test_count_mismatch(service, monkeypatch):
    monkeypatch.setattr(engine, "DocumentStore", lambda *a, **kw: SimpleNamespace(document_count=4))
    with pytest.raises(ValueError, match="count mismatch"):
        engine.RetrievalEngine(service.settings)
