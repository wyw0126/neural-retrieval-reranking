import faiss
import numpy as np
import pytest

from app.config import Settings
from app.retrieval import dense


class FakeEncoder:
    max_seq_length = 256

    def __init__(self, *args, **kwargs):
        pass

    def eval(self):
        return self

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, queries, **kwargs):
        assert kwargs["normalize_embeddings"]
        return np.array([[1, 0] if q == "first" else [0, 1] for q in queries], dtype=np.float32)


@pytest.fixture
def retriever(monkeypatch, tmp_path):
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(2))
    index.add_with_ids(np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32),
                       np.array([80, 20, 50], dtype=np.int64))
    path = tmp_path / "tiny.faiss"
    faiss.write_index(index, str(path))
    monkeypatch.setattr(dense, "SentenceTransformer", FakeEncoder)
    monkeypatch.setattr(dense, "verify_model_and_index",
                        lambda *a: ("fake", {"embedding_dim": 2, "document_count": 3}))
    settings = Settings(_env_file=None, embedding_dim=2, model_device="cpu",
                        faiss_device="cpu", faiss_index=path,
                        retrieve_top_k=3, rerank_top_k=1, max_length=256)
    return dense.DenseRetriever(settings)


def test_real_faiss_ids(retriever):
    hits = retriever.retrieve("first")
    assert [h.passage_id for h in hits] == [80, 20, 50]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.retrieval_score for h in hits] == [1, 0, -1]


def test_batch_and_cap(retriever):
    rows = retriever.retrieve_many(["first", "second"], top_k=100)
    assert len(rows[0]) == 3
    assert rows[1][0].passage_id == 20
    assert retriever.retrieve_many([]) == []


@pytest.mark.parametrize("k", [0, 201, True, 2.5])
def test_invalid_k(retriever, k):
    with pytest.raises(ValueError):
        retriever.retrieve("first", k)


def test_empty_query(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_invalid_embedding(retriever, monkeypatch):
    monkeypatch.setattr(retriever.model, "encode", lambda *a, **k: [[float("nan"), 0]])
    with pytest.raises(RuntimeError, match="Non-finite"):
        retriever.retrieve("first")


def test_gpu_position_mapping():
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
    scores, ids = dense.search(index, np.array([[1, 0]], dtype=np.float32),
                               3, np.array([80, 20], dtype=np.int64))
    assert ids.tolist() == [[80, 20, -1]]
