from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.config import Settings
from app.retrieval.document_store import Document
from app.retrieval import reranker


class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        assert isinstance(kwargs["activation_fn"], torch.nn.Identity)
        assert kwargs["local_files_only"]
        self.max_length = kwargs["max_length"]
        self.model = SimpleNamespace(config=SimpleNamespace(num_labels=1), eval=lambda: None)
        self.calls = []

    def predict(self, pairs, **kwargs):
        assert isinstance(kwargs["activation_fn"], torch.nn.Identity)
        assert torch.is_inference_mode_enabled()
        self.calls.append((pairs, kwargs))
        return np.array([float(text) for _, text in pairs])


@pytest.fixture
def ranker(monkeypatch):
    monkeypatch.setattr(reranker, "CrossEncoder", FakeCrossEncoder)
    return reranker.Reranker(Settings(_env_file=None, model_device="cpu",
                                     max_length=256, rerank_batch_size=2,
                                     retrieve_top_k=100, rerank_top_k=2))


def test_stable_sort_and_all_candidates(ranker):
    docs = [Document(8, "-3"), Document(5, "4"), Document(2, "4"), Document(9, "1")]
    hits = ranker.rerank("question", docs)
    assert [h.passage_id for h in hits] == [5, 2]
    assert [h.rank for h in hits] == [1, 2]
    assert [h.original_rank for h in hits] == [2, 3]
    assert [h.rerank_score for h in hits] == [4.0, 4.0]
    assert [h.text for h in hits] == ["4", "4"]
    pairs, kwargs = ranker.model.calls[0]
    assert pairs == [("question", d.text) for d in docs]
    assert kwargs["batch_size"] == 2


def test_raw_negative_and_k_cap(ranker):
    hits = ranker.rerank("q", [Document(1, "-3")], top_k=10)
    assert hits[0].rerank_score == -3.0


def test_empty_candidates(ranker):
    assert ranker.rerank("q", []) == []
    assert not ranker.model.calls


@pytest.mark.parametrize("k", [0, 201, True, 1.5])
def test_invalid_k(ranker, k):
    with pytest.raises(ValueError):
        ranker.rerank("q", [Document(1, "1")], k)


@pytest.mark.parametrize("query,docs", [
    (" ", [Document(1, "1")]),
    ("q", [Document(1, " ")]),
    ("q", [Document(1, "1"), Document(1, "2")]),
    ("q", [Document(-1, "1")]),
    ("q", [Document(i, "1") for i in range(201)]),
])
def test_invalid_input(ranker, query, docs):
    with pytest.raises(ValueError):
        ranker.rerank(query, docs)


def test_column_scores(ranker, monkeypatch):
    monkeypatch.setattr(ranker.model, "predict", lambda *a, **k: np.array([[2.0], [3.0]]))
    assert [h.passage_id for h in ranker.rerank("q", [Document(1, "a"), Document(2, "b")])] == [2, 1]


@pytest.mark.parametrize("scores", [np.array([1, 2]), np.array([float("nan")]), np.array([float("inf")])])
def test_invalid_scores(ranker, monkeypatch, scores):
    monkeypatch.setattr(ranker.model, "predict", lambda *a, **k: scores)
    with pytest.raises(RuntimeError):
        ranker.rerank("q", [Document(1, "a")])
