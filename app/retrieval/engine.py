"""Unified online retrieval engine; shared by CLI, API, RAG and Agent."""
from dataclasses import dataclass, asdict
from math import isfinite
from numbers import Integral
from threading import BoundedSemaphore
from time import perf_counter

from app.config import Settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.document_store import DocumentStore, Document
from app.retrieval.reranker import Reranker


@dataclass(frozen=True)
class SearchHit:
    passage_id: int
    text: str
    retrieval_score: float
    rerank_score: float | None
    retrieval_rank: int
    rank: int


@dataclass(frozen=True)
class SearchResponse:
    query: str
    use_reranker: bool
    requested_retrieve_top_k: int
    requested_result_top_k: int
    candidate_count: int
    results: tuple[SearchHit, ...]
    timings_ms: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class RetrievalEngine:
    """Create once per service process, not once per request."""

    def __init__(self, settings: Settings):
        self.settings = settings
        settings.validate_resources()
        self._slots = BoundedSemaphore(settings.max_concurrent_searches)
        self.store = DocumentStore(
            settings.document_store,
            expected_sha256=settings.collection_sha256,
        )
        self.dense = DenseRetriever(settings)
        if self.store.document_count != self.dense.document_count:
            raise ValueError("FAISS and document store count mismatch")
        self.reranker = Reranker(settings)

    @staticmethod
    def _validate_k(k: int) -> int:
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= 200:
            raise ValueError("top_k must be an integer between 1 and 200")
        return int(k)

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

    def model_info(self) -> dict:
        return {
            "biencoder": self.settings.model_biencoder.name,
            "biencoder_hash": self.dense.model_hash,
            "reranker": self.settings.model_reranker.name,
            "model_device": self.settings.model_device,
            "faiss_device": self.settings.faiss_device,
            "embedding_dim": self.settings.embedding_dim,
            "document_count": self.store.document_count,
            "collection_sha256": self.store.collection_sha256,
            "reranker_max_length": self.settings.max_length,
        }

    def retrieve(self, query: str, top_k: int | None = None) -> SearchResponse:
        k = self.settings.retrieve_top_k if top_k is None else top_k
        return self.search(query, retrieve_top_k=k, rerank_top_k=k, use_reranker=False)

    def rerank(self, query: str, documents: list[Document], top_k: int | None = None):
        """Score caller-supplied documents without accessing FAISS or SQLite."""
        with self._slots:
            return self.reranker.rerank(query, documents, top_k)

    def search(self, query: str, *, retrieve_top_k: int | None = None,
               rerank_top_k: int | None = None, use_reranker: bool = True) -> SearchResponse:
        self._validate_query(query)
        if not isinstance(use_reranker, bool):
            raise ValueError("use_reranker must be a boolean")
        k = self._validate_k(self.settings.retrieve_top_k if retrieve_top_k is None else retrieve_top_k)
        n = self._validate_k(self.settings.rerank_top_k if rerank_top_k is None else rerank_top_k)
        if n > k:
            raise ValueError("rerank_top_k must not exceed retrieve_top_k")
        started = perf_counter()
        with self._slots:
            acquired = perf_counter()
            hits = self.dense.retrieve(query, top_k=k)
            retrieved = perf_counter()
            ids = [hit.passage_id for hit in hits]
            if len(ids) != len(set(ids)) or len(ids) > k:
                raise RuntimeError("Invalid retrieval candidate set")
            if any(hit.rank != i or not isfinite(hit.retrieval_score)
                   for i, hit in enumerate(hits, 1)):
                raise RuntimeError("Invalid retrieval ranking")
            documents = self.store.get_many(ids, strict=True)
            looked_up = perf_counter()
            if [doc.passage_id for doc in documents] != ids:
                raise RuntimeError("Document order or identity mismatch")
            if use_reranker and documents:
                ranked = self.reranker.rerank(query, documents, top_k=n)
                hit_map = {hit.passage_id: hit for hit in hits}
                doc_map = {doc.passage_id: doc for doc in documents}
                if len(ranked) != min(n, len(ids)) or len({r.passage_id for r in ranked}) != len(ranked):
                    raise RuntimeError("Invalid reranked candidate count")
                results = []
                for i, item in enumerate(ranked, 1):
                    original = hit_map.get(item.passage_id)
                    if (original is None or item.original_rank != original.rank
                        or item.rank != i or item.text != doc_map[item.passage_id].text
                        or not isfinite(item.rerank_score)):
                        raise RuntimeError("Reranked candidate identity or rank mismatch")
                    results.append(SearchHit(item.passage_id, item.text,
                                             original.retrieval_score, item.rerank_score,
                                             original.rank, item.rank))
                if any(a.rerank_score < b.rerank_score for a, b in zip(results, results[1:])):
                    raise RuntimeError("Rerank scores not descending")
            else:
                results = [SearchHit(hit.passage_id, doc.text, hit.retrieval_score,
                                     None, hit.rank, hit.rank)
                           for hit, doc in zip(hits[:n], documents[:n])]
            finished = perf_counter()
        return SearchResponse(
            query, use_reranker, k, n, len(hits), tuple(results),
            {"queue": (acquired-started)*1000,
             "retrieval": (retrieved-acquired)*1000,
             "document_lookup": (looked_up-retrieved)*1000,
             "reranking": (finished-looked_up)*1000 if use_reranker and documents else 0.0,
             "total": (finished-started)*1000},
        )
