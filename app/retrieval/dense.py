"""Online E2 dense retrieval with the existing offline FAISS contract."""
from dataclasses import dataclass
from threading import Lock
from numbers import Integral

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings
from scripts.evaluation.evaluate_reranker import (
    verify_model_and_index, load_faiss, search,
)


@dataclass(frozen=True)
class RetrievalHit:
    passage_id: int
    retrieval_score: float
    rank: int


class DenseRetriever:
    """Load once, reuse for requests; serialize access to model/GPU index."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = Lock()
        self.model_hash, metadata = verify_model_and_index(
            settings.model_biencoder, settings.faiss_index
        )
        self.model = SentenceTransformer(
            str(settings.model_biencoder),
            device=settings.model_device,
            local_files_only=True,
        )
        self.model.eval()
        self.index, self._id_map, self._gpu_resources = load_faiss(
            settings.faiss_index, settings.faiss_device, settings.faiss_gpu_id
        )
        self.document_count = int(self.index.ntotal)
        dim = settings.embedding_dim
        if (
            self.model.get_sentence_embedding_dimension() != dim
            or self.index.d != dim
            or int(metadata["embedding_dim"]) != dim
        ):
            raise ValueError("Bi-Encoder/FAISS embedding dimension mismatch")
        if self.document_count <= 0 or self.document_count != int(metadata["document_count"]):
            raise ValueError("FAISS document count mismatch")
        self.max_seq_length = self.model.max_seq_length
        if self.max_seq_length != settings.max_length:
            raise ValueError("Bi-Encoder max_seq_length differs from configured max_length")
        self.index_metadata = metadata

    def _encode(self, queries: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            queries, batch_size=min(len(queries), 128),
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape != (len(queries), self.settings.embedding_dim):
            raise RuntimeError("Unexpected query embedding shape")
        if not np.isfinite(vectors).all():
            raise RuntimeError("Non-finite query embedding")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
            raise RuntimeError("Query embeddings are not unit normalized")
        return vectors

    def retrieve_many(self, queries: list[str], top_k: int | None = None) -> list[list[RetrievalHit]]:
        if not queries:
            return []
        if any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError("Queries must be non-empty strings")
        k = self.settings.retrieve_top_k if top_k is None else top_k
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= 200:
            raise ValueError("top_k must be an integer between 1 and 200")
        k = min(int(k), self.document_count)
        with self._lock:
            vectors = self._encode(queries)
            scores, ids = search(self.index, vectors, k, self._id_map)
        if scores.shape != (len(queries), k) or ids.shape != scores.shape:
            raise RuntimeError("Unexpected FAISS result shape")
        results = []
        for row_scores, row_ids in zip(scores, ids):
            hits = []
            seen = set()
            for score, pid in zip(row_scores, row_ids):
                pid = int(pid)
                if pid == -1:
                    continue
                if pid < 0 or pid in seen or not np.isfinite(score):
                    raise RuntimeError("Invalid or duplicate FAISS result")
                seen.add(pid)
                hits.append(RetrievalHit(pid, float(score), len(hits) + 1))
            results.append(hits)
        return results

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        return self.retrieve_many([query], top_k)[0]
