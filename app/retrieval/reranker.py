"""Online Cross-Encoder reranking, matching the offline scoring protocol."""
from dataclasses import dataclass
from numbers import Integral
from threading import Lock

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from app.config import Settings
from app.retrieval.document_store import Document


@dataclass(frozen=True)
class RerankedDocument:
    passage_id: int
    text: str
    rerank_score: float
    rank: int
    original_rank: int


class Reranker:
    """Load a single local model and reuse it for batched pair scoring."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = Lock()
        self.model = CrossEncoder(
            str(settings.model_reranker),
            device=settings.model_device,
            max_length=settings.max_length,
            activation_fn=torch.nn.Identity(),
            local_files_only=True,
        )
        if self.model.max_length != settings.max_length:
            raise RuntimeError("Cross-Encoder max_length mismatch")
        if self.model.model.config.num_labels != 1:
            raise ValueError("Cross-Encoder must have exactly one output label")
        self.model.model.eval()

    def rerank(
        self, query: str, documents: list[Document], top_k: int | None = None,
    ) -> list[RerankedDocument]:
        """Score every candidate, then return the stable descending Top K."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        k = self.settings.rerank_top_k if top_k is None else top_k
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= 200:
            raise ValueError("top_k must be an integer between 1 and 200")
        if len(documents) > 200:
            raise ValueError("At most 200 candidate documents are supported")
        if not documents:
            return []
        seen = set()
        for doc in documents:
            if not isinstance(doc, Document):
                raise TypeError("Candidates must be Document objects")
            if (isinstance(doc.passage_id, bool)
                or not isinstance(doc.passage_id, Integral)
                or not 0 <= doc.passage_id <= 2**63 - 1):
                raise ValueError("Invalid passage ID")
            if not isinstance(doc.text, str) or not doc.text.strip():
                raise ValueError("Document text must be non-empty")
            if doc.passage_id in seen:
                raise ValueError("Duplicate candidate passage ID")
            seen.add(doc.passage_id)
        pairs = [(query, doc.text) for doc in documents]
        with self._lock, torch.inference_mode():
            scores = self.model.predict(
                pairs,
                batch_size=self.settings.rerank_batch_size,
                activation_fn=torch.nn.Identity(),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        scores = np.asarray(scores)
        if scores.shape == (len(documents), 1):
            scores = scores[:, 0]
        if scores.shape != (len(documents),):
            raise RuntimeError(f"Unexpected Cross-Encoder score shape: {scores.shape}")
        if not np.isfinite(scores).all():
            raise RuntimeError("Non-finite Cross-Encoder scores")
        order = np.argsort(-scores, kind="stable")[:int(k)]
        return [
            RerankedDocument(
                passage_id=int(documents[int(i)].passage_id),
                text=documents[int(i)].text,
                rerank_score=float(scores[int(i)]),
                rank=rank,
                original_rank=int(i) + 1,
            )
            for rank, i in enumerate(order, start=1)
        ]
