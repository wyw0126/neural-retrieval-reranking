"""Rank-preserving evidence selection; no model runtime dependencies."""
import json
import re
import unicodedata
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Iterable, Protocol

from app.config import Settings


class Hit(Protocol):
    passage_id: int
    text: str
    retrieval_score: float
    rerank_score: float | None
    retrieval_rank: int
    rank: int


def estimate_tokens(text: str) -> int:
    """Conservative UTF-8 byte budget proxy, NOT actual provider token count."""
    return len(text.encode("utf-8"))


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text: str) -> set[str]:
    # Character shingles work for both space-separated and CJK languages.
    compact = re.sub(r"\s+", " ", text)
    return {compact[i:i+5] for i in range(len(compact)-4)} if len(compact)>=5 else set()


@dataclass(frozen=True)
class Evidence:
    evidence_id: int
    passage_id: int
    text: str
    retrieval_score: float
    rerank_score: float | None
    retrieval_rank: int
    rank: int
    truncated: bool

    def prompt_item(self) -> dict:
        return {"evidence_id": self.evidence_id, "text": self.text}


@dataclass(frozen=True)
class ContextResult:
    evidence: tuple[Evidence, ...]
    context: str
    estimated_tokens: int
    budget: int
    input_count: int
    dropped: dict[str, int]
    count_method: str


def render(evidence: Iterable[Evidence]) -> str:
    return json.dumps([e.prompt_item() for e in evidence], ensure_ascii=False, separators=(",", ":"))


class ContextBuilder:
    def __init__(self, settings: Settings, *, count_tokens: Callable[[str], int] | None = None):
        self.settings = settings
        self.count_tokens = count_tokens or estimate_tokens
        self.count_method = "utf8_byte_proxy" if count_tokens is None else "custom"

    def build(self, hits: Iterable[Hit]) -> ContextResult:
        hits = list(hits)
        budget = self.settings.rag_context_budget
        per_doc = self.settings.rag_document_budget
        selected = []
        seen_ids = set()
        seen_text = set()
        seen_shingles = []
        dropped = dict(invalid=0, duplicate_id=0, duplicate_text=0, near_duplicate=0, budget=0, limit=0)
        for hit in sorted(hits, key=lambda h: h.rank):
            if (hit.passage_id < 0 or hit.rank < 1 or hit.retrieval_rank < 1
                    or not isinstance(hit.text, str) or not hit.text.strip()
                    or not isfinite(hit.retrieval_score)
                    or (hit.rerank_score is not None and not isfinite(hit.rerank_score))):
                dropped["invalid"] += 1
                continue
            if hit.passage_id in seen_ids:
                dropped["duplicate_id"] += 1
                continue
            normalized = normalize(hit.text)
            if normalized in seen_text:
                dropped["duplicate_text"] += 1
                continue
            grams = shingles(normalized)
            if grams and any(len(grams & old)/len(grams | old) >= self.settings.rag_dedup_threshold
                             for old in seen_shingles):
                dropped["near_duplicate"] += 1
                continue
            if len(selected) >= self.settings.rag_max_evidence:
                dropped["limit"] += 1
                continue
            text = hit.text.strip()
            def item(prefix):
                return Evidence(len(selected)+1, hit.passage_id, prefix, hit.retrieval_score,
                                hit.rerank_score, hit.retrieval_rank, hit.rank, prefix != text)
            def fits(prefix):
                return (self.count_tokens(prefix) <= per_doc and
                        self.count_tokens(render([*selected, item(prefix)])) <= budget)
            if not fits(text):
                # For byte proxy the predicate is monotonic; for custom tokenizers
                # binary search is a heuristic, followed by an explicit budget check.
                lo, hi = 0, len(text)
                while lo < hi:
                    mid = (lo+hi+1)//2
                    if fits(text[:mid]):
                        lo = mid
                    else:
                        hi = mid-1
                prefix = text[:lo].rstrip()
                if (not prefix or self.count_tokens(prefix) < self.settings.rag_min_document_budget
                        or not fits(prefix)):
                    dropped["budget"] += 1
                    continue
                evidence = item(prefix)
            else:
                evidence = item(text)
            selected.append(evidence)
            seen_ids.add(hit.passage_id)
            seen_text.add(normalized)
            if grams:
                seen_shingles.append(grams)
        context = render(selected)
        estimated = self.count_tokens(context)
        if estimated > budget:
            raise ValueError("Context budget too small for serialization")
        return ContextResult(tuple(selected), context, estimated, budget, len(hits), dropped, self.count_method)
