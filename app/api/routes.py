"""HTTP retrieval routes; blocking inference runs in FastAPI's worker pool."""
from time import perf_counter

from fastapi import APIRouter, HTTPException
from app.api.dependencies import EngineDep, SettingsDep

from app.retrieval.document_store import Document
from app.schemas import (
    RetrieveRequest, SearchRequest, SearchResponse,
    RerankRequest, RerankResponse,
)

router = APIRouter(tags=["retrieval"])




@router.post("/retrieve", response_model=SearchResponse)
def retrieve(body: RetrieveRequest, engine: EngineDep):
    """Return dense retrieval results, without Cross-Encoder inference."""
    return SearchResponse.model_validate(engine.retrieve(body.query, top_k=body.top_k))


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, engine: EngineDep, settings: SettingsDep):
    """Retrieve candidates, optionally rerank, and return the final Top K."""
    k = settings.retrieve_top_k if body.retrieve_top_k is None else body.retrieve_top_k
    n = min(settings.rerank_top_k, k) if body.rerank_top_k is None else body.rerank_top_k
    if n > k:
        raise HTTPException(status_code=422, detail="rerank_top_k must not exceed retrieve_top_k")
    result = engine.search(
        body.query, retrieve_top_k=k, rerank_top_k=n,
        use_reranker=body.use_reranker,
    )
    return SearchResponse.model_validate(result)


@router.post("/rerank", response_model=RerankResponse)
def rerank(body: RerankRequest, engine: EngineDep):
    """Rerank caller-provided documents; no FAISS or SQLite lookup."""
    documents = [Document(doc.passage_id, doc.text) for doc in body.documents]
    started = perf_counter()
    results = engine.rerank(body.query, documents, top_k=body.top_k)
    return RerankResponse(
        query=body.query, candidate_count=len(documents),
        results=results, elapsed_ms=(perf_counter() - started) * 1000,
    )
