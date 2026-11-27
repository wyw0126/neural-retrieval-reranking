"""Shared FastAPI dependencies without loading model runtime on import."""
from typing import Annotated, Protocol

from fastapi import Depends, HTTPException, Request

from app.config import Settings
from app.retrieval.document_store import Document


class EngineProtocol(Protocol):
    """Structural contract for real and test retrieval engines."""

    def retrieve(self, query: str, top_k: int | None = None): ...
    def search(self, query: str, *, retrieve_top_k: int | None = None,
               rerank_top_k: int | None = None, use_reranker: bool = True): ...
    def rerank(self, query: str, documents: list[Document], top_k: int | None = None): ...
    def model_info(self) -> dict: ...


async def get_engine(request: Request) -> EngineProtocol:
    state = request.app.state
    engine = getattr(state, "engine", None)
    if not getattr(state, "ready", False) or engine is None:
        raise HTTPException(status_code=503, detail="Retrieval engine not ready")
    return engine


async def get_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if not getattr(request.app.state, "ready", False) or settings is None:
        raise HTTPException(status_code=503, detail="Retrieval engine not ready")
    return settings


EngineDep = Annotated[EngineProtocol, Depends(get_engine)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
