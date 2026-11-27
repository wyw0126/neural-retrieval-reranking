"""FastAPI application factory and retrieval engine lifecycle."""
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request

from app.config import Settings, load_settings
from app.schemas import HealthResponse, ModelInfoResponse
from app.api.routes import router

logger = logging.getLogger(__name__)


def build_engine(settings: Settings):
    # Lazy import: importing the ASGI app does not load model dependencies.
    from app.retrieval.engine import RetrievalEngine
    return RetrievalEngine(settings)


def create_app(
    settings: Settings | None = None,
    *,
    engine_factory: Callable | None = None,
) -> FastAPI:
    """Create an independent ASGI application; initialize on lifespan startup."""
    factory = engine_factory if engine_factory is not None else build_engine

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        application.state.engine = None
        config = settings if settings is not None else load_settings()
        application.state.settings = config
        logger.setLevel(config.log_level)
        started = perf_counter()
        logger.info("Initializing retrieval engine")
        engine = factory(config)
        application.state.engine = engine
        application.state.ready = True
        logger.info("Retrieval engine ready in %.3f seconds", perf_counter() - started)
        try:
            yield
        finally:
            application.state.ready = False
            application.state.engine = None
            del engine
            logger.info("Retrieval engine stopped")

    application = FastAPI(
        title="Neural Retrieval Service",
        description="E2 dense retrieval and Cross-Encoder reranking",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.ready = False
    application.state.engine = None

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request):
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="Retrieval engine not ready")
        return HealthResponse(status="ok", ready=True)

    @application.get("/models", response_model=ModelInfoResponse, tags=["system"])
    def models(request: Request):
        engine = request.app.state.engine
        if not request.app.state.ready or engine is None:
            raise HTTPException(status_code=503, detail="Retrieval engine not ready")
        return ModelInfoResponse.model_validate(engine.model_info())

    application.include_router(router)
    return application


app = create_app()
