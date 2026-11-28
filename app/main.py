"""FastAPI application factory and retrieval engine lifecycle."""
import asyncio
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
from time import perf_counter

from fastapi import FastAPI
from app.api.dependencies import EngineDep

from app.config import Settings, load_settings
from app.schemas import HealthResponse, ModelInfoResponse, ErrorResponse
from app.api.routes import router
from app.api.errors import install_error_handling
from app.api.rag import router as rag_router
from app.llm.client import LLMClient
from app.rag.chain import RAGEngine

logger = logging.getLogger(__name__)


def build_engine(settings: Settings):
    # Lazy import: importing the ASGI app does not load model dependencies.
    from app.retrieval.engine import RetrievalEngine
    return RetrievalEngine(settings)


def create_app(
    settings: Settings | None = None,
    *,
    engine_factory: Callable | None = None,
    llm_factory: Callable | None = None,
) -> FastAPI:
    """Create an independent ASGI application; initialize on lifespan startup."""
    factory = engine_factory if engine_factory is not None else build_engine
    make_llm = llm_factory if llm_factory is not None else LLMClient

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        application.state.engine = None
        application.state.rag_engine = None
        application.state.rag_slots = None
        application.state.rag_tasks = set()
        config = settings if settings is not None else load_settings()
        application.state.settings = config
        logger.setLevel(config.log_level)
        logging.getLogger("app.api").setLevel(config.log_level)
        started = perf_counter()
        logger.info("Initializing retrieval engine")
        engine = factory(config)
        async with AsyncExitStack() as resources:
            try:
                if config.rag_enabled:
                    llm = await resources.enter_async_context(make_llm(config))
                    application.state.rag_engine = RAGEngine(config, engine, llm)
                    application.state.rag_slots = asyncio.Semaphore(config.rag_max_concurrent_requests)
                application.state.engine = engine
                application.state.ready = True
                logger.info("Retrieval engine ready in %.3f seconds", perf_counter() - started)
                yield
            finally:
                application.state.ready = False
                if application.state.rag_tasks:
                    await asyncio.gather(*tuple(application.state.rag_tasks), return_exceptions=True)
                application.state.rag_engine = None
                application.state.rag_slots = None
                application.state.engine = None
                logger.info("Retrieval service stopped")

    application = FastAPI(
        title="Neural Retrieval Service",
        description="E2 dense retrieval, Cross-Encoder reranking and optional grounded RAG",
        version="0.2.0",
        lifespan=lifespan,
        responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    application.state.ready = False
    application.state.engine = None
    application.state.rag_engine = None
    application.state.rag_slots = None
    application.state.rag_tasks = set()

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(engine: EngineDep):
        return HealthResponse(status="ok", ready=True)

    @application.get("/models", response_model=ModelInfoResponse, tags=["system"])
    def models(engine: EngineDep):
        return ModelInfoResponse.model_validate(engine.model_info())

    install_error_handling(application)
    application.include_router(router, responses={422: {"model": ErrorResponse}})
    application.include_router(rag_router)
    return application


app = create_app()
