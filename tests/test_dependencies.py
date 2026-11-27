"""Dependency injection and override tests without loading retrieval models."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import EngineDep, SettingsDep, get_engine, get_settings
from app.config import Settings
from app.main import create_app


@pytest.fixture
def probe():
    app = FastAPI()
    @app.get("/probe")
    def read(engine: EngineDep, settings: SettingsDep):
        return {"name": engine.name, "top_k": settings.retrieve_top_k}
    return app


def test_shared_state(probe):
    probe.state.ready = True
    probe.state.engine = SimpleNamespace(name="shared")
    probe.state.settings = Settings(_env_file=None, retrieve_top_k=20, rerank_top_k=10)
    with TestClient(probe) as client:
        assert client.get("/probe").json() == {"name":"shared","top_k":20}


def test_missing_state(probe):
    with TestClient(probe) as client:
        assert client.get("/probe").status_code == 503


def test_missing_settings(probe):
    probe.state.ready = True
    probe.state.engine = SimpleNamespace(name="shared")
    with TestClient(probe) as client:
        assert client.get("/probe").status_code == 503


def test_overrides(probe):
    probe.dependency_overrides[get_engine] = lambda: SimpleNamespace(name="fake")
    probe.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, retrieve_top_k=7, rerank_top_k=2)
    with TestClient(probe) as client:
        assert client.get("/probe").json() == {"name":"fake","top_k":7}
    probe.dependency_overrides.clear()


def test_request_dependency_cache():
    app = FastAPI()
    calls = []
    def override():
        calls.append(1)
        return object()
    app.dependency_overrides[get_engine] = override
    @app.get("/probe")
    def read(first: EngineDep, second: EngineDep):
        return {"same":first is second}
    with TestClient(app) as client:
        assert client.get("/probe").json() == {"same":True}
        assert len(calls) == 1
        assert client.get("/probe").status_code == 200
        assert len(calls) == 2


def test_real_route_uses_override():
    app = create_app(engine_factory=lambda _: pytest.fail("Must not initialize real engine"))
    fake = SimpleNamespace(model_info=lambda: {
        "biencoder":"fake-e2","biencoder_hash":"fake","reranker":"fake-e6",
        "model_device":"cpu","faiss_device":"cpu","embedding_dim":384,
        "document_count":3,"collection_sha256":"a"*64,"reranker_max_length":256,
    })
    app.dependency_overrides[get_engine] = lambda: fake
    client = TestClient(app)
    try:
        assert client.get("/models").json()["biencoder"] == "fake-e2"
    finally:
        client.close()
