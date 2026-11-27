"""HTTP/lifespan tests with an injected lightweight engine."""
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeEngine:
    def model_info(self):
        return {
            "biencoder": "e2", "biencoder_hash": "fake",
            "reranker": "e6-a", "model_device": "cpu", "faiss_device": "cpu",
            "embedding_dim": 384, "document_count": 8841823,
            "collection_sha256": "a" * 64, "reranker_max_length": 256,
        }


@pytest.fixture
def settings():
    return Settings(_env_file=None, model_device="cpu")


def test_lifecycle_once(settings):
    calls = []
    def factory(config):
        calls.append(config)
        return FakeEngine()
    app = create_app(settings, engine_factory=factory)
    assert not calls
    assert app.state.engine is None
    assert not app.state.ready
    with TestClient(app) as client:
        engine = app.state.engine
        assert app.state.settings is settings
        assert client.get("/health").json() == {"status": "ok", "ready": True}
        assert client.get("/health").status_code == 200
        assert client.get("/models").json()["document_count"] == 8841823
        assert app.state.engine is engine
        assert calls == [settings]
    assert not app.state.ready
    assert app.state.engine is None


def test_startup_failure(settings):
    def fail(config):
        raise RuntimeError("cannot load engine")
    app = create_app(settings, engine_factory=fail)
    with pytest.raises(RuntimeError, match="cannot load engine"):
        with TestClient(app):
            pass
    assert not app.state.ready
    assert app.state.engine is None


def test_unready(settings):
    app = create_app(settings, engine_factory=lambda _: FakeEngine())
    # Without the context manager TestClient does not run lifespan.
    client = TestClient(app)
    try:
        assert client.get("/health").status_code == 503
        assert client.get("/models").status_code == 503
    finally:
        client.close()


def test_openapi(settings):
    app = create_app(settings, engine_factory=lambda _: FakeEngine())
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
        assert "/health" in schema["paths"]
        assert "/models" in schema["paths"]
        assert client.get("/docs").status_code == 200


def test_independent_apps(settings):
    first = create_app(settings, engine_factory=lambda _: FakeEngine())
    second = create_app(settings, engine_factory=lambda _: FakeEngine())
    with TestClient(first), TestClient(second):
        assert first.state.engine is not second.state.engine
    assert first.state.engine is None and second.state.engine is None
