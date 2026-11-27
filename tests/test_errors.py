"""HTTP error and request correlation tests without GPU."""
import re
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import create_app
from app.config import Settings


class BrokenEngine:
    def retrieve(self, *args, **kwargs):
        raise RuntimeError("private model path /secret/checkpoint")


@pytest.fixture
def app():
    application = create_app(Settings(_env_file=None), engine_factory=lambda _: BrokenEngine())

    @application.get("/test-auth")
    def auth():
        raise HTTPException(401, "Authentication required",
                            headers={"WWW-Authenticate": "Bearer"})

    return application


def check_id(response):
    rid = response.headers["x-request-id"]
    assert re.fullmatch(r"[a-f0-9]{32}", rid)
    assert float(response.headers["x-process-time-ms"]) >= 0
    return rid


def test_success(app):
    with TestClient(app) as client:
        first = client.get("/health")
        second = client.get("/health")
        assert first.status_code == 200
        assert first.json() == {"status":"ok", "ready":True}
        assert check_id(first) != check_id(second)


def test_validation(app):
    with TestClient(app) as client:
        response = client.post("/retrieve", json={"query":"private input","top_k":"bad"})
        assert response.status_code == 422
        data = response.json()
        assert data["code"] == "VALIDATION_ERROR"
        assert data["request_id"] == check_id(response)
        assert data["details"][0]["location"] == ["body","top_k"]
        assert "private input" not in response.text and '"bad"' not in response.text


def test_malformed_json(app):
    with TestClient(app) as client:
        response = client.post("/search", content="{broken",
                               headers={"Content-Type":"application/json"})
        assert response.status_code == 422
        assert response.json()["code"] == "VALIDATION_ERROR"
        assert response.json()["request_id"] == check_id(response)


@pytest.mark.parametrize("path,method,expected,code", [
    ("/unknown","GET",404,"NOT_FOUND"),
    ("/health","POST",405,"METHOD_NOT_ALLOWED"),
    ("/test-auth","GET",401,"UNAUTHORIZED"),
])
def test_http_errors(app, path, method, expected, code):
    with TestClient(app) as client:
        response = client.request(method,path)
        assert response.status_code == expected
        assert response.json()["code"] == code
        assert response.json()["request_id"] == check_id(response)
        if expected == 405:
            assert "GET" in response.headers["allow"]
        if expected == 401:
            assert response.headers["www-authenticate"] == "Bearer"


def test_internal_failure(app):
    with TestClient(app) as client:
        response = client.post("/retrieve",json={"query":"test"})
        assert response.status_code == 500
        assert response.json()["code"] == "INTERNAL_ERROR"
        assert response.json()["request_id"] == check_id(response)
        assert "secret" not in response.text


def test_unready(app):
    client = TestClient(app)
    try:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["code"] == "SERVICE_UNAVAILABLE"
        assert response.json()["request_id"] == check_id(response)
    finally:
        client.close()


def test_request_id_not_trusted(app):
    with TestClient(app) as client:
        response = client.get("/health",headers={"X-Request-ID":"external"})
        assert check_id(response) != "external"
