from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import ROOT, Settings


def test_relative_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, model_biencoder="models/example")
    assert settings.model_biencoder == ROOT / "models/example"


def test_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    monkeypatch.setenv("MODEL_RERANKER", str(tmp_path))
    settings = Settings(_env_file=None)
    assert settings.model_device == "cpu"
    assert settings.model_reranker == tmp_path


def test_invalid_top_k():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieve_top_k=10, rerank_top_k=11)


def test_invalid_device():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_device="invalid")


def test_resource_check(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    index = tmp_path / "index"
    index.touch()
    store = tmp_path / "documents.sqlite3"
    settings = Settings(
        _env_file=None, model_biencoder=model, model_reranker=model,
        faiss_index=index, document_store=store,
    )
    settings.validate_resources(require_document_store=False)
    with pytest.raises(FileNotFoundError):
        settings.validate_resources()
    store.touch()
    settings.validate_resources()
