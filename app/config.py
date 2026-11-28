"""Runtime configuration for the online retrieval service."""
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_biencoder: Path = Path("models/minilm-msmarco-e2-hn")
    model_reranker: Path = Path("checkpoints/e6_a_listwise/checkpoint-300")
    faiss_index: Path = Path("data/indexes/faiss/minilm_e2_flatip.faiss")
    collection_path: Path = Path("data/raw/msmarco/collection.tsv")
    document_store: Path = Path("data/processed/msmarco/documents.sqlite3")

    collection_sha256: str = Field(default="86e0c5820be6b22280337a67e54cbb626645453d9ac3c377bac45abb81c5653d", pattern=r"^[0-9a-fA-F]{64}$")

    model_device: Literal["cpu", "cuda"] = "cuda"
    faiss_device: Literal["cpu", "gpu"] = "cpu"
    faiss_gpu_id: int = Field(default=0, ge=0)
    embedding_dim: int = Field(default=384, gt=0)
    max_length: int = Field(default=256, gt=0)
    rerank_batch_size: int = Field(default=64, gt=0)
    retrieve_top_k: int = Field(default=100, ge=1, le=200)
    rerank_top_k: int = Field(default=10, ge=1, le=200)
    max_concurrent_searches: int = Field(default=1, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    llm_timeout: float = Field(default=60.0, gt=0, le=300)
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_max_tokens: int = Field(default=2048, ge=1, le=32768)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_json_retries: int = Field(default=1, ge=0, le=3)

    agent_router_enabled: bool = False
    agent_rewrite_enabled: bool = False

    rag_enabled: bool = False
    rag_max_concurrent_requests: int = Field(default=4, ge=1, le=100)
    rag_request_timeout: float = Field(default=120.0, gt=0, le=900)
    rag_semantic_validation: bool = False
    rag_prompt_budget_bytes: int = Field(default=16000, ge=1024, le=1000000)
    rag_prompt_retry_reserve_bytes: int = Field(default=2048, ge=0, le=100000)
    rag_context_budget: int = Field(default=6000, ge=64, le=100000)
    rag_document_budget: int = Field(default=1200, ge=16, le=100000)
    rag_min_document_budget: int = Field(default=64, ge=1)
    rag_max_evidence: int = Field(default=10, ge=1, le=200)
    rag_dedup_threshold: float = Field(default=0.90, gt=0, le=1)

    @field_validator(
        "model_biencoder", "model_reranker", "faiss_index",
        "collection_path", "document_store",
    )
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        value = value.expanduser()
        return (ROOT / value).resolve()

    @model_validator(mode="after")
    def validate_top_k(self):
        if self.rerank_top_k > self.retrieve_top_k:
            raise ValueError("rerank_top_k must not exceed retrieve_top_k")
        if self.rag_min_document_budget > self.rag_document_budget:
            raise ValueError("rag_min_document_budget must not exceed rag_document_budget")
        return self

    def validate_resources(self, *, require_document_store: bool = True) -> None:
        """Explicit startup check; importing this module never loads models."""
        directories = [self.model_biencoder, self.model_reranker]
        files = [self.faiss_index]
        if require_document_store:
            files.append(self.document_store)
        missing = [str(p) for p in directories if not p.is_dir()]
        missing += [str(p) for p in files if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing resources: " + ", ".join(missing))


def load_settings() -> Settings:
    return Settings()


if __name__ == "__main__":
    print(load_settings().model_dump_json(indent=2))
