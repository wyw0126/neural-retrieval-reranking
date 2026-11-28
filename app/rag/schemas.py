"""Structured generation contract for evidence-grounded RAG."""
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

EvidenceID = Annotated[int, Field(strict=True, ge=1)]


class RAGAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]
    citations: list[EvidenceID] = Field(max_length=200)
    insufficient_evidence: bool

    @model_validator(mode="after")
    def validate_contract(self):
        if self.insufficient_evidence and self.citations:
            raise ValueError("Refusal must not cite evidence")
        if not self.insufficient_evidence and not self.citations:
            raise ValueError("Supported answer requires citations")
        if len(self.citations) != len(set(self.citations)):
            raise ValueError("Duplicate citations")
        return self
