"""Provider-independent chat inputs and results."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatResult(BaseModel):
    content: str
    model: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    elapsed_ms: float = Field(ge=0)
    attempts: int = Field(ge=1)
