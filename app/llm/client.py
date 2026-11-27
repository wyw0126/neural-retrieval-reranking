"""Async OpenAI-compatible Chat Completions client."""
import asyncio
import json
from time import perf_counter
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm.exceptions import LLMConfigurationError, LLMRequestError, LLMResponseError
from app.llm.schemas import ChatMessage, ChatResult, TokenUsage

RETRYABLE = {408, 429, 500, 502, 503, 504}


class LLMClient:
    def __init__(self, settings: Settings, *, transport=None):
        base = settings.llm_base_url.strip().rstrip("/")
        url = urlsplit(base)
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.query or url.fragment
                or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})):
            raise LLMConfigurationError("LLM_BASE_URL must be HTTPS or local HTTP")
        if not settings.llm_model.strip():
            raise LLMConfigurationError("LLM_MODEL is required")
        key = settings.llm_api_key.get_secret_value().strip()
        if not key and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise LLMConfigurationError("LLM_API_KEY is required for remote API")
        self.settings = settings
        self.http = httpx.AsyncClient(
            base_url=base + "/", timeout=settings.llm_timeout,
            headers={"Authorization": f"Bearer {key}"} if key else {},
            transport=transport, trust_env=False, follow_redirects=False,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        await self.http.aclose()

    async def chat(self, messages: list[ChatMessage], *, json_mode=False) -> ChatResult:
        if not messages or any(not m.content.strip() for m in messages):
            raise ValueError("Messages must be nonempty")
        payload = {
            "model": self.settings.llm_model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        started = perf_counter()
        for attempt in range(1, self.settings.llm_max_retries + 2):
            try:
                response = await self.http.post("chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt > self.settings.llm_max_retries:
                    raise LLMRequestError() from None
            except httpx.HTTPError:
                raise LLMRequestError() from None
            else:
                if response.is_success:
                    return self._parse(response, attempt, started)
                if response.status_code not in RETRYABLE or attempt > self.settings.llm_max_retries:
                    raise LLMRequestError(response.status_code)
            await asyncio.sleep(min(0.5 * 2 ** (attempt - 1), 8))
        raise LLMRequestError()

    def _parse(self, response, attempt, started):
        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty content")
            usage = TokenUsage.model_validate(data["usage"]) if data.get("usage") is not None else None
            return ChatResult(content=content, model=data.get("model") or self.settings.llm_model,
                              finish_reason=choice.get("finish_reason"), usage=usage,
                              elapsed_ms=(perf_counter()-started)*1000, attempts=attempt)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise LLMResponseError("Invalid LLM response") from None

    async def chat_json(self, messages: list[ChatMessage], schema: type[BaseModel]):
        result = await self.chat(messages, json_mode=True)
        if result.finish_reason not in (None, "stop"):
            raise LLMResponseError("Incomplete or blocked LLM response")
        try:
            value = json.loads(result.content)
            if not isinstance(value, dict):
                raise ValueError("Expected object")
            parsed = schema.model_validate(value, strict=True)
        except (ValueError, ValidationError):
            raise LLMResponseError("LLM output does not match JSON schema") from None
        return parsed, result
