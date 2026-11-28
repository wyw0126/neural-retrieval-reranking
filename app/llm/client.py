"""Async OpenAI-compatible Chat Completions client."""
import asyncio
import json
from time import perf_counter
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm.exceptions import LLMConfigurationError, LLMRequestError, LLMResponseError, LLMStructuredOutputError
from app.llm.schemas import ChatMessage, ChatResult, TokenUsage

RETRYABLE = {408, 429, 500, 502, 503, 504}


def message_bytes(messages: list[ChatMessage]) -> int:
    return len(json.dumps([m.model_dump() for m in messages], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def prepare_json_messages(messages: list[ChatMessage], schema: type[BaseModel]) -> list[ChatMessage]:
    instruction = (
        "Return exactly one JSON object matching this JSON Schema. "
        "Use the declared JSON types exactly: integers are unquoted numbers, "
        "booleans are true or false, and arrays must contain the declared item types. "
        "Do not use Markdown fences or add extra fields. "
        "JSON Schema:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )
    # Merge into the initial system message for providers requiring one system turn.
    prompt = list(messages)
    if prompt[0].role == "system":
        prompt[0] = ChatMessage(role="system", content=prompt[0].content + "\n\n" + instruction)
    else:
        prompt.insert(0, ChatMessage(role="system", content=instruction))
    return prompt


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
        except ValueError:
            raise LLMResponseError("Provider returned non-JSON response; check API base URL") from None
        try:
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

    async def chat_json(self, messages: list[ChatMessage], schema: type[BaseModel], *, max_message_bytes: int | None = None):
        """Inject schema, strictly validate, and retry invalid output with safe feedback.

        Returned ChatResult aggregates timing, attempts and usage across completed
        generation calls, not just the final successful response.
        """
        if not messages or any(not m.content.strip() for m in messages):
            raise ValueError("Messages must be nonempty")
        prompt = prepare_json_messages(messages, schema)
        started = perf_counter()
        attempts = 0
        usages = []
        feedback = None
        for _ in range(self.settings.llm_json_retries + 1):
            current = list(prompt)
            if feedback is not None:
                current.append(ChatMessage(role="user", content=(
                    "The previous generation failed strict JSON validation. "
                    "Generate the complete corrected object for the original task. "
                    "Validation errors (field location and error type only): " + feedback
                )))
            if max_message_bytes is not None and message_bytes(current) > max_message_bytes:
                raise LLMStructuredOutputError("JSON request exceeds message byte budget")
            result = await self.chat(current, json_mode=True)
            attempts += result.attempts
            usages.append(result.usage)
            if result.finish_reason not in (None, "stop"):
                raise LLMResponseError("Incomplete or blocked LLM response")
            try:
                value = json.loads(result.content)
            except ValueError:
                feedback = "Invalid JSON syntax"
                continue
            if not isinstance(value, dict):
                feedback = "Expected a JSON object"
                continue
            try:
                parsed = schema.model_validate(value, strict=True)
            except ValidationError as exc:
                # Never include raw output values, validator context or error messages.
                feedback = json.dumps([
                    {"location": list(e["loc"]), "type": e["type"]}
                    for e in exc.errors(include_input=False, include_context=False, include_url=False)
                ][:20], ensure_ascii=False)
                continue
            usage = None
            if all(u is not None for u in usages):
                usage = TokenUsage(
                    prompt_tokens=sum(u.prompt_tokens for u in usages),
                    completion_tokens=sum(u.completion_tokens for u in usages),
                    total_tokens=sum(u.total_tokens for u in usages),
                )
            return parsed, result.model_copy(update={
                "usage": usage, "attempts": attempts,
                "elapsed_ms": (perf_counter() - started) * 1000,
            })
        raise LLMStructuredOutputError(
            "LLM output does not match JSON schema after bounded repair: " + (feedback or "unknown")
        )
