import asyncio
import httpx
import pytest
from pydantic import BaseModel
from app.config import Settings
from app.llm.client import LLMClient
from app.llm.schemas import ChatMessage
from app.llm.exceptions import LLMConfigurationError, LLMRequestError, LLMResponseError


def settings(**kwargs):
    return Settings(_env_file=None, llm_base_url="https://example.test/v1",
                    llm_api_key="test-key", llm_model="test-model", **kwargs)


MESSAGES = [ChatMessage(role="user", content="Return JSON.")]


def response(content='{"answer":"ok"}', finish="stop"):
    return httpx.Response(200, json={"model":"test-model","choices":[
        {"message":{"content":content},"finish_reason":finish}],
        "usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}})


class Answer(BaseModel):
    answer: str


def test_success():
    async def run():
        def handler(request):
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer test-key"
            return response()
        async with LLMClient(settings(),transport=httpx.MockTransport(handler)) as client:
            parsed, result = await client.chat_json(MESSAGES,Answer)
            assert parsed.answer == "ok" and result.usage.total_tokens == 5
            assert result.attempts == 1
        assert client.http.is_closed
    asyncio.run(run())


def test_retry(monkeypatch):
    async def no_sleep(_): pass
    monkeypatch.setattr("app.llm.client.asyncio.sleep",no_sleep)
    calls=[]
    def handler(request):
        calls.append(1)
        return httpx.Response(429) if len(calls)==1 else response()
    async def run():
        async with LLMClient(settings(),transport=httpx.MockTransport(handler)) as client:
            assert (await client.chat(MESSAGES)).attempts == 2
    asyncio.run(run())


def test_auth_no_retry():
    calls=[]
    def handler(request):
        calls.append(1)
        return httpx.Response(401,text="secret upstream response")
    async def run():
        async with LLMClient(settings(),transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(LLMRequestError) as err:
                await client.chat(MESSAGES)
            assert err.value.status_code == 401 and "secret" not in str(err.value)
    asyncio.run(run())
    assert len(calls)==1


@pytest.mark.parametrize("content,finish",[("not json","stop"),('{"answer":1}',"stop"),('{"answer":"ok"}',"length")])
def test_invalid_json(content,finish):
    async def run():
        async with LLMClient(settings(),transport=httpx.MockTransport(lambda _:response(content,finish))) as client:
            with pytest.raises(LLMResponseError):
                await client.chat_json(MESSAGES,Answer)
    asyncio.run(run())


def test_timeout():
    def handler(request):
        raise httpx.ReadTimeout("private",request=request)
    async def run():
        async with LLMClient(settings(llm_max_retries=0),transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(LLMRequestError):
                await client.chat(MESSAGES)
    asyncio.run(run())


def test_missing_config():
    with pytest.raises(LLMConfigurationError):
        LLMClient(Settings(_env_file=None,llm_base_url="",llm_model="",llm_api_key=""))


def test_secret_mask():
    assert "test-key" not in settings().model_dump_json()
