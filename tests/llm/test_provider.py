import json

import httpx
import pytest

from app.config import Settings
from app.exceptions import ConfigurationError, LLMProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResponse
from app.llm.provider import StubLLMProvider, build_llm_provider


class FailingProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "failing"

    @property
    def model(self) -> str:
        return "fail-model"

    def generate(self, messages, *, temperature=0.0, max_tokens=None) -> LLMResponse:
        raise RuntimeError("upstream down")

    def generate_structured(self, messages, *, schema, temperature=0.0):
        raise RuntimeError("upstream down")

    def generate_with_tools(self, messages, *, tools, temperature=0.0) -> LLMResponse:
        raise RuntimeError("upstream down")


def test_stub_provider_generate():
    provider = StubLLMProvider(model="stub-model")
    result = provider.generate([LLMMessage(role="user", content="hello")])
    assert result.content is not None
    assert "hello" in result.content
    assert result.model == "stub-model"


def test_stub_provider_structured_and_tools():
    provider = StubLLMProvider()
    structured = provider.generate_structured(
        [LLMMessage(role="user", content="x")],
        schema={"title": "Intent"},
    )
    assert structured["ok"] is True
    tools = provider.generate_with_tools(
        [
            LLMMessage(
                role="user",
                content="Question: What topics are in Primary 4 Mathematics?",
            )
        ],
        tools=[
            {
                "name": "get_curriculum_structure",
                "description": "structure",
                "parameters": {},
            }
        ],
    )
    assert tools.tool_calls
    assert tools.tool_calls[0].name == "get_curriculum_structure"


def test_build_provider_defaults_to_stub():
    provider = build_llm_provider(Settings(llm_provider="stub", llm_model="m1"))
    assert provider.name == "stub"
    assert provider.model == "m1"


def test_unknown_provider_is_configuration_error():
    with pytest.raises(ConfigurationError):
        build_llm_provider(Settings(llm_provider="anthropic", llm_api_key="x"))


def test_openai_requires_api_key():
    with pytest.raises(ConfigurationError):
        build_llm_provider(Settings(llm_provider="openai", llm_api_key=""))


def test_deepseek_requires_api_key():
    with pytest.raises(ConfigurationError):
        build_llm_provider(Settings(llm_provider="deepseek", llm_api_key=""))


def test_deepseek_provider_name():
    provider = build_llm_provider(
        Settings(llm_provider="deepseek", llm_api_key="test-key", llm_model="deepseek-chat")
    )
    assert provider.name == "deepseek"
    assert provider.model == "deepseek-chat"


def test_deepseek_structured_uses_json_object():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "Test",
                                    "evidence": [],
                                    "limitations": [],
                                    "confidence": "high",
                                }
                            )
                        }
                    }
                ],
                "model": "deepseek-v4-flash",
            },
        )

    settings = Settings(
        llm_provider="deepseek",
        llm_api_key="test-key",
        llm_model="deepseek-v4-flash",
        llm_base_url="https://api.deepseek.com",
    )
    provider = build_llm_provider(settings)
    inner = provider._inner  # type: ignore[attr-defined]
    inner._client = httpx.Client(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    result = inner.generate_structured(
        [LLMMessage(role="user", content="Return json please")],
        schema={"title": "GroundedAnswer", "type": "object", "properties": {}},
    )
    assert result["answer"] == "Test"
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_provider_failures_can_be_wrapped():
    failing = FailingProvider()
    with pytest.raises(RuntimeError):
        failing.generate([LLMMessage(role="user", content="q")])
    err = LLMProviderError("LLM generate failed: upstream down")
    assert err.status_code == 502
    assert err.code == "LLM_PROVIDER_FAILURE"
