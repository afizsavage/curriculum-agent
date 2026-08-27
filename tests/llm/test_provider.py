import json

import httpx
import pytest

from app.config import Settings
from app.exceptions import ConfigurationError, LLMProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResponse
from app.llm.deepseek import DeepSeekResponsesProvider, messages_to_responses_payload
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
        Settings(
            llm_provider="deepseek",
            llm_api_key="test-key",
            llm_model="deepseek-v4-flash",
        )
    )
    assert provider.name == "deepseek"
    assert provider.model == "deepseek-v4-flash"
    assert isinstance(provider._inner, DeepSeekResponsesProvider)  # type: ignore[attr-defined]


def test_messages_to_responses_payload_maps_tools():
    instructions, items = messages_to_responses_payload(
        [
            LLMMessage(role="system", content="Be helpful."),
            LLMMessage(role="user", content="Find fractions"),
            LLMMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_curriculum",
                            "arguments": '{"query":"fractions"}',
                        },
                    }
                ],
            ),
            LLMMessage(
                role="tool",
                tool_call_id="call_1",
                content='{"status":"success"}',
            ),
        ]
    )
    assert instructions == "Be helpful."
    assert items[0]["type"] == "message"
    assert items[0]["role"] == "user"
    assert items[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "search_curriculum",
        "arguments": '{"query":"fractions"}',
    }
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"status":"success"}',
    }


def test_messages_to_responses_payload_drops_unpaired_tool_calls():
    _, items = messages_to_responses_payload(
        [
            LLMMessage(role="user", content="q"),
            LLMMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call_keep",
                        "type": "function",
                        "function": {"name": "a", "arguments": "{}"},
                    },
                    {
                        "id": "call_drop",
                        "type": "function",
                        "function": {"name": "b", "arguments": "{}"},
                    },
                ],
            ),
            LLMMessage(role="tool", tool_call_id="call_keep", content='{"ok":true}'),
        ]
    )
    call_ids = [i.get("call_id") for i in items if i.get("type") == "function_call"]
    output_ids = [
        i.get("call_id") for i in items if i.get("type") == "function_call_output"
    ]
    assert call_ids == ["call_keep"]
    assert output_ids == ["call_keep"]


def test_messages_to_responses_payload_keeps_content_before_tool_pairs():
    _, items = messages_to_responses_payload(
        [
            LLMMessage(role="user", content="q"),
            LLMMessage(
                role="assistant",
                content="Looking that up.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_curriculum", "arguments": "{}"},
                    }
                ],
            ),
            LLMMessage(role="tool", tool_call_id="call_1", content='{"ok":true}'),
        ]
    )
    assert items[0]["role"] == "user"
    assert items[1] == {
        "type": "message",
        "role": "assistant",
        "content": "Looking that up.",
    }
    assert items[2]["type"] == "function_call"
    assert items[3]["type"] == "function_call_output"


def test_messages_to_responses_payload_replays_reasoning_before_tools():
    _, items = messages_to_responses_payload(
        [
            LLMMessage(role="user", content="q"),
            LLMMessage(
                role="assistant",
                reasoning=[
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "content": [
                            {"type": "reasoning_text", "text": "I should search."}
                        ],
                    }
                ],
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_curriculum", "arguments": "{}"},
                    }
                ],
            ),
            LLMMessage(role="tool", tool_call_id="call_1", content='{"ok":true}'),
        ]
    )
    assert items[1]["type"] == "reasoning"
    assert items[1]["content"][0]["type"] == "reasoning_text"
    assert items[2]["type"] == "function_call"
    assert items[3]["type"] == "function_call_output"


def test_deepseek_extracts_reasoning_items():
    data = {
        "output": [
            {
                "type": "reasoning",
                "id": "rs_abc",
                "content": [{"type": "reasoning_text", "text": "Plan A"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "search_curriculum",
                "arguments": "{}",
            },
        ]
    }
    items = DeepSeekResponsesProvider._extract_reasoning_items(data)
    assert len(items) == 1
    assert items[0]["id"] == "rs_abc"
    assert items[0]["content"][0]["text"] == "Plan A"



def test_deepseek_uses_responses_api():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "deepseek-v4-flash",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "answer": "Test",
                                        "evidence": [],
                                        "limitations": [],
                                        "confidence": "high",
                                    }
                                ),
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    settings = Settings(
        llm_provider="deepseek",
        llm_api_key="test-key",
        llm_model="deepseek-v4-flash",
        llm_base_url="https://api.deepseek.com",
    )
    provider = DeepSeekResponsesProvider(settings)
    provider._client = httpx.Client(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    result = provider.generate_structured(
        [
            LLMMessage(role="system", content="Return JSON."),
            LLMMessage(role="user", content="Answer please"),
        ],
        schema={"title": "GroundedAnswer", "type": "object", "properties": {}},
    )
    assert result["answer"] == "Test"
    assert captured["path"] == "/responses"
    assert "messages" not in captured["body"]
    assert captured["body"]["instructions"] == "Return JSON."
    assert captured["body"]["input"][0]["role"] == "user"
    assert captured["body"]["text"]["format"]["type"] == "json_schema"
    assert provider.last_token_usage == {"input_tokens": 10, "output_tokens": 5}


def test_deepseek_tool_calling_via_responses():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_2",
                "model": "deepseek-v4-flash",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "fc_1",
                        "name": "get_curriculum_structure",
                        "arguments": '{"grade":"CLASS_4"}',
                    }
                ],
            },
        )

    provider = DeepSeekResponsesProvider(
        Settings(
            llm_provider="deepseek",
            llm_api_key="test-key",
            llm_model="deepseek-v4-flash",
            llm_base_url="https://api.deepseek.com",
        )
    )
    provider._client = httpx.Client(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    result = provider.generate_with_tools(
        [LLMMessage(role="user", content="Primary 4 structure")],
        tools=[
            {
                "name": "get_curriculum_structure",
                "description": "structure",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    assert result.tool_calls
    assert result.tool_calls[0].name == "get_curriculum_structure"
    assert result.tool_calls[0].id == "fc_1"
    assert captured["body"]["tools"][0]["type"] == "function"
    assert captured["body"]["tools"][0]["name"] == "get_curriculum_structure"
    assert "function" not in captured["body"]["tools"][0]
    assert captured["body"]["reasoning"] == {"effort": "none"}


def test_deepseek_structured_disables_thinking():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_3",
                "model": "deepseek-v4-flash",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "answer": "ok",
                                        "evidence": [],
                                        "limitations": [],
                                        "confidence": "high",
                                    }
                                ),
                            }
                        ],
                    }
                ],
            },
        )

    provider = DeepSeekResponsesProvider(
        Settings(
            llm_provider="deepseek",
            llm_api_key="test-key",
            llm_model="deepseek-v4-flash",
            llm_base_url="https://api.deepseek.com",
        )
    )
    provider._client = httpx.Client(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    result = provider.generate_structured(
        [LLMMessage(role="user", content="Return json")],
        schema={"title": "GroundedAnswer", "type": "object", "properties": {}},
    )
    assert result["answer"] == "ok"
    assert captured["body"]["reasoning"] == {"effort": "none"}


def test_deepseek_structured_accepts_markdown_fenced_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_fence",
                "model": "deepseek-v4-flash",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "```json\n{\"answer\":\"ok\",\"evidence\":[],\"limitations\":[],\"confidence\":\"high\"}\n```",
                            }
                        ],
                    }
                ],
            },
        )

    provider = DeepSeekResponsesProvider(
        Settings(
            llm_provider="deepseek",
            llm_api_key="test-key",
            llm_model="deepseek-v4-flash",
            llm_base_url="https://api.deepseek.com",
        )
    )
    provider._client = httpx.Client(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    result = provider.generate_structured(
        [LLMMessage(role="user", content="Return json")],
        schema={"title": "GroundedAnswer", "type": "object", "properties": {}},
    )
    assert result["answer"] == "ok"


def test_provider_failures_can_be_wrapped():
    failing = FailingProvider()
    with pytest.raises(RuntimeError):
        failing.generate([LLMMessage(role="user", content="q")])
    err = LLMProviderError("LLM generate failed: upstream down")
    assert err.status_code == 502
    assert err.code == "LLM_PROVIDER_FAILURE"
