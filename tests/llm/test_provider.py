import json

import httpx
import pytest

from app.agent.orchestrator import CurriculumQAAgent
from app.config import Settings
from app.exceptions import ConfigurationError, LLMProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResponse
from app.llm.deepseek import DeepSeekResponsesProvider, messages_to_responses_payload
from app.llm.provider import (
    OPENROUTER_DEFAULT_BASE_URL,
    OPENROUTER_FREE_MODEL,
    OpenAICompatibleProvider,
    StubLLMProvider,
    build_llm_provider,
)


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
            llm_base_url="https://openrouter.ai/api/v1",
        )
    )
    assert provider.name == "deepseek"
    assert provider.model == "deepseek-v4-flash"
    inner = provider._inner  # type: ignore[attr-defined]
    assert isinstance(inner, DeepSeekResponsesProvider)
    assert str(inner._client.base_url).rstrip("/") == "https://api.deepseek.com"


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


def _install_mock(provider, handler, *, base_url: str) -> None:
    inner = provider._inner
    inner._client = httpx.Client(
        base_url=base_url,
        headers=dict(inner._client.headers),
        transport=httpx.MockTransport(handler),
    )


def test_openai_provider_unchanged_by_openrouter_defaults():
    provider = build_llm_provider(
        Settings(
            llm_provider="openai",
            llm_api_key="openai-key",
            llm_model="gpt-4o-mini",
            llm_base_url="https://api.openai.com/v1",
            openrouter_api_key="or-key",
        )
    )
    inner = provider._inner  # type: ignore[attr-defined]
    assert isinstance(inner, OpenAICompatibleProvider)
    assert inner.name == "openai"
    assert inner.model == "gpt-4o-mini"
    assert inner._require_parameters is False
    assert str(inner._client.base_url).rstrip("/") == "https://api.openai.com/v1"
    assert inner._client.headers["authorization"] == "Bearer openai-key"


def test_openai_chat_completions_omits_openrouter_routing():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    provider = build_llm_provider(
        Settings(
            llm_provider="openai",
            llm_api_key="openai-key",
            llm_model="gpt-4o-mini",
        )
    )
    _install_mock(provider, handler, base_url="https://api.openai.com/v1")
    result = provider.generate([LLMMessage(role="user", content="hello")])
    assert result.content == "ok"
    assert captured["path"].endswith("/chat/completions")
    assert "provider" not in captured["body"]
    assert captured["body"]["model"] == "gpt-4o-mini"


def test_openrouter_requires_api_key():
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        build_llm_provider(
            Settings(
                llm_provider="openrouter",
                llm_api_key="",
                openrouter_api_key="",
            )
        )


def test_openrouter_prefers_dedicated_key_and_free_router():
    provider = build_llm_provider(
        Settings(
            llm_provider="openrouter",
            llm_model="deepseek-v4-flash",
            llm_api_key="deepseek-or-openai-key",
            llm_base_url="https://api.deepseek.com",
            openrouter_model=OPENROUTER_FREE_MODEL,
            openrouter_api_key="or-key",
            openrouter_http_referer="https://curriculum.example",
            openrouter_app_title="Curriculum QA",
        )
    )
    inner = provider._inner  # type: ignore[attr-defined]
    assert provider.name == "openrouter"
    assert provider.model == OPENROUTER_FREE_MODEL
    assert isinstance(inner, OpenAICompatibleProvider)
    assert inner._require_parameters is True
    assert str(inner._client.base_url).rstrip("/") == OPENROUTER_DEFAULT_BASE_URL
    assert inner._client.headers["authorization"] == "Bearer or-key"
    assert inner._client.headers["http-referer"] == "https://curriculum.example"
    assert inner._client.headers["x-openrouter-title"] == "Curriculum QA"


def test_openrouter_falls_back_to_llm_api_key_and_honors_explicit_model():
    provider = build_llm_provider(
        Settings(
            llm_provider="openrouter",
            llm_api_key="shared-key",
            openrouter_api_key="",
            llm_model="google/gemma-3-27b-it:free",
            openrouter_model="",
            llm_base_url="https://openrouter.ai/api/v1",
        )
    )
    inner = provider._inner  # type: ignore[attr-defined]
    assert provider.model == "google/gemma-3-27b-it:free"
    assert inner._client.headers["authorization"] == "Bearer shared-key"
    assert "http-referer" not in inner._client.headers


def test_openrouter_chat_completions_preserves_request_shape():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "some/free-model",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            },
        )

    provider = build_llm_provider(
        Settings(
            llm_provider="openrouter",
            openrouter_api_key="or-key",
            openrouter_model=OPENROUTER_FREE_MODEL,
        )
    )
    _install_mock(provider, handler, base_url=OPENROUTER_DEFAULT_BASE_URL)
    result = provider.generate(
        [LLMMessage(role="user", content="hello")],
        temperature=0.0,
        max_tokens=32,
    )
    assert result.content == "hello"
    assert result.model == "some/free-model"
    assert captured["path"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer or-key"
    body = captured["body"]
    assert body["model"] == OPENROUTER_FREE_MODEL
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 32
    assert "provider" not in body


def test_openrouter_structured_and_tools_require_supporting_endpoints():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if "tools" in body:
            return httpx.Response(
                200,
                json={
                    "model": OPENROUTER_FREE_MODEL,
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_curriculum_structure",
                                            "arguments": '{"grade":"CLASS_4"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "model": OPENROUTER_FREE_MODEL,
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "ok",
                                    "evidence": [],
                                    "limitations": [],
                                    "confidence": "high",
                                }
                            )
                        }
                    }
                ],
            },
        )

    provider = build_llm_provider(
        Settings(
            llm_provider="openrouter",
            llm_api_key="shared-key",
            openrouter_api_key="",
            openrouter_model=OPENROUTER_FREE_MODEL,
        )
    )
    _install_mock(provider, handler, base_url=OPENROUTER_DEFAULT_BASE_URL)
    structured = provider.generate_structured(
        [LLMMessage(role="user", content="Return json")],
        schema={"title": "GroundedAnswer", "type": "object", "properties": {}},
    )
    tools = provider.generate_with_tools(
        [LLMMessage(role="user", content="Primary 4 structure")],
        tools=[
            {
                "name": "get_curriculum_structure",
                "description": "structure",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    assert structured["answer"] == "ok"
    assert tools.tool_calls[0].name == "get_curriculum_structure"
    assert tools.tool_calls[0].arguments == {"grade": "CLASS_4"}
    assert captured[0]["response_format"]["type"] == "json_schema"
    assert captured[0]["response_format"]["json_schema"]["name"] == "GroundedAnswer"
    assert captured[0]["provider"] == {"require_parameters": True}
    assert captured[1]["tools"][0]["type"] == "function"
    assert captured[1]["tools"][0]["function"]["name"] == "get_curriculum_structure"
    assert captured[1]["tool_choice"] == "auto"
    assert captured[1]["provider"] == {"require_parameters": True}


def test_openrouter_agent_uses_same_provider_for_verifier_override():
    settings = Settings(
        llm_provider="openrouter",
        llm_model="deepseek-v4-flash",
        openrouter_model=OPENROUTER_FREE_MODEL,
        openrouter_api_key="or-key",
        llm_api_key="other-key",
        verifier_llm_model="meta-llama/llama-3.3-70b-instruct:free",
        agent_checkpointing_enabled=False,
    )
    agent = CurriculumQAAgent(settings=settings, checkpointer=None)
    assert agent.llm.name == "openrouter"
    assert agent.llm.model == OPENROUTER_FREE_MODEL
    assert agent.verifier_llm.name == "openrouter"
    assert agent.verifier_llm.model == "meta-llama/llama-3.3-70b-instruct:free"
    assert agent.answer_node.generator.llm.name == "openrouter"
    assert agent.answer_node.generator.llm.model == OPENROUTER_FREE_MODEL


def test_provider_failures_can_be_wrapped():
    failing = FailingProvider()
    with pytest.raises(RuntimeError):
        failing.generate([LLMMessage(role="user", content="q")])
    err = LLMProviderError("LLM generate failed: upstream down")
    assert err.status_code == 502
    assert err.code == "LLM_PROVIDER_FAILURE"
