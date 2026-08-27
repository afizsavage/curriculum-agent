"""DeepSeek Responses API provider (POST /responses)."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx

from app.config import Settings
from app.exceptions import ConfigurationError, LLMProviderError, LLMTimeoutError
from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCallRequest
from app.llm.json_utils import parse_llm_json
from app.logging_utils import get_logger

logger = get_logger(__name__)


class DeepSeekResponsesProvider(LLMProvider):
    """DeepSeek via the OpenAI-compatible Responses API.

    See https://api-docs.deepseek.com/guides/responses_api/
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ConfigurationError("LLM_API_KEY is required for deepseek provider")
        self._settings = settings
        self._model = settings.llm_model
        base_url = settings.llm_base_url.rstrip("/")
        if not base_url or base_url in {
            "https://api.openai.com/v1",
            "https://api.openai.com",
        }:
            base_url = self.DEFAULT_BASE_URL
        # Responses API lives at /responses on the root base URL (not /v1).
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
        )
        self.last_token_usage: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        data = self._create_response(
            messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return LLMResponse(
            content=self._extract_output_text(data),
            model=data.get("model") or self._model,
            reasoning=self._extract_reasoning_items(data),
            raw=data,
        )

    def generate_structured(
        self,
        messages: list[LLMMessage],
        *,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        text_formats = [
            {
                "type": "json_schema",
                "name": schema.get("title") or "response",
                "schema": schema,
                "strict": False,
            },
            {"type": "json_object"},
        ]
        last_error: Exception | None = None
        for text_format in text_formats:
            try:
                data = self._create_response(
                    messages,
                    temperature=temperature,
                    max_output_tokens=8192,
                    text={"format": text_format},
                    # Disable thinking so multi-step agents don't need reasoning replay.
                    reasoning={"effort": "none"},
                )
                content = self._extract_output_text(data)
                if not content:
                    raise LLMProviderError("Empty structured response from LLM")
                try:
                    return parse_llm_json(content)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "deepseek.structured_json_parse_failed preview=%r",
                        content[:240],
                    )
                    # Fall through to the next text format / raise below.
                    last_error = LLMProviderError("LLM returned invalid JSON")
                    last_error.__cause__ = exc
                    continue
            except LLMProviderError as exc:
                last_error = exc
                detail = str(exc).lower()
                if "format" not in detail and "json" not in detail and "400" not in detail:
                    raise
                continue
        if last_error is not None:
            raise last_error
        raise LLMProviderError("LLM returned invalid JSON")

    def generate_with_tools(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        responses_tools = [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters")
                or {"type": "object", "properties": {}},
            }
            for tool in tools
        ]
        data = self._create_response(
            messages,
            temperature=temperature,
            tools=responses_tools,
            tool_choice="auto",
            # Thinking is on by default; with tools, DeepSeek requires every
            # reasoning_text to be replayed. Disable thinking for tool loops.
            reasoning={"effort": "none"},
        )
        return LLMResponse(
            content=self._extract_output_text(data),
            tool_calls=self._extract_tool_calls(data),
            reasoning=self._extract_reasoning_items(data),
            model=data.get("model") or self._model,
            raw=data,
        )

    def _create_response(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        text: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instructions, input_items = messages_to_responses_payload(messages)
        body: dict[str, Any] = {
            "model": self._model,
            "temperature": temperature,
        }
        if instructions:
            body["instructions"] = instructions
        if input_items:
            body["input"] = input_items
        elif not instructions:
            raise LLMProviderError("Responses API requires instructions or input")
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if text is not None:
            body["text"] = text
        if reasoning is not None:
            body["reasoning"] = reasoning

        data = self._post("/responses", body)
        usage = data.get("usage")
        if isinstance(usage, dict):
            self.last_token_usage = usage
        return data

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=body)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"LLM request failed: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text.strip()
            if len(detail) > 300:
                detail = detail[:300] + "..."
            message = f"LLM provider returned HTTP {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise LLMProviderError(message)
        try:
            return response.json()
        except ValueError as exc:
            raise LLMProviderError("LLM provider returned non-JSON") from exc

    @staticmethod
    def _extract_output_text(data: dict[str, Any]) -> str | None:
        if isinstance(data.get("output_text"), str) and data["output_text"]:
            return data["output_text"]

        chunks: list[str] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            # Some Responses payloads put text directly on output items.
            if item_type in {"output_text", "text"} and item.get("text"):
                chunks.append(str(item["text"]))
                continue
            if item_type not in {None, "message"}:
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content)
                continue
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text"} and part.get("text"):
                    chunks.append(str(part["text"]))
                elif isinstance(part.get("content"), str) and part["content"].strip():
                    chunks.append(str(part["content"]))
        return "".join(chunks) if chunks else None

    @staticmethod
    def _extract_reasoning_items(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Capture reasoning/thinking items that must be replayed on follow-up turns.

        DeepSeek returns HTTP 400 if thinking-mode reasoning_text is omitted from
        the next Responses API request after a tool call.
        """
        items: list[dict[str, Any]] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            # Prefer a near-verbatim copy; DeepSeek validates reasoning continuity.
            replay = {
                key: value
                for key, value in item.items()
                if key not in {"status"}
            }
            content = replay.get("content")
            if not content:
                text = item.get("text")
                if not text:
                    summary_bits = [
                        str(part.get("text"))
                        for part in (item.get("summary") or [])
                        if isinstance(part, dict) and part.get("text")
                    ]
                    text = "\n".join(summary_bits) if summary_bits else None
                if text:
                    replay["content"] = [{"type": "reasoning_text", "text": text}]
            if replay.get("content") or replay.get("encrypted_content"):
                items.append(replay)
        return items

    @staticmethod
    def _extract_tool_calls(data: dict[str, Any]) -> list[ToolCallRequest]:
        calls: list[ToolCallRequest] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue
            args_raw = item.get("arguments") or "{}"
            try:
                arguments = (
                    json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                )
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCallRequest(
                    id=str(item.get("call_id") or item.get("id") or uuid4()),
                    name=str(item.get("name") or ""),
                    arguments=arguments,
                )
            )
        return calls


def messages_to_responses_payload(
    messages: list[LLMMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Map chat-style LLMMessage list to Responses API instructions + input.

    Every function_call must have a matching function_call_output. Unpaired
    calls are dropped so DeepSeek does not return
    "No tool output found for tool call ...".
    """
    instructions_parts: list[str] = []
    pending_calls: list[dict[str, Any]] = []
    outputs_by_id: dict[str, str] = {}
    input_items: list[dict[str, Any]] = []

    def flush_tool_pairs() -> None:
        nonlocal pending_calls
        if not pending_calls:
            return
        for call in pending_calls:
            call_id = str(call.get("call_id") or "")
            if not call_id or call_id not in outputs_by_id:
                continue
            input_items.append(call)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": outputs_by_id[call_id],
                }
            )
            outputs_by_id.pop(call_id, None)
        pending_calls = []

    for message in messages:
        role = message.role
        if role == "system":
            if message.content:
                instructions_parts.append(message.content)
            continue

        if role == "user":
            flush_tool_pairs()
            input_items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": message.content or "",
                }
            )
            continue

        if role == "assistant":
            flush_tool_pairs()
            # Thinking-mode reasoning must be replayed adjacent to the assistant turn.
            for reasoning_item in message.reasoning or []:
                if isinstance(reasoning_item, dict) and reasoning_item.get("type") == "reasoning":
                    input_items.append(reasoning_item)
            # Emit text before function_calls so nothing sits between a call and
            # its output (Responses API pairing requirement).
            if message.content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": message.content,
                    }
                )
            if message.tool_calls:
                for raw in message.tool_calls:
                    fn = raw.get("function") or {}
                    args = fn.get("arguments")
                    if args is None:
                        args = raw.get("arguments")
                    if not isinstance(args, str):
                        args = json.dumps(args or {})
                    pending_calls.append(
                        {
                            "type": "function_call",
                            "call_id": str(raw.get("id") or uuid4()),
                            "name": str(fn.get("name") or raw.get("name") or ""),
                            "arguments": args,
                        }
                    )
            continue

        if role == "tool":
            call_id = message.tool_call_id or ""
            if call_id:
                outputs_by_id[call_id] = message.content or ""
            # Flush as soon as we have outputs for the pending batch.
            if pending_calls and all(
                str(call.get("call_id") or "") in outputs_by_id for call in pending_calls
            ):
                flush_tool_pairs()
            continue

        flush_tool_pairs()
        if message.content:
            input_items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": message.content,
                }
            )

    flush_tool_pairs()
    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, input_items
