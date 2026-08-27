"""Retrieval node: LLM tool selection → curriculum tool execution → evidence."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from app.agent.state import CurriculumQAState, RetrievedContextItem
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus, ToolCallRecord
from app.curriculum.errors import CurriculumAPIError
from app.enums import AgentStatus
from app.llm.base import LLMMessage, LLMProvider, ToolCallRequest
from app.logging_utils import get_logger, log_agent_event
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a curriculum retrieval planner for the MBSSE Curriculum Q&A agent.
Select curriculum tools to gather authoritative evidence for the user's question.
Only use the provided tools. Prefer precise tools (get_curriculum_structure, get_topic,
get_learning_objectives) when grade/subject/topic are known. Use search_curriculum for
concept discovery. Do not invent curriculum facts. When enough evidence exists, stop
requesting tools and reply with a short note that retrieval is complete.

When verification feedback lists missing evidence, prioritize targeted tools that
satisfy those gaps rather than repeating broad searches already executed.
"""


def tool_call_key(name: str, arguments: dict[str, Any] | None) -> str:
    """Stable key for duplicate tool-call suppression."""
    normalized = json.dumps(arguments or {}, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{name}:{normalized}".encode()).hexdigest()[:16]
    return f"{name}:{digest}"


class RetrievalNode:
    """UNDERSTAND → RETRIEVE multi-step tool loop with hard limits."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        tools: ToolRegistry,
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.settings = settings

    def run(
        self,
        state: CurriculumQAState,
        *,
        request_id: str | None = None,
    ) -> CurriculumQAState:
        state.status = AgentStatus.RETRIEVING
        curriculum_tools = [
            t for t in self.tools.llm_tool_specs() if t.get("name") != "echo"
        ]
        if not curriculum_tools:
            state.evidence_status = EvidenceStatus.ERROR
            state.error = "No curriculum tools registered"
            state.status = AgentStatus.FAILED
            return state

        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=self._user_prompt(state),
            ),
        ]

        # Cap tool work per retrieval round while respecting global tool budget.
        remaining = self.settings.agent_max_tool_calls - state.tool_calls
        if remaining <= 0:
            self._finalize_evidence_status(state)
            state.status = AgentStatus.RETRIEVED
            return state
        round_tool_budget = max(1, min(4, remaining))
        tools_this_round = 0

        planning_steps = 0
        max_planning_steps = 4
        while (
            state.tool_calls < self.settings.agent_max_tool_calls
            and tools_this_round < round_tool_budget
            and planning_steps < max_planning_steps
        ):
            planning_steps += 1
            response = self.llm.generate_with_tools(messages, tools=curriculum_tools)
            if not response.tool_calls:
                break

            executed_calls: list[ToolCallRequest] = []
            tool_messages: list[LLMMessage] = []
            for call in response.tool_calls:
                if (
                    state.tool_calls >= self.settings.agent_max_tool_calls
                    or tools_this_round >= round_tool_budget
                ):
                    break
                key = tool_call_key(call.name, call.arguments)
                if key in state.executed_tool_keys:
                    tool_messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=call.id,
                            content=json.dumps(
                                {
                                    "tool": call.name,
                                    "status": "skipped_duplicate",
                                    "evidence_count": 0,
                                    "error": (
                                        "Identical tool call already executed; "
                                        "use a different tool or arguments."
                                    ),
                                }
                            ),
                        )
                    )
                    executed_calls.append(call)
                    continue

                self._execute_call(state, call, request_id=request_id)
                state.executed_tool_keys.append(key)
                tools_this_round += 1
                executed_calls.append(call)
                last = state.retrieval_history[-1] if state.retrieval_history else None
                tool_messages.append(
                    LLMMessage(
                        role="tool",
                        tool_call_id=call.id,
                        content=json.dumps(
                            {
                                "tool": call.name,
                                "status": last.status if last else "unknown",
                                "evidence_count": last.evidence_count if last else 0,
                                "error": last.error if last else None,
                                "sample_evidence": [
                                    e.model_dump() for e in state.evidence[-3:]
                                ],
                            }
                        ),
                    )
                )

            if not executed_calls:
                break

            messages.append(
                self._assistant_tool_message(response, tool_calls=executed_calls)
            )
            messages.extend(tool_messages)

        self._finalize_evidence_status(state)
        state.status = AgentStatus.RETRIEVED
        state.metadata["tools_used"] = list(
            dict.fromkeys(r.tool for r in state.retrieval_history)
        )
        state.metadata["evidence_count"] = len(state.evidence)
        state.metadata["evidence_status"] = state.evidence_status.value
        state.metadata["retrieval_rounds"] = state.retrieval_rounds
        log_agent_event(
            logger,
            "agent.retrieve.end",
            request_id=request_id,
            conversation_id=state.conversation_id,
            question=state.question,
            status=state.status.value,
            iteration=state.iteration,
            tool_calls=state.tool_calls,
            model=self.llm.model,
            evidence_count=len(state.evidence),
            evidence_status=state.evidence_status.value,
            retrieval_rounds=state.retrieval_rounds,
        )
        return state

    def _user_prompt(self, state: CurriculumQAState) -> str:
        filters = {
            "intent": state.intent,
            "level": state.level,
            "grade": state.grade,
            "subject": state.subject,
            "topic": state.topic,
        }
        parts = [
            f"Question: {state.question}",
            f"Known filters: {json.dumps(filters)}",
        ]
        if state.pending_missing_evidence:
            parts.append(
                "Verification feedback — missing evidence to retrieve next:\n"
                + json.dumps(
                    [
                        item.model_dump() if hasattr(item, "model_dump") else item
                        for item in state.pending_missing_evidence
                    ],
                    indent=2,
                )
            )
            parts.append(
                "Prefer targeted tools (get_topic, get_learning_objectives, "
                "get_curriculum_structure) that address the missing evidence."
            )
        elif state.retrieval_rounds > 1:
            parts.append(
                "Prior retrieval was insufficient. Expand with more targeted "
                "curriculum tools; avoid repeating identical tool calls."
            )
        else:
            parts.append("Select and call the appropriate curriculum tool(s).")
        return "\n".join(parts)

    def _execute_call(
        self,
        state: CurriculumQAState,
        call: ToolCallRequest,
        *,
        request_id: str | None,
    ) -> None:
        if call.name not in state.selected_tools:
            state.selected_tools.append(call.name)
        started = time.perf_counter()
        api_status = None
        try:
            result = self.tools.execute(call.name, **(call.arguments or {}))
            latency = (time.perf_counter() - started) * 1000
            state.bump_tool_calls()
            if result.success:
                evidence_rows = (result.data or {}).get("evidence") or []
                for row in evidence_rows:
                    item = CurriculumEvidence.model_validate(row)
                    if item.entity_id and any(
                        e.entity_id == item.entity_id for e in state.evidence
                    ):
                        continue
                    state.evidence.append(item)
                    state.retrieved_context.append(
                        RetrievedContextItem(
                            source=item.source_reference or call.name,
                            content=item.content or item.name or "",
                            metadata=item.model_dump(),
                        )
                    )
                record = ToolCallRecord(
                    tool=call.name,
                    arguments=call.arguments or {},
                    status="success",
                    evidence_count=len(evidence_rows),
                    latency_ms=round(latency, 2),
                    curriculum_api_status=200,
                )
            else:
                error_code = (
                    (result.data or {}) if isinstance(result.data, dict) else {}
                ).get("error_code")
                status = "not_found" if error_code == "CURRICULUM_NOT_FOUND" else "error"
                if error_code == "CURRICULUM_NOT_FOUND":
                    api_status = 404
                elif error_code == "CURRICULUM_TIMEOUT":
                    api_status = 504
                elif error_code in {"CURRICULUM_UNAVAILABLE", "CURRICULUM_API_ERROR"}:
                    api_status = 503
                record = ToolCallRecord(
                    tool=call.name,
                    arguments=call.arguments or {},
                    status=status,
                    error=result.error,
                    evidence_count=0,
                    latency_ms=round(latency, 2),
                    curriculum_api_status=api_status,
                )
        except CurriculumAPIError as exc:
            latency = (time.perf_counter() - started) * 1000
            state.bump_tool_calls()
            record = ToolCallRecord(
                tool=call.name,
                arguments=call.arguments or {},
                status="error",
                error=str(exc),
                evidence_count=0,
                latency_ms=round(latency, 2),
                curriculum_api_status=exc.status_code,
            )
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            state.bump_tool_calls()
            record = ToolCallRecord(
                tool=call.name,
                arguments=call.arguments or {},
                status="error",
                error=str(exc),
                evidence_count=0,
                latency_ms=round(latency, 2),
            )

        state.retrieval_history.append(record)
        log_agent_event(
            logger,
            "agent.tool.execute",
            request_id=request_id,
            conversation_id=state.conversation_id,
            tool_name=record.tool,
            tool_arguments=record.arguments,
            tool_arguments_hash=tool_call_key(record.tool, record.arguments),
            tool_status=record.status,
            tool_execution_time=record.latency_ms,
            curriculum_api_status=record.curriculum_api_status,
            evidence_count=record.evidence_count,
            error=record.error,
            iteration=state.iteration,
            retrieval_rounds=state.retrieval_rounds,
        )

    @staticmethod
    def _assistant_tool_message(
        response,
        *,
        tool_calls: list[ToolCallRequest] | None = None,
    ) -> LLMMessage:
        calls = tool_calls if tool_calls is not None else response.tool_calls
        formatted = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments or {}),
                },
            }
            for call in calls
        ]
        return LLMMessage(
            role="assistant",
            content=response.content,
            tool_calls=formatted,
            reasoning=list(response.reasoning) if response.reasoning else None,
        )

    @staticmethod
    def _finalize_evidence_status(state: CurriculumQAState) -> None:
        if state.evidence:
            failed = [r for r in state.retrieval_history if r.status == "error"]
            state.evidence_status = (
                EvidenceStatus.PARTIAL if failed else EvidenceStatus.FOUND
            )
            return
        if any(r.status == "not_found" for r in state.retrieval_history):
            state.evidence_status = EvidenceStatus.NOT_FOUND
            return
        if any(r.status == "error" for r in state.retrieval_history):
            state.evidence_status = EvidenceStatus.ERROR
            return
        state.evidence_status = EvidenceStatus.NOT_FOUND
