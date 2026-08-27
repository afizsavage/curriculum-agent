"""Retrieval node: LLM tool selection → curriculum tool execution → evidence."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from app.agent.state import CurriculumQAState, RetrievedContextItem
from app.agent.trace import (
    evidence_preview,
    get_current_trace,
    summarize_evidence_bag,
    timed_ms,
)
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
        trace = get_current_trace()
        evidence_before = len(state.evidence)
        new_evidence = 0
        duplicate_evidence = 0
        if hasattr(self.llm, "set_active_node"):
            self.llm.set_active_node("retrieve")

        curriculum_tools = [
            t for t in self.tools.llm_tool_specs() if t.get("name") != "echo"
        ]
        if not curriculum_tools:
            state.evidence_status = EvidenceStatus.ERROR
            state.error = "No curriculum tools registered"
            state.status = AgentStatus.FAILED
            return state

        retrieval_hints = {
            "intent": state.intent,
            "level": state.level,
            "grade": state.grade,
            "subject": state.subject,
            "topic": state.topic,
            "pending_missing_evidence": [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in (state.pending_missing_evidence or [])
            ],
            "retrieval_rounds": state.retrieval_rounds,
            "user_prompt": self._user_prompt(state),
        }
        if trace is not None:
            trace.emit(
                "agent.retrieval.start",
                iteration=state.iteration,
                node="retrieve",
                evidence_before=evidence_before,
                retrieval_hints=retrieval_hints,
            )
            # No separate planner object yet — log current selection inputs.
            trace.emit(
                "agent.retrieval.plan",
                iteration=state.iteration,
                plan=retrieval_hints,
            )

        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=self._user_prompt(state),
            ),
        ]

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
                    if trace is not None:
                        trace.emit(
                            "agent.tool.start",
                            iteration=state.iteration,
                            tool_call_number=state.tool_calls + 1,
                            node="retrieve",
                            tool_name=call.name,
                            arguments=call.arguments or {},
                            duplicate_tool_call=True,
                            skipped=True,
                        )
                        trace.emit(
                            "agent.tool.end",
                            iteration=state.iteration,
                            tool_call_number=state.tool_calls + 1,
                            tool_name=call.name,
                            duration_ms=0,
                            success=False,
                            error="skipped_duplicate",
                            result_count=0,
                            duplicate_tool_call=True,
                            skipped=True,
                        )
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

                added, dupes = self._execute_call(state, call, request_id=request_id)
                new_evidence += added
                duplicate_evidence += dupes
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

        bag = summarize_evidence_bag(state.evidence)
        if trace is not None:
            it = trace.ensure_iteration(state.iteration)
            ev = it["evidence"]
            ev["new"] = new_evidence
            ev["duplicate"] = duplicate_evidence
            ev["total_after"] = len(state.evidence)
            ev["by_type"] = bag["by_type"]
            ev["by_grade"] = bag["by_grade"]
            ev["by_subject"] = bag["by_subject"]
            trace.emit(
                "agent.retrieval.end",
                iteration=state.iteration,
                node="retrieve",
                total_evidence=len(state.evidence),
                new_evidence=new_evidence,
                duplicate_evidence=duplicate_evidence,
                evidence_by_type=bag["by_type"],
                evidence_by_grade=bag["by_grade"],
                evidence_by_subject=bag["by_subject"],
                tools_this_round=tools_this_round,
                evidence_status=state.evidence_status.value,
            )

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
            new_evidence=new_evidence,
            duplicate_evidence=duplicate_evidence,
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
    ) -> tuple[int, int]:
        if call.name not in state.selected_tools:
            state.selected_tools.append(call.name)
        started = time.perf_counter()
        api_status = None
        trace = get_current_trace()
        tool_call_number = state.tool_calls + 1
        key = tool_call_key(call.name, call.arguments)
        duplicate_seen = bool(trace and key in trace.seen_tool_keys)
        if trace is not None:
            trace.seen_tool_keys.add(key)
            trace.emit(
                "agent.tool.start",
                iteration=state.iteration,
                tool_call_number=tool_call_number,
                node="retrieve",
                tool_name=call.name,
                arguments=call.arguments or {},
                duplicate_tool_call=duplicate_seen,
            )

        new_count = 0
        dup_count = 0
        result_previews: list[dict[str, Any]] = []
        try:
            result = self.tools.execute(call.name, **(call.arguments or {}))
            latency = timed_ms(started)
            state.bump_tool_calls()
            if result.success:
                evidence_rows = (result.data or {}).get("evidence") or []
                for row in evidence_rows:
                    item = CurriculumEvidence.model_validate(row)
                    preview = evidence_preview(item)
                    is_dup = bool(
                        item.entity_id
                        and any(e.entity_id == item.entity_id for e in state.evidence)
                    )
                    if is_dup:
                        dup_count += 1
                        if trace is not None:
                            trace.emit(
                                "agent.evidence.add",
                                iteration=state.iteration,
                                tool_call_number=tool_call_number,
                                source_tool=call.name,
                                duplicate=True,
                                disposition="duplicate",
                                **preview,
                            )
                            trace.evidence_adds.append(
                                {
                                    **preview,
                                    "duplicate": True,
                                    "iteration": state.iteration,
                                    "tool_call_number": tool_call_number,
                                    "source_tool": call.name,
                                }
                            )
                        continue
                    state.evidence.append(item)
                    state.retrieved_context.append(
                        RetrievedContextItem(
                            source=item.source_reference or call.name,
                            content=item.content or item.name or "",
                            metadata=item.model_dump(),
                        )
                    )
                    new_count += 1
                    result_previews.append(preview)
                    if trace is not None:
                        trace.emit(
                            "agent.evidence.add",
                            iteration=state.iteration,
                            tool_call_number=tool_call_number,
                            source_tool=call.name,
                            duplicate=False,
                            disposition="new",
                            **preview,
                        )
                        trace.evidence_adds.append(
                            {
                                **preview,
                                "duplicate": False,
                                "iteration": state.iteration,
                                "tool_call_number": tool_call_number,
                                "source_tool": call.name,
                            }
                        )
                        it = trace.ensure_iteration(state.iteration)
                        it["evidence"]["items"].append(preview)
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
            latency = timed_ms(started)
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
            latency = timed_ms(started)
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
        tool_row = {
            "tool_call_number": tool_call_number,
            "iteration": state.iteration,
            "tool_name": record.tool,
            "arguments": record.arguments,
            "duration_ms": record.latency_ms,
            "success": record.status == "success",
            "error": record.error,
            "result_count": record.evidence_count,
            "new_evidence": new_count,
            "duplicate_evidence": dup_count,
            "results": result_previews[:12],
            "duplicate_tool_call": duplicate_seen,
            "curriculum_api_status": record.curriculum_api_status,
        }
        if trace is not None:
            trace.tool_calls.append(tool_row)
            it = trace.ensure_iteration(state.iteration)
            it["tool_calls"].append(tool_row)
            trace.emit(
                "agent.tool.end",
                iteration=state.iteration,
                tool_call_number=tool_call_number,
                tool_name=record.tool,
                duration_ms=record.latency_ms,
                success=record.status == "success",
                error=record.error,
                result_count=record.evidence_count,
                new_evidence=new_count,
                duplicate_evidence=dup_count,
                results=result_previews[:12],
                duplicate_tool_call=duplicate_seen,
                curriculum_api_status=record.curriculum_api_status,
            )
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
            tool_call_number=tool_call_number,
            duplicate_tool_call=duplicate_seen,
        )
        return new_count, dup_count

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
