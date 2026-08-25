"""Retrieval node: LLM tool selection → curriculum tool execution → evidence."""

from __future__ import annotations

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
"""


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

        while (
            state.tool_calls < self.settings.agent_max_tool_calls
            and state.iteration < self.settings.agent_max_iterations
        ):
            state.bump_iteration()
            response = self.llm.generate_with_tools(messages, tools=curriculum_tools)
            if not response.tool_calls:
                break

            for call in response.tool_calls:
                if state.tool_calls >= self.settings.agent_max_tool_calls:
                    break
                self._execute_call(state, call, request_id=request_id)
                # Feed tool result back for multi-step retrieval.
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=json.dumps(
                            {
                                "tool_call": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                }
                            }
                        ),
                    )
                )
                last = state.retrieval_history[-1] if state.retrieval_history else None
                messages.append(
                    LLMMessage(
                        role="tool",
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

            # Heuristic stop: if we already have structure/topic evidence, allow one more round max
            if state.evidence and state.tool_calls >= 1:
                # Continue only if last call was search and found topics without details
                last = state.retrieval_history[-1]
                if last.tool != "search_curriculum" or last.evidence_count == 0:
                    if last.tool in {
                        "get_curriculum_structure",
                        "get_topic",
                        "get_learning_objectives",
                        "get_subject",
                    }:
                        # Optionally follow search → get_topic for first hit
                        pass

        self._finalize_evidence_status(state)
        state.status = AgentStatus.RETRIEVED
        state.metadata["tools_used"] = list(
            dict.fromkeys(r.tool for r in state.retrieval_history)
        )
        state.metadata["evidence_count"] = len(state.evidence)
        state.metadata["evidence_status"] = state.evidence_status.value
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
        return (
            f"Question: {state.question}\n"
            f"Known filters: {json.dumps(filters)}\n"
            "Select and call the appropriate curriculum tool(s)."
        )

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
                error_code = ((result.data or {}) if isinstance(result.data, dict) else {}).get(
                    "error_code"
                )
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
            tool_status=record.status,
            tool_execution_time=record.latency_ms,
            curriculum_api_status=record.curriculum_api_status,
            evidence_count=record.evidence_count,
            error=record.error,
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
