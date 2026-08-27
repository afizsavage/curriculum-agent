"""Retrieval node: LLM tool selection → curriculum tool execution → evidence."""

from __future__ import annotations

import json
import time
from typing import Any

from app.agent.retrieval_state import (
    RetrievalState,
    build_retrieval_objective,
    is_low_value_broad_call,
    is_relevant_evidence,
    objective_key,
    targeted_tool_calls_from_missing,
    tool_call_key,
    tool_fingerprint,
)
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
Only use the provided tools. Prefer resolve_curriculum_context when grade and
subject (and ideally topic) are known — it resolves GradeCurriculum units and
learning outcomes in one structured call. Fall back to get_curriculum_structure,
get_topic, get_learning_objectives, and search_curriculum when needed. Use
search_curriculum for concept discovery. Do not invent curriculum facts. When
enough evidence exists, stop requesting tools and reply with a short note that
retrieval is complete.

When verification feedback lists missing evidence, prioritize targeted tools that
satisfy those gaps rather than repeating broad searches already executed.
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
        trace = get_current_trace()
        evidence_before = len(state.evidence)
        rs = state.retrieval_state
        if not isinstance(rs, RetrievalState):
            rs = RetrievalState.model_validate(rs)
            state.retrieval_state = rs

        new_evidence = 0
        new_relevant = 0
        duplicate_evidence = 0
        tools_attempted = 0
        tools_executed = 0
        tools_skipped = 0
        if hasattr(self.llm, "set_active_node"):
            self.llm.set_active_node("retrieve")

        curriculum_tools = [
            t for t in self.tools.llm_tool_specs() if t.get("name") != "echo"
        ]
        available = {t["name"] for t in curriculum_tools if t.get("name")}
        if not curriculum_tools:
            state.evidence_status = EvidenceStatus.ERROR
            state.error = "No curriculum tools registered"
            state.status = AgentStatus.FAILED
            return state

        follow_up = bool(state.pending_missing_evidence) and state.retrieval_rounds > 1
        subject = state.subject or rs.resolved_subject
        objective = build_retrieval_objective(
            pending_missing=state.pending_missing_evidence,
            grade=state.grade,
            subject=subject,
            topic=state.topic,
        )
        obj_key = objective_key(objective)
        repeated_objective = obj_key in {objective_key(o) for o in rs.objectives}
        if obj_key not in {objective_key(o) for o in rs.objectives}:
            rs.objectives.append(objective)

        retrieval_hints = {
            "intent": state.intent,
            "level": state.level,
            "grade": state.grade,
            "subject": state.subject,
            "resolved_subject": rs.resolved_subject,
            "topic": state.topic,
            "pending_missing_evidence": [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in (state.pending_missing_evidence or [])
            ],
            "retrieval_rounds": state.retrieval_rounds,
            "retrieval_objective": objective,
            "user_prompt": self._user_prompt(state, objective=objective),
        }
        if trace is not None:
            trace.emit(
                "agent.retrieval.start",
                iteration=state.iteration,
                node="retrieve",
                evidence_before=evidence_before,
                retrieval_hints=retrieval_hints,
            )

        remaining = self.settings.agent_max_tool_calls - state.tool_calls
        if remaining <= 0:
            self._finalize_evidence_status(state)
            state.status = AgentStatus.RETRIEVED
            return state
        round_tool_budget = max(1, min(4, remaining))

        # Follow-up after verify→retrieve_more: goal-directed tools only.
        planned_calls: list[ToolCallRequest] = []
        plan_mode = "llm"
        if follow_up:
            plan_mode = "targeted"
            planned_calls = targeted_tool_calls_from_missing(
                state.pending_missing_evidence,
                available_tools=available,
                grade=state.grade,
                subject=subject,
                topic=state.topic,
                retrieval_state=rs,
            )
            if not planned_calls:
                rs.no_progress = True
                rs.no_progress_reason = (
                    "repeated_retrieval_objective"
                    if repeated_objective
                    else "no_non_duplicate_targeted_retrieval"
                )
                rs.retrieval_rounds_without_progress += 1
                if trace is not None:
                    trace.emit(
                        "agent.retrieval.plan",
                        iteration=state.iteration,
                        objective=objective,
                        candidate_tools=[],
                        selected_tool=None,
                        reason=rs.no_progress_reason,
                        duplicate=True,
                        expected_information_gain="none",
                        plan_mode=plan_mode,
                    )
                    trace.emit(
                        "agent.retrieval.end",
                        iteration=state.iteration,
                        node="retrieve",
                        total_evidence=len(state.evidence),
                        new_evidence=0,
                        new_relevant_evidence=0,
                        duplicate_evidence=0,
                        tools_this_round=0,
                        tools_skipped_this_round=0,
                        retrieval_gain=0,
                        no_progress=True,
                        no_progress_reason=rs.no_progress_reason,
                        evidence_status=state.evidence_status.value,
                        retrieval_metrics=rs.metrics_snapshot(),
                    )
                self._finalize_evidence_status(state)
                state.status = AgentStatus.RETRIEVED
                state.metadata["retrieval_state"] = rs.metrics_snapshot()
                state.metadata["no_retrieval_progress"] = True
                return state

            if trace is not None:
                trace.emit(
                    "agent.retrieval.plan",
                    iteration=state.iteration,
                    objective=objective,
                    candidate_tools=[c.name for c in planned_calls],
                    selected_tool=planned_calls[0].name if planned_calls else None,
                    reason="Verifier identified specific missing evidence",
                    duplicate=False,
                    expected_information_gain="high",
                    plan_mode=plan_mode,
                )
            rs.targeted_retrievals += len(planned_calls)

            for call in planned_calls:
                if (
                    state.tool_calls >= self.settings.agent_max_tool_calls
                    or tools_executed >= round_tool_budget
                ):
                    break
                tools_attempted += 1
                added, rel, dupes, skipped = self._process_call(
                    state,
                    call,
                    request_id=request_id,
                    follow_up_round=True,
                    objective=objective,
                )
                if skipped:
                    tools_skipped += 1
                    continue
                tools_executed += 1
                new_evidence += added
                new_relevant += rel
                duplicate_evidence += dupes
        else:
            # Fast path: LLM selects tools (preserve Run-4 style).
            if trace is not None:
                trace.emit(
                    "agent.retrieval.plan",
                    iteration=state.iteration,
                    objective=objective,
                    candidate_tools=sorted(available),
                    selected_tool=None,
                    reason="Initial retrieval via LLM planner",
                    duplicate=False,
                    expected_information_gain="medium",
                    plan_mode=plan_mode,
                )

            messages = [
                LLMMessage(role="system", content=SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=self._user_prompt(state, objective=objective),
                ),
            ]
            planning_steps = 0
            max_planning_steps = 4
            while (
                state.tool_calls < self.settings.agent_max_tool_calls
                and tools_executed < round_tool_budget
                and planning_steps < max_planning_steps
            ):
                planning_steps += 1
                response = self.llm.generate_with_tools(
                    messages, tools=curriculum_tools
                )
                if not response.tool_calls:
                    break

                executed_calls: list[ToolCallRequest] = []
                tool_messages: list[LLMMessage] = []
                for call in response.tool_calls:
                    if (
                        state.tool_calls >= self.settings.agent_max_tool_calls
                        or tools_executed >= round_tool_budget
                    ):
                        break
                    tools_attempted += 1
                    if call.name in {
                        "search_curriculum",
                        "get_curriculum_structure",
                    }:
                        rs.broad_retrievals += 1
                    else:
                        rs.targeted_retrievals += 1

                    added, rel, dupes, skipped = self._process_call(
                        state,
                        call,
                        request_id=request_id,
                        follow_up_round=False,
                        objective=objective,
                    )
                    if skipped:
                        tools_skipped += 1
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
                                            "Identical or low-value tool call "
                                            "already covered; use a different "
                                            "tool or arguments."
                                        ),
                                    }
                                ),
                            )
                        )
                        executed_calls.append(call)
                        continue

                    tools_executed += 1
                    new_evidence += added
                    new_relevant += rel
                    duplicate_evidence += dupes
                    executed_calls.append(call)
                    last = (
                        state.retrieval_history[-1]
                        if state.retrieval_history
                        else None
                    )
                    tool_messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=call.id,
                            content=json.dumps(
                                {
                                    "tool": call.name,
                                    "status": last.status if last else "unknown",
                                    "evidence_count": (
                                        last.evidence_count if last else 0
                                    ),
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
                    self._assistant_tool_message(
                        response, tool_calls=executed_calls
                    )
                )
                messages.extend(tool_messages)

        rs.last_retrieval_gain = new_evidence
        rs.last_relevant_gain = new_relevant
        rs.cumulative_retrieval_gain += new_evidence
        rs.cumulative_relevant_gain += new_relevant
        if new_relevant > 0 or new_evidence > 0:
            rs.retrieval_rounds_with_progress += 1
            rs.no_progress = False
            rs.no_progress_reason = None
        else:
            rs.retrieval_rounds_without_progress += 1
            if follow_up or tools_executed == 0:
                rs.no_progress = True
                rs.no_progress_reason = rs.no_progress_reason or (
                    "zero_relevant_new_evidence"
                    if tools_executed
                    else "all_candidate_tools_duplicates"
                )

        round_summary = {
            "iteration": state.iteration,
            "objective": objective,
            "tools_attempted": tools_attempted,
            "tools_executed": tools_executed,
            "tools_skipped": tools_skipped,
            "new_evidence": new_evidence,
            "new_relevant_evidence": new_relevant,
            "duplicate_evidence": duplicate_evidence,
            "retrieval_gain": new_evidence,
            "relevant_retrieval_gain": new_relevant,
        }
        rs.rounds.append(round_summary)

        self._finalize_evidence_status(state)
        state.status = AgentStatus.RETRIEVED
        state.metadata["tools_used"] = list(
            dict.fromkeys(r.tool for r in state.retrieval_history)
        )
        state.metadata["evidence_count"] = len(state.evidence)
        state.metadata["evidence_status"] = state.evidence_status.value
        state.metadata["retrieval_rounds"] = state.retrieval_rounds
        state.metadata["retrieval_state"] = rs.metrics_snapshot()
        state.metadata["last_retrieval_gain"] = rs.last_retrieval_gain
        state.metadata["cumulative_retrieval_gain"] = rs.cumulative_retrieval_gain
        state.metadata["last_retrieval_relevance"] = rs.last_relevant_gain
        if rs.no_progress:
            state.metadata["no_retrieval_progress"] = True

        bag = summarize_evidence_bag(state.evidence)
        if trace is not None:
            it = trace.ensure_iteration(state.iteration)
            ev = it["evidence"]
            ev["new"] = new_evidence
            ev["new_relevant"] = new_relevant
            ev["duplicate"] = duplicate_evidence
            ev["total_after"] = len(state.evidence)
            ev["by_type"] = bag["by_type"]
            ev["by_grade"] = bag["by_grade"]
            ev["by_subject"] = bag["by_subject"]
            it["retrieval_round"] = round_summary
            trace.emit(
                "agent.retrieval.round_summary",
                **round_summary,
            )
            trace.emit(
                "agent.retrieval.end",
                iteration=state.iteration,
                node="retrieve",
                total_evidence=len(state.evidence),
                new_evidence=new_evidence,
                new_relevant_evidence=new_relevant,
                duplicate_evidence=duplicate_evidence,
                evidence_by_type=bag["by_type"],
                evidence_by_grade=bag["by_grade"],
                evidence_by_subject=bag["by_subject"],
                tools_this_round=tools_executed,
                tools_skipped_this_round=tools_skipped,
                retrieval_gain=new_evidence,
                evidence_status=state.evidence_status.value,
                retrieval_metrics=rs.metrics_snapshot(),
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
            new_relevant_evidence=new_relevant,
            duplicate_evidence=duplicate_evidence,
            tools_skipped=tools_skipped,
        )
        return state

    def _process_call(
        self,
        state: CurriculumQAState,
        call: ToolCallRequest,
        *,
        request_id: str | None,
        follow_up_round: bool,
        objective: str,
    ) -> tuple[int, int, int, bool]:
        """Execute or skip a tool call. Returns (new, relevant, dupes, skipped)."""
        rs = state.retrieval_state
        trace = get_current_trace()
        fp = tool_fingerprint(call.name, call.arguments)
        key = tool_call_key(call.name, call.arguments)

        skip_reason = None
        previous = rs.previous_call_number(fp)
        if previous is not None or key in state.executed_tool_keys:
            skip_reason = "duplicate_call"
        elif is_low_value_broad_call(
            call.name,
            call.arguments,
            rs,
            follow_up_round=follow_up_round,
        ):
            skip_reason = "low_value_broad_repeat"

        if skip_reason:
            rs.tools_skipped += 1
            rs.duplicate_tool_calls_prevented += 1
            if trace is not None:
                trace.emit(
                    "agent.tool.skip",
                    iteration=state.iteration,
                    tool_name=call.name,
                    arguments=call.arguments or {},
                    fingerprint=fp,
                    reason=skip_reason,
                    previous_tool_call=previous,
                    previous_call_number=previous,
                    objective=objective,
                )
                trace.emit(
                    "agent.retrieval.plan",
                    iteration=state.iteration,
                    objective=objective,
                    candidate_tools=[call.name],
                    selected_tool=call.name,
                    reason=skip_reason,
                    duplicate=True,
                    expected_information_gain="none",
                    fingerprint=fp,
                )
            return 0, 0, 0, True

        if trace is not None:
            trace.emit(
                "agent.retrieval.plan",
                iteration=state.iteration,
                objective=objective,
                candidate_tools=[call.name],
                selected_tool=call.name,
                reason=(
                    "Verifier-targeted gap"
                    if follow_up_round
                    else "LLM-selected retrieval"
                ),
                duplicate=False,
                expected_information_gain="high" if follow_up_round else "medium",
                fingerprint=fp,
            )

        added, dupes = self._execute_call(state, call, request_id=request_id)
        tool_call_number = state.tool_calls  # already bumped
        rs.remember_fingerprint(fp, tool_call_number)
        if key not in state.executed_tool_keys:
            state.executed_tool_keys.append(key)
        rs.tools_executed.append(call.name)
        if call.name == "search_curriculum":
            q = (call.arguments or {}).get("query")
            if q and str(q) not in rs.queries_executed:
                rs.queries_executed.append(str(q))
        if call.name == "get_curriculum_structure":
            rs.note_structure_call(call.arguments)

        relevant = 0
        # Count relevant among newly appended evidence at the end of the bag.
        if added:
            for item in state.evidence[-added:]:
                rs.note_evidence(item, is_new=True)
                if is_relevant_evidence(
                    item,
                    grade=state.grade,
                    subject=state.subject or rs.resolved_subject,
                    topic=state.topic,
                    pending_missing=state.pending_missing_evidence,
                ):
                    relevant += 1
            rs.update_coverage_from_evidence(state.evidence[-added:])
        return added, relevant, dupes, False

    def _user_prompt(
        self,
        state: CurriculumQAState,
        *,
        objective: str | None = None,
    ) -> str:
        filters = {
            "intent": state.intent,
            "level": state.level,
            "grade": state.grade,
            "subject": state.subject,
            "topic": state.topic,
            "resolved_subject": state.retrieval_state.resolved_subject,
        }
        parts = [
            f"Question: {state.question}",
            f"Known filters: {json.dumps(filters)}",
        ]
        if objective:
            parts.append(f"Retrieval objective: {objective}")
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
                "Prefer targeted tools (resolve_curriculum_context, "
                "get_topic, get_learning_objectives) "
                "that address the missing evidence. Do not repeat identical "
                "or broad structure searches already executed."
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
        fp = tool_fingerprint(call.name, call.arguments)
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
                fingerprint=fp,
                duplicate_tool_call=duplicate_seen,
            )

        new_count = 0
        dup_count = 0
        result_previews: list[dict[str, Any]] = []
        observability: dict[str, Any] | None = None
        try:
            result = self.tools.execute(call.name, **(call.arguments or {}))
            latency = timed_ms(started)
            state.bump_tool_calls()
            if result.success:
                evidence_rows = (result.data or {}).get("evidence") or []
                observability = (
                    (result.data or {}).get("observability")
                    if isinstance(result.data, dict)
                    else None
                )
                for row in evidence_rows:
                    item = CurriculumEvidence.model_validate(row)
                    preview = evidence_preview(item)
                    is_dup = bool(
                        item.entity_id
                        and any(e.entity_id == item.entity_id for e in state.evidence)
                    )
                    if is_dup:
                        dup_count += 1
                        state.retrieval_state.note_evidence(item, is_new=False)
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
            "fingerprint": fp,
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
        if observability:
            tool_row["observability"] = observability
        if trace is not None:
            trace.tool_calls.append(tool_row)
            it = trace.ensure_iteration(state.iteration)
            it["tool_calls"].append(tool_row)
            emit_kwargs: dict[str, Any] = {
                "iteration": state.iteration,
                "tool_call_number": tool_call_number,
                "tool_name": record.tool,
                "duration_ms": record.latency_ms,
                "success": record.status == "success",
                "error": record.error,
                "result_count": record.evidence_count,
                "new_evidence": new_count,
                "duplicate_evidence": dup_count,
                "results": result_previews[:12],
                "duplicate_tool_call": duplicate_seen,
                "fingerprint": fp,
                "curriculum_api_status": record.curriculum_api_status,
            }
            if observability:
                emit_kwargs["observability"] = observability
            trace.emit("agent.tool.end", **emit_kwargs)
        log_agent_event(
            logger,
            "agent.tool.execute",
            request_id=request_id,
            conversation_id=state.conversation_id,
            tool_name=record.tool,
            tool_arguments=record.arguments,
            tool_arguments_hash=key,
            tool_fingerprint=fp,
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
