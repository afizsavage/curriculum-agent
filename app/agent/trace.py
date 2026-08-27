"""Structured execution traces for Curriculum Q&A diagnostics.

Observability only — does not change agent routing, retrieval, or verification.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable
from uuid import uuid4

from app.logging_utils import get_logger, log_agent_event

logger = get_logger(__name__)

_current_trace: ContextVar["AgentRunTrace | None"] = ContextVar(
    "agent_run_trace", default=None
)

# Keep recent traces in memory for debug endpoint / tests.
_MAX_TRACES = 64
_TRACE_DIR = Path(__file__).resolve().parents[2] / "data" / "traces"


def new_agent_run_id() -> str:
    return f"run-{uuid4().hex[:16]}"


def get_current_trace() -> "AgentRunTrace | None":
    return _current_trace.get()


def evidence_preview(item: Any, *, content_limit: int = 160) -> dict[str, Any]:
    """Bounded summary of a curriculum evidence row."""
    meta = getattr(item, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    content = getattr(item, "content", None)
    if isinstance(content, str) and len(content) > content_limit:
        content = content[:content_limit] + "…"
    return {
        "id": getattr(item, "entity_id", None),
        "entity_type": getattr(item, "entity_type", None),
        "name": getattr(item, "name", None),
        "grade": getattr(item, "grade", None),
        "subject": getattr(item, "subject", None),
        "topic": getattr(item, "topic", None),
        "level": getattr(item, "level", None),
        "parent_id": meta.get("parent_id") or meta.get("topic_id"),
        "source": getattr(item, "source_reference", None)
        or getattr(item, "source", None),
        "content_preview": content,
    }


@dataclass
class AgentRunTrace:
    agent_run_id: str
    request_id: str | None
    conversation_id: str | None
    agent_name: str
    question: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ended_at: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    iterations: dict[int, dict[str, Any]] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence_adds: list[dict[str, Any]] = field(default_factory=list)
    routes: list[dict[str, Any]] = field(default_factory=list)
    node_timings_ms: dict[str, float] = field(default_factory=dict)
    phase_timings_ms: dict[str, float] = field(default_factory=dict)
    seen_tool_keys: set[str] = field(default_factory=set)
    llm_counts: dict[str, int] = field(
        default_factory=lambda: {
            "total": 0,
            "understand": 0,
            "retrieval": 0,
            "generation": 0,
            "verification": 0,
            "other": 0,
        }
    )
    final: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "agent_run_id": self.agent_run_id,
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in fields.items() if v is not None},
        }
        with self._lock:
            self.events.append(payload)
        question = fields.pop("question", None)
        if question is None and event.endswith(".start"):
            question = self.question
        log_agent_event(
            logger,
            event,
            request_id=self.request_id,
            conversation_id=self.conversation_id,
            agent_run_id=self.agent_run_id,
            question=question,
            **fields,
        )

    def ensure_iteration(self, iteration: int) -> dict[str, Any]:
        with self._lock:
            if iteration not in self.iterations:
                self.iterations[iteration] = {
                    "iteration": iteration,
                    "nodes": [],
                    "tool_calls": [],
                    "evidence": {
                        "new": 0,
                        "duplicate": 0,
                        "total_after": 0,
                        "by_type": {},
                        "by_grade": {},
                        "by_subject": {},
                        "items": [],
                    },
                    "generation": None,
                    "verification": None,
                    "route": None,
                }
            return self.iterations[iteration]

    def add_node_timing(self, node: str, duration_ms: float) -> None:
        with self._lock:
            self.node_timings_ms[node] = self.node_timings_ms.get(node, 0.0) + duration_ms
            phase = _node_to_phase(node)
            self.phase_timings_ms[phase] = (
                self.phase_timings_ms.get(phase, 0.0) + duration_ms
            )

    def record_llm(
        self,
        *,
        node: str,
        iteration: int | None,
        model: str | None,
        kind: str,
        duration_ms: float,
        success: bool,
        error: str | None = None,
        tool_calls_requested: int | None = None,
        structured_output_valid: bool | None = None,
    ) -> None:
        bucket = _llm_bucket(node, kind)
        with self._lock:
            self.llm_counts["total"] += 1
            self.llm_counts[bucket] = self.llm_counts.get(bucket, 0) + 1
            self.llm_calls.append(
                {
                    "node": node,
                    "iteration": iteration,
                    "model": model,
                    "kind": kind,
                    "duration_ms": round(duration_ms, 2),
                    "success": success,
                    "error": error,
                    "tool_calls_requested": tool_calls_requested,
                    "structured_output_valid": structured_output_valid,
                    "bucket": bucket,
                }
            )
        self.emit(
            "agent.llm.end",
            node=node,
            iteration=iteration,
            model=model,
            kind=kind,
            duration_ms=round(duration_ms, 2),
            success=success,
            error=error,
            tool_calls_requested=tool_calls_requested,
            structured_output_valid=structured_output_valid,
            llm_bucket=bucket,
        )

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            iteration_list = [
                self.iterations[i] for i in sorted(self.iterations.keys())
            ]
            return {
                "agent_run_id": self.agent_run_id,
                "request_id": self.request_id,
                "conversation_id": self.conversation_id,
                "agent_name": self.agent_name,
                "question": self.question,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "events": list(self.events),
                "iterations": iteration_list,
                "tool_calls": list(self.tool_calls),
                "llm_calls": list(self.llm_calls),
                "evidence_adds": list(self.evidence_adds),
                "routes": list(self.routes),
                "node_timings_ms": dict(self.node_timings_ms),
                "phase_timings_ms": dict(self.phase_timings_ms),
                "llm_counts": dict(self.llm_counts),
                "final": dict(self.final),
            }


class TraceStore:
    """In-memory ring buffer of recent AgentRunTrace objects."""

    def __init__(self, *, max_traces: int = _MAX_TRACES) -> None:
        self._max = max_traces
        self._traces: OrderedDict[str, AgentRunTrace] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, trace: AgentRunTrace) -> None:
        with self._lock:
            self._traces[trace.agent_run_id] = trace
            self._traces.move_to_end(trace.agent_run_id)
            while len(self._traces) > self._max:
                self._traces.popitem(last=False)

    def get(self, agent_run_id: str) -> AgentRunTrace | None:
        with self._lock:
            return self._traces.get(agent_run_id)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._traces.keys())


_STORE = TraceStore()


def get_trace_store() -> TraceStore:
    return _STORE


@contextmanager
def start_agent_run(
    *,
    question: str,
    request_id: str | None,
    conversation_id: str | None = None,
    agent_name: str = "curriculum_qa",
    agent_run_id: str | None = None,
    persist: bool = True,
) -> Generator[AgentRunTrace, None, None]:
    trace = AgentRunTrace(
        agent_run_id=agent_run_id or new_agent_run_id(),
        request_id=request_id,
        conversation_id=conversation_id,
        agent_name=agent_name,
        question=question,
    )
    token = _current_trace.set(trace)
    trace.emit(
        "agent.request.start",
        agent_name=agent_name,
        question=question,
    )
    try:
        yield trace
    finally:
        _current_trace.reset(token)
        get_trace_store().put(trace)
        if persist:
            _persist_trace(trace)


def bind_conversation_id(trace: AgentRunTrace, conversation_id: str | None) -> None:
    if conversation_id:
        trace.conversation_id = conversation_id


def finish_agent_run(
    trace: AgentRunTrace,
    *,
    status: str,
    final_node: str | None,
    iteration: int,
    tool_calls: int,
    retrieval_rounds: int,
    verification_attempts: int,
    evidence_count: int,
    latency_ms: float,
    termination_reason: str | None,
    verification_score: float | None = None,
    visited_nodes: list[str] | None = None,
    max_iterations: int | None = None,
    max_tool_calls: int | None = None,
    max_retrieval_rounds: int | None = None,
) -> None:
    trace.ended_at = datetime.now(timezone.utc).isoformat()
    trace.final = {
        "status": status,
        "final_node": final_node,
        "iteration": iteration,
        "tool_calls": tool_calls,
        "retrieval_rounds": retrieval_rounds,
        "verification_attempts": verification_attempts,
        "evidence_count": evidence_count,
        "latency_ms": round(latency_ms, 2),
        "termination_reason": termination_reason,
        "verification_score": verification_score,
        "visited_nodes": visited_nodes or [],
        "llm_counts": dict(trace.llm_counts),
        "phase_timings_ms": {
            k: round(v, 2) for k, v in trace.phase_timings_ms.items()
        },
        "node_timings_ms": {
            k: round(v, 2) for k, v in trace.node_timings_ms.items()
        },
        "limits": {
            "max_iterations": max_iterations,
            "max_tool_calls": max_tool_calls,
            "max_retrieval_rounds": max_retrieval_rounds,
        },
    }
    # Compact per-iteration growth summary event.
    growth = []
    for it in sorted(trace.iterations.keys()):
        row = trace.iterations[it]
        growth.append(
            {
                "iteration": it,
                "tool_calls": len(row.get("tool_calls") or []),
                "evidence_total": (row.get("evidence") or {}).get("total_after"),
                "evidence_new": (row.get("evidence") or {}).get("new"),
                "evidence_duplicate": (row.get("evidence") or {}).get("duplicate"),
                "verification": (row.get("verification") or {}).get("recommendation")
                if row.get("verification")
                else None,
            }
        )
    trace.emit(
        "agent.request.end",
        status=status,
        final_node=final_node,
        iteration=iteration,
        tool_calls=tool_calls,
        retrieval_rounds=retrieval_rounds,
        verification_attempts=verification_attempts,
        evidence_count=evidence_count,
        latency_ms=round(latency_ms, 2),
        termination_reason=termination_reason,
        verification_score=verification_score,
        visited_nodes=visited_nodes,
        llm_counts=trace.llm_counts,
        phase_timings_ms=trace.final["phase_timings_ms"],
        iteration_growth=growth,
    )


def _persist_trace(trace: AgentRunTrace) -> None:
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = _TRACE_DIR / f"{trace.agent_run_id}.json"
        path.write_text(json.dumps(trace.to_dict(), indent=2, default=str))
    except Exception:  # pragma: no cover - disk issues must not break asks
        logger.exception("agent.trace.persist_failed agent_run_id=%s", trace.agent_run_id)


def _node_to_phase(node: str) -> str:
    mapping = {
        "understand": "understand",
        "prepare_cycle": "routing",
        "retrieve": "retrieval",
        "generate_answer": "generation",
        "verify_answer": "verification",
        "clarify": "routing",
        "fallback": "routing",
        "finish": "routing",
    }
    return mapping.get(node, "other")


def _llm_bucket(node: str, kind: str) -> str:
    if node in {"retrieve", "retrieval"} or kind == "tools":
        return "retrieval"
    if node in {"generate_answer", "generation", "answer"} or kind == "generation":
        return "generation"
    if node in {"verify_answer", "verification", "verify"} or kind == "verification":
        return "verification"
    if node == "understand":
        return "understand"
    return "other"


def summarize_evidence_bag(items: Iterable[Any]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_grade: dict[str, int] = {}
    by_subject: dict[str, int] = {}
    for item in items:
        et = str(getattr(item, "entity_type", None) or "unknown")
        by_type[et] = by_type.get(et, 0) + 1
        g = str(getattr(item, "grade", None) or "unknown")
        by_grade[g] = by_grade.get(g, 0) + 1
        s = str(getattr(item, "subject", None) or "unknown")
        by_subject[s] = by_subject.get(s, 0) + 1
    return {"by_type": by_type, "by_grade": by_grade, "by_subject": by_subject}


def timed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
