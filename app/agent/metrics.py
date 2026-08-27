"""In-process agent metrics for Sprint 4 observability."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    insufficient_evidence_requests: int = 0
    clarification_requests: int = 0
    error_requests: int = 0
    verification_passes: int = 0
    verification_failures: int = 0
    max_iteration_terminations: int = 0
    retrieval_failures: int = 0
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(
        self,
        *,
        status: str,
        iterations: int,
        tool_calls: int,
        latency_ms: float,
        verification_passed: bool | None = None,
        max_iterations: bool = False,
        retrieval_failed: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: float = 0.0,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_iterations += iterations
            self.total_tool_calls += tool_calls
            self.total_latency_ms += latency_ms
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.estimated_cost += estimated_cost

            if status == "completed":
                self.successful_requests += 1
            elif status == "insufficient_evidence":
                self.insufficient_evidence_requests += 1
            elif status == "needs_clarification":
                self.clarification_requests += 1
            elif status in {"failed", "error"}:
                self.error_requests += 1

            if verification_passed is True:
                self.verification_passes += 1
            elif verification_passed is False:
                self.verification_failures += 1

            if max_iterations:
                self.max_iteration_terminations += 1
            if retrieval_failed:
                self.retrieval_failures += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = max(self.total_requests, 1)
            verifications = self.verification_passes + self.verification_failures
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "insufficient_evidence_requests": self.insufficient_evidence_requests,
                "clarification_requests": self.clarification_requests,
                "error_requests": self.error_requests,
                "verification_pass_rate": (
                    self.verification_passes / verifications if verifications else 0.0
                ),
                "average_iterations": self.total_iterations / total,
                "average_tool_calls": self.total_tool_calls / total,
                "max_iteration_rate": self.max_iteration_terminations / total,
                "retrieval_failure_rate": self.retrieval_failures / total,
                "average_latency_ms": self.total_latency_ms / total,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "estimated_cost": self.estimated_cost,
            }

    def reset(self) -> None:
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.insufficient_evidence_requests = 0
            self.clarification_requests = 0
            self.error_requests = 0
            self.verification_passes = 0
            self.verification_failures = 0
            self.max_iteration_terminations = 0
            self.retrieval_failures = 0
            self.total_iterations = 0
            self.total_tool_calls = 0
            self.total_latency_ms = 0.0
            self.input_tokens = 0
            self.output_tokens = 0
            self.estimated_cost = 0.0


_METRICS = AgentMetrics()


def get_metrics() -> AgentMetrics:
    return _METRICS
