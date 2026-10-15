"""Repeatable behaviour checks for the teacher assistant.

Deterministic and mocked runs do not call a model provider. Live provider
evaluation is a separate opt-in and is refused in production. This module does
not enable the assistant HTTP route and does not treat the local identity stub
as authentication.

Answer text is kept off the result records unless the caller sets
``include_answers``. Callers must not log those records.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from app.assistant.chat import execute_chat, reset_assistant_runtime
from app.assistant.chat import configure_assistant_runtime
from app.assistant.contract import (
    GENERAL_LABEL,
    GENERAL_SOURCE,
    AssistantTurnRequest,
    CurriculumOutcome,
)
from app.assistant.dispatch import dispatch_turn
from app.assistant.follow_up import resolve_context_follow_up
from app.assistant.identity import (
    VerifiedTeacherClaims,
    bind_verified_claims,
    current_identity_gate,
    register_identity_gate,
    reset_identity_gate,
)
from app.assistant.threads import (
    AssistantThreadStore,
    QaOutcomeSummary,
    ThreadStoreError,
    ThreadUnavailable,
)
from app.config import Settings
from app.exceptions import LLMProviderError, LLMTimeoutError
from app.schemas.agent import AskMetadata, AskResponse, VerificationSummary
from app.schemas.assistant import AssistantChatRequest

DATASET_VERSION = "assistant-behaviour-v1"
_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "assistant_behaviour_v1.json"
)
_FIXTURE_CURRICULUM = "Fixture curriculum sentence for harness checks."
_FIXTURE_GENERAL = "Fixture general teaching note for harness checks."
_EVAL_TEACHER = "eval-teacher"
_OTHER_TEACHER = "eval-other-teacher"
_P95_MIN_SAMPLES = 20
_ROUTES = ("curriculum", "general", "mixed", "clarify")


class LiveEvaluationRefused(RuntimeError):
    """The live provider run is not allowed in this process."""


class _EvalGate:
    """In-process test principal. This is not a login and not the local stub."""

    def verify(self, request: object) -> VerifiedTeacherClaims:
        return VerifiedTeacherClaims(_EVAL_TEACHER, "eval-school")


def dataset_path() -> Path:
    return _DATASET_PATH


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    payload = json.loads((path or _DATASET_PATH).read_text(encoding="utf-8"))
    if payload.get("version") != DATASET_VERSION:
        raise ValueError("Assistant evaluation dataset version is not supported")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Assistant evaluation dataset has no cases")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError("Assistant evaluation case ids must be unique")
        seen.add(case_id)
        if case.get("acceptance") not in {"required", "observe"}:
            raise ValueError(f"{case_id} has an unknown acceptance")
        if case.get("gold_curriculum_facts") != []:
            raise ValueError(f"{case_id} must not encode curriculum facts")
        route = case.get("expected_route")
        if route is not None and route not in _ROUTES:
            raise ValueError(f"{case_id} has an unknown expected route")
        intent = case.get("intent_route")
        if intent is not None and intent not in _ROUTES:
            raise ValueError(f"{case_id} has an unknown intent route")
    return payload


def run_deterministic(*, include_answers: bool = False) -> dict[str, Any]:
    """Route, mocked QA, isolation, and failure checks. No provider calls."""
    dataset = load_dataset()
    gate = _EvalGate()
    register_identity_gate(gate)
    try:
        with tempfile.TemporaryDirectory(prefix="assistant-eval-") as tmp:
            records = [
                _run_case(case, Path(tmp), gate, include_answers=include_answers)
                for case in dataset["cases"]
            ]
    finally:
        reset_identity_gate()
        reset_assistant_runtime()
    return _report(records, mode="deterministic", include_answers=include_answers)


def assert_live_allowed(settings: Settings) -> None:
    """Refuse live evaluation unless this process opted in and is not production."""
    if os.environ.get("ASSISTANT_EVAL_LIVE") != "1":
        raise LiveEvaluationRefused(
            "Set ASSISTANT_EVAL_LIVE=1 to run the live provider evaluation."
        )
    if str(settings.environment or "").strip().lower() == "production":
        raise LiveEvaluationRefused("Live assistant evaluation is refused in production.")
    if settings.assistant_enabled:
        raise LiveEvaluationRefused(
            "Turn ASSISTANT_ENABLED off before the live evaluation."
        )
    if settings.assistant_local_stub_identity:
        raise LiveEvaluationRefused(
            "Live evaluation does not use the local identity stub."
        )
    provider = (settings.llm_provider or "").strip().lower()
    if provider in {"", "stub"}:
        raise LiveEvaluationRefused("Live evaluation needs a non-stub LLM provider.")
    key = (settings.llm_api_key or settings.openrouter_api_key or "").strip()
    if not key:
        raise LiveEvaluationRefused("Live evaluation needs a configured provider credential.")


def run_live(settings: Settings, *, include_answers: bool = False) -> dict[str, Any]:
    """Opt-in run through the existing QA agent and configured provider.

    The HTTP assistant stays disabled. The eval gate is a local test principal,
    not Lepa authentication. Answer text is omitted unless ``include_answers``.
    """
    assert_live_allowed(settings)
    dataset = load_dataset()
    question_cases = [
        case
        for case in dataset["cases"]
        if case["mock"] not in {"isolation", "persist_failure", "provider_error", "qa_timeout"}
        and case["acceptance"] == "required"
        and case.get("expected_route")
    ]
    gate = _EvalGate()
    register_identity_gate(gate)
    reset_assistant_runtime()
    records: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="assistant-eval-live-") as tmp:
            eval_settings = settings.model_copy(
                update={
                    "assistant_enabled": False,
                    "assistant_local_stub_identity": False,
                    "assistant_thread_sqlite_path": str(Path(tmp) / "threads.sqlite"),
                }
            )
            for case in question_cases:
                records.append(
                    _run_live_case(
                        case,
                        eval_settings,
                        include_answers=include_answers,
                    )
                )
    finally:
        reset_identity_gate()
        reset_assistant_runtime()
    report = _report(records, mode="live", include_answers=include_answers)
    report["metrics"]["provider_cost"] = {
        "estimated_usd": None,
        "reason": "No price table is configured. Token counts are recorded only when the provider returns them.",
    }
    return report


def format_summary(report: dict[str, Any]) -> str:
    """Concise text summary. It does not include answer text."""
    metrics = report["metrics"]
    routing = metrics["routing_accuracy"]
    lines = [
        f"Assistant behaviour evaluation ({report['dataset']})",
        f"Mode: {report['mode']}",
        "Identity: eval-test-gate (not authentication)",
        f"Assistant enabled default: {str(report['assistant_enabled_default']).lower()}",
        (
            "Routing accuracy (required cases): "
            f"{routing['correct']}/{routing['denominator']}"
        ),
    ]
    for route in _ROUTES:
        bucket = routing["by_route"][route]
        lines.append(
            f"  {route}: {bucket['correct']}/{bucket['denominator']}"
        )
    intent = metrics["intent_agreement"]
    lines.append(
        "Intent agreement (includes observed gaps): "
        f"{intent['correct']}/{intent['denominator']}"
    )
    evidence = metrics["insufficient_or_incomplete_evidence"]
    lines.append(
        "Insufficient or coverage-rejected evidence: "
        f"{evidence['passed']}/{evidence['denominator']}"
    )
    partial = metrics["partial_evidence"]
    lines.append(
        "Partial evidence blocked from general advice: "
        f"{partial['blocked']}/{partial['denominator']}"
    )
    authority = metrics["authority_separation"]
    lines.append(
        "Curriculum authority kept separate: "
        f"{authority['passed']}/{authority['denominator']}"
    )
    gold = metrics["complete_list_gold_facts"]
    lines.append(
        "Curriculum cases encoding a gold fact list: "
        f"{gold['encoded']}/{gold['denominator']}"
    )
    factual = metrics["curriculum_factual_accuracy"]
    lines.append(f"Curriculum factual accuracy: {factual['reason']}")
    isolation = metrics["conversation_isolation"]
    lines.append(
        "Conversation isolation: "
        f"{isolation['passed']}/{isolation['denominator']}"
    )
    follow = metrics["follow_up"]
    lines.append(
        "Follow-up turns that reuse the prior question: "
        f"{follow['prior_context_used']}/{follow['denominator']}"
    )
    failure = metrics["failure_handling"]
    lines.append(
        "Provider and persistence failures: "
        f"{failure['passed']}/{failure['denominator']}"
    )
    latency = metrics["latency_ms"]
    p95 = "not reported" if latency["p95"] is None else f"{latency['p95']:.1f}"
    median = "n/a" if latency["median"] is None else f"{latency['median']:.1f}"
    lines.append(
        f"Latency ms ({latency['scope']}): n={latency['n']} median={median} p95={p95}"
    )
    cost = metrics["provider_cost"]
    lines.append(f"Provider cost: {cost['reason']}")
    gaps = [case["id"] for case in report["cases"] if case.get("gap")]
    if gaps:
        lines.append("Recorded gaps: " + ", ".join(gaps))
    return "\n".join(lines)


def _run_case(
    case: dict[str, Any],
    tmp: Path,
    gate: _EvalGate,
    *,
    include_answers: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    mock = case["mock"]
    if mock == "isolation":
        detail = _run_isolation(tmp, gate)
    elif mock == "persist_failure":
        detail = _run_persist_failure(case, tmp)
    elif mock == "follow_up_load_failure":
        detail = _run_load_failure(case, tmp)
    elif mock in {"follow_up", "follow_up_insufficient"}:
        detail = _run_follow_up(case, tmp, gate)
    else:
        detail = _run_dispatch(case)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return _record(case, detail, elapsed_ms, include_answers=include_answers)


def _run_dispatch(case: dict[str, Any]) -> dict[str, Any]:
    qa_calls: list[AssistantTurnRequest] = []
    general_calls: list[AssistantTurnRequest] = []
    mock = case["mock"]

    def qa(request: AssistantTurnRequest) -> CurriculumOutcome:
        qa_calls.append(request)
        if mock == "qa_timeout":
            raise LLMTimeoutError("timed out")
        return _outcome(mock, request.question)

    def general(request: AssistantTurnRequest) -> str:
        general_calls.append(request)
        if mock == "provider_error":
            raise LLMProviderError("provider failed")
        return _FIXTURE_GENERAL

    request = AssistantTurnRequest(question=case["question"], conversation_id=None)
    needs_qa = case["expected_route"] in {"curriculum", "mixed"}
    needs_general = case["expected_route"] in {"general", "mixed"} or mock in {
        "provider_error",
        "mixed_completed",
        "mixed_partial",
        "insufficient",
    }
    response = dispatch_turn(
        request,
        qa=qa if needs_qa or mock == "qa_timeout" else None,
        general=general if needs_general else None,
    )
    passed = _safety_passed(case, response, qa_calls, general_calls)
    official = response.official_curriculum_answer
    return {
        "actual_route": response.route.value,
        "completion_status": _completion_status(response),
        "evidence_status": _evidence_status(response),
        "verification_passed": _verification_passed(response),
        "coverage_decision": _coverage(response),
        "failure_class": _failure_class(response),
        "general_suppressed_reason": response.general_suppressed_reason,
        "safety_pass": passed,
        "qa_called": bool(qa_calls),
        "general_called": bool(general_calls),
        "conversation_id": qa_calls[0].conversation_id if qa_calls else None,
        "question_sent": qa_calls[0].question if qa_calls else None,
        "answer_text": official,
        "general_text": response.general.text if response.general is not None else None,
    }


def _safety_passed(case: dict[str, Any], response, qa_calls, general_calls) -> bool:
    mock = case["mock"]
    if response.route.value != case["expected_route"]:
        return False
    if qa_calls and qa_calls[0].conversation_id is not None:
        return False
    if qa_calls and qa_calls[0].question != case["question"]:
        return False
    if mock == "clarify":
        return not qa_calls and not general_calls and response.clarification is not None
    if mock == "general":
        general = response.general
        return (
            not qa_calls
            and bool(general_calls)
            and response.curriculum is None
            and general is not None
            and general.source == GENERAL_SOURCE
            and general.label == GENERAL_LABEL
        )
    if mock == "provider_error":
        return (
            not qa_calls
            and response.curriculum is None
            and response.general is None
            and response.general_failure is not None
            and _failure_class(response) == "provider_error"
        )
    if mock == "qa_timeout":
        return (
            not general_calls
            and _failure_class(response) == "timeout"
            and response.general is None
        )
    if mock in {"insufficient", "coverage_reject", "needs_clarification"}:
        return not general_calls and response.general is None
    if mock == "completed":
        official = response.official_curriculum_answer or ""
        return (
            not general_calls
            and response.general is None
            and response.curriculum is not None
            and response.curriculum.source == "mbsse_curriculum"
            and _FIXTURE_GENERAL not in official
        )
    if mock == "mixed_completed":
        official = response.official_curriculum_answer or ""
        general = response.general
        return (
            bool(qa_calls)
            and bool(general_calls)
            and general is not None
            and general.label == GENERAL_LABEL
            and general.source == GENERAL_SOURCE
            and _FIXTURE_GENERAL not in official
            and official == _FIXTURE_CURRICULUM
        )
    if mock == "mixed_partial":
        official = response.official_curriculum_answer or ""
        return (
            bool(qa_calls)
            and not general_calls
            and response.general is None
            and response.general_suppressed_reason == "partial_evidence"
            and response.curriculum is not None
            and response.curriculum.response is not None
            and response.curriculum.response.status == "completed"
            and response.curriculum.response.metadata.evidence_status == "partial"
            and _FIXTURE_GENERAL not in official
        )
    return False


def _run_isolation(tmp: Path, gate: _EvalGate) -> dict[str, Any]:
    store = AssistantThreadStore(tmp / "isolation.sqlite")
    owner = bind_verified_claims(VerifiedTeacherClaims(_EVAL_TEACHER, "eval-school"), gate=gate)
    other = bind_verified_claims(VerifiedTeacherClaims(_OTHER_TEACHER, "eval-school"), gate=gate)
    try:
        created = store.create_thread(owner)
        store.append_message(owner, created.thread_id, "user", "private note")
        blocked_read = False
        blocked_append = False
        try:
            store.get_thread(other, created.thread_id)
        except ThreadUnavailable:
            blocked_read = True
        try:
            store.append_message(other, created.thread_id, "user", "overwrite")
        except ThreadUnavailable:
            blocked_append = True
        raw_rejected = False
        try:
            store.create_thread("teacher-from-client")  # type: ignore[arg-type]
        except Exception:
            raw_rejected = True
        passed = blocked_read and blocked_append and raw_rejected
        return {
            "actual_route": None,
            "completion_status": "isolated" if passed else "isolation_failed",
            "evidence_status": None,
            "verification_passed": None,
            "coverage_decision": None,
            "failure_class": None if passed else "isolation",
            "general_suppressed_reason": None,
            "safety_pass": passed,
            "qa_called": False,
            "general_called": False,
            "conversation_id": None,
            "question_sent": None,
            "answer_text": None,
            "general_text": None,
        }
    finally:
        store.close()


def _run_persist_failure(case: dict[str, Any], tmp: Path) -> dict[str, Any]:
    calls = {"general": 0}

    def general(_request: AssistantTurnRequest) -> str:
        calls["general"] += 1
        return _FIXTURE_GENERAL

    configure_assistant_runtime(general=general)
    real = AssistantThreadStore.append_message

    def fail_assistant(self, principal, thread_id, role, content):
        if role == "assistant":
            raise ThreadStoreError("Assistant thread store is unavailable")
        return real(self, principal, thread_id, role, content)

    AssistantThreadStore.append_message = fail_assistant  # type: ignore[method-assign]
    settings = _eval_settings(tmp / "persist.sqlite")
    try:
        response = execute_chat(
            AssistantChatRequest(question=case["question"]),
            _request(),
            settings=settings,
        )
    finally:
        AssistantThreadStore.append_message = real  # type: ignore[method-assign]
        reset_assistant_runtime()
    passed = (
        response.route == "general"
        and response.persistence_warning == "The reply was produced but not fully stored."
        and response.curriculum is None
        and calls["general"] == 1
    )
    return {
        "actual_route": response.route,
        "completion_status": "persistence_warning" if response.persistence_warning else response.route,
        "evidence_status": None,
        "verification_passed": None,
        "coverage_decision": None,
        "failure_class": "persistence" if response.persistence_warning else None,
        "general_suppressed_reason": None,
        "safety_pass": passed,
        "qa_called": False,
        "general_called": calls["general"] > 0,
        "conversation_id": None,
        "question_sent": None,
        "answer_text": None,
        "general_text": response.general_text,
    }


def _run_follow_up(case: dict[str, Any], tmp: Path, gate: _EvalGate) -> dict[str, Any]:
    store = AssistantThreadStore(tmp / f"{case['id']}.sqlite")
    owner = bind_verified_claims(VerifiedTeacherClaims(_EVAL_TEACHER, "eval-school"), gate=gate)
    other = bind_verified_claims(VerifiedTeacherClaims(_OTHER_TEACHER, "eval-school"), gate=gate)
    qa_calls: list[AssistantTurnRequest] = []
    general_calls: list[AssistantTurnRequest] = []
    try:
        created = store.create_thread(owner)
        prior = case.get("prior_question")
        if prior:
            store.append_message(owner, created.thread_id, "user", prior)
            if case["mock"] == "follow_up_insufficient":
                store.record_qa_outcome(
                    owner,
                    created.thread_id,
                    QaOutcomeSummary(
                        status="insufficient_evidence",
                        evidence_status="not_found",
                    ),
                )
        record = store.get_thread(owner, created.thread_id)
        resolution = resolve_context_follow_up(
            case["question"],
            record.messages,
            record.qa_outcome,
        )
        store.append_message(owner, created.thread_id, "user", case["question"])
        expected = case.get("expected_route")

        def qa(request: AssistantTurnRequest) -> CurriculumOutcome:
            qa_calls.append(request)
            kind = "insufficient" if case["mock"] == "follow_up_insufficient" else "completed"
            return _outcome(kind, request.question)

        def general(request: AssistantTurnRequest) -> str:
            general_calls.append(request)
            return _FIXTURE_GENERAL

        response = dispatch_turn(
            AssistantTurnRequest(
                question=resolution.dispatch_question,
                conversation_id=None,
            ),
            qa=qa if expected in {"curriculum", "mixed"} else None,
            general=general if expected in {"general", "mixed"} else None,
        )
        loaded = store.get_thread(owner, created.thread_id)
        hidden = False
        try:
            store.get_thread(other, created.thread_id)
        except ThreadUnavailable:
            hidden = True
        sent = qa_calls[0].question if qa_calls else resolution.dispatch_question
        kept_topic = True
        if prior and resolution.used_prior and expected == "curriculum":
            kept_topic = "primary 4" in sent and "mathematics" in sent and sent != prior
        passed = (
            response.route.value == expected
            and hidden
            and resolution.used_prior is bool(prior)
            and [message.content for message in loaded.messages][-1] == case["question"]
            and (not prior or loaded.messages[0].content == prior)
            and kept_topic
            and _FIXTURE_GENERAL not in (response.official_curriculum_answer or "")
            and created.thread_id != "checkpoints.sqlite"
        )
        if case["mock"] == "follow_up_insufficient":
            passed = passed and not general_calls and response.general is None
        if expected == "general":
            passed = passed and not qa_calls and bool(general_calls)
        if expected == "clarify":
            passed = passed and not qa_calls and not general_calls
        return {
            "actual_route": response.route.value,
            "completion_status": _completion_status(response),
            "evidence_status": _evidence_status(response),
            "verification_passed": _verification_passed(response),
            "coverage_decision": _coverage(response),
            "failure_class": None,
            "general_suppressed_reason": response.general_suppressed_reason,
            "safety_pass": passed,
            "qa_called": bool(qa_calls),
            "general_called": bool(general_calls),
            "conversation_id": qa_calls[0].conversation_id if qa_calls else None,
            "question_sent": sent,
            "prior_context_used": resolution.used_prior,
            "answer_text": response.official_curriculum_answer,
            "general_text": response.general.text if response.general is not None else None,
        }
    finally:
        store.close()


def _run_load_failure(case: dict[str, Any], tmp: Path) -> dict[str, Any]:
    calls = {"n": 0}

    def general(_request: AssistantTurnRequest) -> str:
        calls["n"] += 1
        return _FIXTURE_GENERAL

    class _Agent:
        llm = None

        def ask(self, *_args, **_kwargs):
            calls["n"] += 1
            raise AssertionError("qa called during a load failure")

    configure_assistant_runtime(qa_agent=_Agent(), general=general)
    path = tmp / f"{case['id']}.sqlite"
    settings = _eval_settings(path)
    store = AssistantThreadStore(path)
    owner = bind_verified_claims(
        VerifiedTeacherClaims(_EVAL_TEACHER, "eval-school"),
        gate=current_identity_gate(),
    )
    created = store.create_thread(owner)
    store.append_message(owner, created.thread_id, "user", case["prior_question"])
    store.close()
    real = AssistantThreadStore.get_thread

    def fail_load(self, principal, thread_id):
        raise ThreadStoreError("Assistant thread store is unavailable")

    AssistantThreadStore.get_thread = fail_load  # type: ignore[method-assign]
    raised = False
    try:
        try:
            execute_chat(
                AssistantChatRequest(question=case["question"], thread_id=created.thread_id),
                _request(),
                settings=settings,
            )
        except ThreadStoreError:
            raised = True
    finally:
        AssistantThreadStore.get_thread = real  # type: ignore[method-assign]
        reset_assistant_runtime()
    return {
        "actual_route": None,
        "completion_status": "load_failed" if raised else "load_failure_missed",
        "evidence_status": None,
        "verification_passed": None,
        "coverage_decision": None,
        "failure_class": "persistence" if raised else None,
        "general_suppressed_reason": None,
        "safety_pass": raised and calls["n"] == 0,
        "qa_called": False,
        "general_called": False,
        "conversation_id": None,
        "question_sent": None,
        "prior_context_used": False,
        "answer_text": None,
        "general_text": None,
    }


def _run_live_case(
    case: dict[str, Any],
    settings: Settings,
    *,
    include_answers: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = execute_chat(
        AssistantChatRequest(question=case["question"]),
        _request(),
        settings=settings,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    curriculum = response.curriculum
    detail = {
        "actual_route": response.route,
        "completion_status": curriculum.status if curriculum is not None else response.route,
        "evidence_status": (
            curriculum.metadata.evidence_status if curriculum is not None else None
        ),
        "verification_passed": (
            curriculum.verification.passed
            if curriculum is not None and curriculum.verification is not None
            else None
        ),
        "coverage_decision": None,
        "failure_class": "persistence" if response.persistence_warning else None,
        "general_suppressed_reason": None,
        "safety_pass": response.thread_id not in (response.general_text or ""),
        "qa_called": curriculum is not None,
        "general_called": response.general_text is not None,
        "conversation_id": curriculum.conversation_id if curriculum is not None else None,
        "question_sent": case["question"],
        "answer_text": curriculum.answer if curriculum is not None else None,
        "general_text": response.general_text,
        "thread_distinct": (
            curriculum is None or response.thread_id != curriculum.conversation_id
        ),
    }
    if detail["conversation_id"] is not None and not detail["thread_distinct"]:
        detail["safety_pass"] = False
    return _record(case, detail, elapsed_ms, include_answers=include_answers)


def _outcome(mock: str, question: str) -> CurriculumOutcome:
    if mock == "insufficient":
        return CurriculumOutcome(
            http_status=200,
            response=_ask(
                question,
                answer="Fixture insufficient-evidence sentence.",
                status="insufficient_evidence",
                evidence_status="not_found",
                passed=False,
            ),
        )
    if mock == "coverage_reject":
        return CurriculumOutcome(
            http_status=200,
            coverage_decision="reject",
            response=_ask(
                question,
                answer=_FIXTURE_CURRICULUM,
                status="completed",
                evidence_status="partial",
                passed=False,
            ),
        )
    if mock == "needs_clarification":
        return CurriculumOutcome(
            http_status=200,
            response=_ask(
                question,
                answer=None,
                status="needs_clarification",
                evidence_status="not_found",
                passed=False,
                clarification="Which grade should I use?",
            ),
        )
    if mock == "mixed_partial":
        return CurriculumOutcome(
            http_status=200,
            coverage_decision="accept",
            response=_ask(
                question,
                answer=_FIXTURE_CURRICULUM,
                status="completed",
                evidence_status="partial",
                passed=True,
            ),
        )
    return CurriculumOutcome(
        http_status=200,
        coverage_decision="accept",
        response=_ask(
            question,
            answer=_FIXTURE_CURRICULUM,
            status="completed",
            evidence_status="found",
            passed=True,
        ),
    )


def _ask(
    question: str,
    *,
    answer: str | None,
    status: str,
    evidence_status: str,
    passed: bool,
    clarification: str | None = None,
) -> AskResponse:
    return AskResponse(
        conversation_id="qa-fixture-conversation",
        question=question,
        answer=answer,
        status=status,
        clarification=clarification,
        verification=VerificationSummary(
            passed=passed,
            score=1.0 if passed else 0.0,
            recommendation="accept" if passed else "fallback",
        ),
        metadata=AskMetadata(
            evidence_status=evidence_status,
            verification_status="passed" if passed else status,
            termination_reason="verification_passed" if passed else status,
        ),
    )


def _record(
    case: dict[str, Any],
    detail: dict[str, Any],
    elapsed_ms: float,
    *,
    include_answers: bool,
) -> dict[str, Any]:
    expected = case.get("expected_route")
    actual = detail.get("actual_route")
    intent = case.get("intent_route")
    record = {
        "id": case["id"],
        "group": case["group"],
        "acceptance": case["acceptance"],
        "expected_route": expected,
        "actual_route": actual,
        "route_match": expected == actual if expected is not None else None,
        "intent_route": intent,
        "intent_match": intent == actual if intent is not None else None,
        "completion_status": detail.get("completion_status"),
        "evidence_status": detail.get("evidence_status"),
        "verification_passed": detail.get("verification_passed"),
        "coverage_decision": detail.get("coverage_decision"),
        "failure_class": detail.get("failure_class"),
        "general_suppressed_reason": detail.get("general_suppressed_reason"),
        "safety_pass": detail.get("safety_pass"),
        "qa_called": detail.get("qa_called"),
        "general_called": detail.get("general_called"),
        "prior_context_used": detail.get("prior_context_used", False),
        "latency_ms": round(elapsed_ms, 3),
        "gap": case.get("gap"),
        "has_prior": bool(case.get("prior_question")),
    }
    if include_answers:
        record["answer_text"] = detail.get("answer_text")
        record["general_text"] = detail.get("general_text")
    return record


def _report(
    records: list[dict[str, Any]],
    *,
    mode: str,
    include_answers: bool,
) -> dict[str, Any]:
    required = [row for row in records if row["acceptance"] == "required"]
    routed = [row for row in required if row["expected_route"] is not None]
    intent_rows = [row for row in records if row["intent_route"] is not None]
    evidence_ids = {"curr-astronomy-missing", "curr-french-coverage", "mixed-outcomes-missing"}
    evidence_rows = [row for row in records if row["id"] in evidence_ids]
    authority_ids = {
        "curr-p4-math-topics",
        "mixed-fractions-teach",
        "gen-lesson-idea",
        "gen-noisy-class",
        "mixed-outcomes-missing",
    }
    authority_rows = [row for row in required if row["id"] in authority_ids]
    curriculum_rows = [row for row in records if row["group"] == "curriculum"]
    isolation_rows = [row for row in records if row["group"] == "isolation"]
    follow_rows = [
        row for row in records if row["group"] == "follow_up" and row.get("has_prior")
    ]
    failure_rows = [row for row in records if row["group"] == "failure"]
    partial_rows = [row for row in records if row["id"] == "obs-partial-allows-advice"]
    latencies = [float(row["latency_ms"]) for row in records]
    return {
        "dataset": DATASET_VERSION,
        "mode": mode,
        "identity_mode": "eval-test-gate",
        "production_authentication": False,
        "assistant_enabled_default": Settings.model_fields["assistant_enabled"].default,
        "include_answers": include_answers,
        "cases": records,
        "metrics": {
            "routing_accuracy": {
                "correct": sum(1 for row in routed if row["route_match"]),
                "denominator": len(routed),
                "by_route": {
                    route: _route_bucket(routed, route) for route in _ROUTES
                },
            },
            "intent_agreement": {
                "correct": sum(1 for row in intent_rows if row["intent_match"]),
                "denominator": len(intent_rows),
            },
            "insufficient_or_incomplete_evidence": {
                "passed": sum(1 for row in evidence_rows if row["safety_pass"] and not row["general_called"]),
                "denominator": len(evidence_rows),
            },
            "partial_evidence": {
                "blocked": sum(1 for row in partial_rows if not row["general_called"]),
                "denominator": len(partial_rows),
            },
            "authority_separation": {
                "passed": sum(1 for row in authority_rows if row["safety_pass"]),
                "denominator": len(authority_rows),
            },
            "complete_list_gold_facts": {
                "encoded": 0,
                "denominator": len(curriculum_rows),
            },
            "curriculum_factual_accuracy": {
                "measured": False,
                "correct": None,
                "denominator": None,
                "reason": (
                    "Unmeasured. No gold set has been verified against "
                    "authoritative curriculum records."
                ),
            },
            "conversation_isolation": {
                "passed": sum(1 for row in isolation_rows if row["safety_pass"]),
                "denominator": len(isolation_rows),
            },
            "follow_up": {
                "prior_context_used": sum(1 for row in follow_rows if row["prior_context_used"]),
                "denominator": len(follow_rows),
            },
            "failure_handling": {
                "passed": sum(1 for row in failure_rows if row["safety_pass"]),
                "denominator": len(failure_rows),
            },
            "latency_ms": {
                "scope": "harness" if mode == "deterministic" else "live_turn",
                "n": len(latencies),
                "median": statistics.median(latencies) if latencies else None,
                "p95": _p95(latencies),
            },
            "provider_cost": {
                "estimated_usd": None,
                "reason": "Not estimated. This run did not call a priced provider and no price table is configured.",
            },
        },
    }


def _route_bucket(rows: list[dict[str, Any]], route: str) -> dict[str, int]:
    subset = [row for row in rows if row["expected_route"] == route]
    return {
        "correct": sum(1 for row in subset if row["route_match"]),
        "denominator": len(subset),
    }


def _p95(values: list[float]) -> float | None:
    if len(values) < _P95_MIN_SAMPLES:
        return None
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _completion_status(response) -> str | None:
    if response.curriculum is None or response.curriculum.response is None:
        return response.route.value
    return response.curriculum.response.status


def _evidence_status(response) -> str | None:
    if response.curriculum is None or response.curriculum.response is None:
        return None
    return response.curriculum.response.metadata.evidence_status


def _verification_passed(response) -> bool | None:
    if response.curriculum is None or response.curriculum.response is None:
        return None
    verification = response.curriculum.response.verification
    if verification is None:
        return None
    return verification.passed


def _coverage(response) -> str | None:
    if response.curriculum is None:
        return None
    return response.curriculum.coverage_decision


def _failure_class(response) -> str | None:
    if response.curriculum is not None and response.curriculum.failure is not None:
        return response.curriculum.failure.kind
    if response.general_failure is not None:
        return "provider_error"
    return None


def _eval_settings(path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="development",
        assistant_enabled=False,
        assistant_local_stub_identity=False,
        assistant_thread_sqlite_path=str(path),
        llm_provider="stub",
    )


def _request() -> Any:
    state = type("State", (), {"request_id": "eval-request"})()
    return type("Request", (), {"state": state})()


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
