"""Verifier payload diagnostics stay in the trace and do not change the verdict."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.graph_routing import route_after_verification
from app.agent.state import CurriculumQAState
from app.agent.trace import start_agent_run
from app.agent.verifier import AnswerVerifier
from app.agent.verifier_diagnostics import field_presence, schema_problems, stash_verifier_model_text
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.exceptions import LLMProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResponse
from app.schemas.verification import (
    VERIFICATION_RESULT_JSON_SCHEMA,
    MissingEvidenceRequest,
    VerificationRecommendation,
    VerificationResult,
)


def _settings() -> Settings:
    return Settings(_env_file=None, curriculum_api_base_url="http://curriculum.test")


def _state() -> CurriculumQAState:
    qa = CurriculumQAState.initial(
        question="List every published Class 4 Science unit and theme, with its content code."
    )
    qa.grade = "CLASS_4"
    qa.subject = "SCIENCE"
    qa.draft_answer = "Living and Non – living Things — C4-U01"
    qa.evidence_status = EvidenceStatus.FOUND
    qa.evidence = [
        CurriculumEvidence(
            entity_type="unit",
            entity_id="unit-1",
            name="Living and Non – living Things",
            grade="CLASS_4",
            subject="SCIENCE",
            content="Living and Non – living Things",
            metadata={"code": "C4-U01"},
        )
    ]
    return qa


class _ScriptedLLM(LLMProvider):
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        error: Exception | None = None,
        raw_text: str | None = None,
        json_extracted: bool = True,
    ) -> None:
        self.payload = payload or {}
        self.error = error
        self.raw_text = raw_text
        self.json_extracted = json_extracted

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def model(self) -> str:
        return "scripted-model"

    def generate(self, messages: list[LLMMessage], *, temperature: float = 0.0, max_tokens: int | None = None) -> LLMResponse:
        raise NotImplementedError

    def generate_with_tools(self, messages, *, tools, temperature: float = 0.0) -> LLMResponse:
        raise NotImplementedError

    def generate_structured(self, messages, *, schema, temperature: float = 0.0) -> dict[str, Any]:
        if self.raw_text is not None:
            stash_verifier_model_text(
                self.raw_text,
                provider=self.name,
                model=self.model,
                temperature=temperature,
                max_output_tokens=8192,
                json_extracted=self.json_extracted,
                error=self.error,
            )
        if self.error is not None:
            raise self.error
        return self.payload


def _complete(**overrides: Any) -> dict[str, Any]:
    payload = {
        "passed": True,
        "score": 1.0,
        "issues": [],
        "unsupported_claims": [],
        "incorrect_claims": [],
        "missing_evidence": [],
        "recommendation": "accept",
    }
    payload.update(overrides)
    return payload


def test_raw_payload_is_stored_on_the_trace_and_not_in_logs(caplog):
    raw = '{"recommendation":"accept","notes":"SECRET_ANSWER_TEXT"}'
    with caplog.at_level(logging.INFO):
        with start_agent_run(question="science units", request_id="req-1", persist=False) as trace:
            AnswerVerifier(
                _ScriptedLLM(_complete(notes="SECRET_ANSWER_TEXT"), raw_text=raw),
                settings=_settings(),
            ).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert event is not None
    assert event["raw_response"] == raw
    assert "SECRET_ANSWER_TEXT" in json.dumps(trace.to_dict())
    assert "SECRET_ANSWER_TEXT" not in caplog.text
    assert event["raw_response_fingerprint"]
    assert event["prompt_sha256"]
    assert event["evidence_ids"] == ["unit-1"]
    assert event["temperature"] == 0.0
    assert event["model"] == "scripted-model"


def test_field_presence_distinguishes_missing_null_and_empty():
    assert field_presence({}, "notes") == "missing"
    assert field_presence({"notes": None}, "notes") == "null"
    assert field_presence({"notes": ""}, "notes") == "empty_string"
    assert field_presence({"issues": []}, "issues") == "empty_array"
    assert field_presence({"notes": "kept"}, "notes") == "present"

    payload = _complete(passed=False, score=0)
    del payload["recommendation"]
    payload["notes"] = None
    payload["issues"] = []
    with start_agent_run(question="science units", request_id="req-2", persist=False) as trace:
        result = AnswerVerifier(_ScriptedLLM(payload), settings=_settings()).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert event["fields"]["recommendation"] == "missing"
    assert event["fields"]["notes"] == "null"
    assert event["fields"]["issues"] == "empty_array"
    assert event["parsed_recommendation"] is None
    assert result.recommendation == VerificationRecommendation.FALLBACK
    assert event["recommendation_before_normalize"] == "fallback"
    assert event["recommendation_after_normalize"] == "fallback"
    assert "missing:recommendation" in event["schema_problems"]
    assert event["schema_valid"] is False


def test_explicit_fallback_stays_fallback_and_accept_is_not_rewritten():
    with start_agent_run(question="science units", request_id="req-3", persist=False) as trace:
        fallback = AnswerVerifier(
            _ScriptedLLM(_complete(passed=False, score=0.0, recommendation="fallback")),
            settings=_settings(),
        ).verify(_state())
    assert fallback.recommendation == VerificationRecommendation.FALLBACK
    event = trace.latest_event("agent.verification.payload")
    assert event["fields"]["recommendation"] == "present"
    assert event["parsed_recommendation"] == "fallback"
    assert event["recommendation_after_normalize"] == "fallback"

    verifier = AnswerVerifier(_ScriptedLLM(_complete()), settings=_settings())
    accepted = VerificationResult(
        passed=True,
        score=1.0,
        recommendation=VerificationRecommendation.ACCEPT,
    )
    normalized = verifier._normalize_recommendation(_state(), accepted)
    assert normalized.recommendation == VerificationRecommendation.ACCEPT

    contradicted = VerificationResult(
        passed=False,
        score=0.2,
        recommendation=VerificationRecommendation.ACCEPT,
        missing_evidence=[MissingEvidenceRequest(topic="C4-U12")],
    )
    rewritten = verifier._normalize_recommendation(_state(), contradicted)
    assert rewritten.recommendation == VerificationRecommendation.RETRIEVE_MORE


def test_malformed_json_is_recorded_without_changing_the_exception_path():
    raw = "not-json SECRET_BROKEN_BODY"
    with start_agent_run(question="science units", request_id="req-4", persist=False) as trace:
        AnswerVerifier(
            _ScriptedLLM(
                error=LLMProviderError("LLM returned invalid JSON"),
                raw_text=raw,
                json_extracted=False,
            ),
            settings=_settings(),
        ).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert event["json_extracted"] is False
    assert event["schema_valid"] is False
    assert event["raw_response"] == raw
    assert event["exception"]["category"] == "LLMProviderError"
    assert "invalid JSON" in event["exception"]["message"]


def test_schema_problems_do_not_reject_a_usable_recommendation():
    payload = _complete(extra_field="ignore-me")
    problems = schema_problems(payload)
    assert "unexpected:extra_field" in problems
    with start_agent_run(question="science units", request_id="req-5", persist=False) as trace:
        result = AnswerVerifier(_ScriptedLLM(payload), settings=_settings()).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert event["schema_valid"] is False
    assert result.recommendation == VerificationRecommendation.ACCEPT
    assert result.passed is True


def test_tracing_disabled_does_not_record_or_raise():
    stash_verifier_model_text(
        "SECRET_ANSWER_TEXT",
        provider="scripted",
        model="scripted-model",
        temperature=0.0,
        max_output_tokens=8192,
        json_extracted=True,
    )
    result = AnswerVerifier(_ScriptedLLM(_complete()), settings=_settings()).verify(_state())
    assert result.recommendation == VerificationRecommendation.ACCEPT


def test_route_and_fallback_reason_are_written_onto_the_payload():
    with start_agent_run(question="science units", request_id="req-6", persist=False) as trace:
        qa = _state()
        AnswerVerifier(
            _ScriptedLLM(_complete(passed=False, score=0.0, recommendation="fallback")),
            settings=_settings(),
        ).verify(qa)
        qa.verification = VerificationResult(
            passed=False,
            score=0.0,
            recommendation=VerificationRecommendation.FALLBACK,
        )
        decision = route_after_verification({"qa": qa}, settings=_settings())
    assert decision == "fallback"
    event = trace.latest_event("agent.verification.payload")
    assert event["final_route"] == "fallback"
    assert event["fallback_reason"] == "verification_fallback"
    assert VERIFICATION_RESULT_JSON_SCHEMA["title"] == "VerificationResult"
