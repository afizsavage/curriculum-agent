"""Verifier payload diagnostics stay in the trace and do not change the verdict."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.graph_routing import route_after_verification
from app.agent.state import CurriculumQAState
from app.agent.trace import start_agent_run
from app.agent.verifier import AnswerVerifier
from app.agent.verifier_diagnostics import (
    classify_schema,
    field_presence,
    fingerprint,
    redact_text,
    schema_problems,
    stash_verifier_model_text,
)
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
        record_raw: bool = False,
    ) -> None:
        self.payload = payload or {}
        self.error = error
        self.raw_text = raw_text
        self.json_extracted = json_extracted
        self.record_raw = record_raw

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
        if self.record_raw or self.raw_text is not None:
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
    payload = _complete(title="VerificationResult")
    classified = classify_schema(payload)
    assert classified["schema_problems"] == ["unexpected:title"]
    assert classified["extra_fields"] == ["title"]
    assert classified["interpretation_problems"] == []
    assert classified["schema_valid"] is False
    assert classified["safely_interpretable"] is True
    assert "unexpected:title" in schema_problems(payload)
    with start_agent_run(question="science units", request_id="req-5", persist=False) as trace:
        result = AnswerVerifier(_ScriptedLLM(payload), settings=_settings()).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert event["schema_valid"] is False
    assert event["safely_interpretable"] is True
    assert event["extra_fields"] == ["title"]
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


def _public(result: VerificationResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "score": result.score,
        "recommendation": result.recommendation.value,
        "issues": list(result.issues),
        "unsupported_claims": list(result.unsupported_claims),
        "incorrect_claims": list(result.incorrect_claims),
        "missing_evidence": list(result.missing_evidence),
    }


def test_explicit_fallback_and_omitted_fields_share_a_result_but_not_a_payload():
    explicit = _complete(
        passed=False,
        score=0,
        recommendation="fallback",
        issues=[],
        unsupported_claims=[],
        incorrect_claims=[],
        missing_evidence=[],
    )
    omitted = {
        "issues": [],
        "unsupported_claims": [],
        "incorrect_claims": [],
        "missing_evidence": [],
    }
    with start_agent_run(question="science units", request_id="req-a", persist=False) as trace_a:
        result_a = AnswerVerifier(_ScriptedLLM(explicit), settings=_settings()).verify(_state())
    with start_agent_run(question="science units", request_id="req-b", persist=False) as trace_b:
        result_b = AnswerVerifier(_ScriptedLLM(omitted), settings=_settings()).verify(_state())
    assert _public(result_a) == _public(result_b)
    assert result_a.recommendation == VerificationRecommendation.FALLBACK
    event_a = trace_a.latest_event("agent.verification.payload")
    event_b = trace_b.latest_event("agent.verification.payload")
    assert event_a["fields"]["recommendation"] == "present"
    assert event_a["fields"]["passed"] == "present"
    assert event_a["fields"]["score"] == "present"
    assert event_a["parsed_recommendation"] == "fallback"
    assert event_a["safely_interpretable"] is True
    assert event_b["fields"]["recommendation"] == "missing"
    assert event_b["fields"]["passed"] == "missing"
    assert event_b["fields"]["score"] == "missing"
    assert event_b["parsed_recommendation"] is None
    assert event_b["safely_interpretable"] is False
    assert "missing:recommendation" in event_b["interpretation_problems"]
    assert event_b["extra_fields"] == []


def test_null_recommendation_is_not_recorded_as_missing_or_explicit():
    payload = _complete(passed=False, score=0, recommendation=None)
    with start_agent_run(question="science units", request_id="req-null", persist=False) as trace:
        result = AnswerVerifier(_ScriptedLLM(payload), settings=_settings()).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert event["fields"]["recommendation"] == "null"
    assert event["parsed_recommendation"] is None
    assert "recommendation_not_in_enum" in event["interpretation_problems"]
    assert event["safely_interpretable"] is False
    assert result.recommendation == VerificationRecommendation.FALLBACK


def test_empty_response_and_unreadable_recommendation_stay_on_the_existing_path():
    with start_agent_run(question="science units", request_id="req-empty", persist=False) as trace:
        AnswerVerifier(
            _ScriptedLLM(
                error=LLMProviderError("Empty structured response from LLM"),
                raw_text=None,
                json_extracted=False,
                record_raw=True,
            ),
            settings=_settings(),
        ).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert event["raw_response_present"] is False
    assert event["raw_response_length"] == 0
    assert event["raw_response_fingerprint"] is None
    assert event["json_extracted"] is False
    assert event["safely_interpretable"] is False

    unreadable = _complete(passed=False, score=0, recommendation="approve")
    with start_agent_run(question="science units", request_id="req-bad", persist=False) as trace:
        result = AnswerVerifier(_ScriptedLLM(unreadable), settings=_settings()).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert "recommendation_not_in_enum" in event["interpretation_problems"]
    assert event["safely_interpretable"] is False
    assert result.recommendation == VerificationRecommendation.FALLBACK


def test_optional_fields_may_be_absent_from_a_valid_recommendation():
    payload = _complete()
    assert "notes" not in payload
    classified = classify_schema(payload)
    assert classified["schema_valid"] is True
    assert classified["safely_interpretable"] is True
    assert classified["interpretation_problems"] == []
    with start_agent_run(question="science units", request_id="req-opt", persist=False) as trace:
        result = AnswerVerifier(_ScriptedLLM(payload), settings=_settings()).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert event["fields"]["notes"] == "missing"
    assert event["fields"]["recommendation"] == "present"
    assert result.recommendation == VerificationRecommendation.ACCEPT


def test_redacted_text_is_what_the_trace_stores_and_hashes(caplog):
    secret = "sk-supersecretkeyvalue"
    raw = f'{{"notes":"token {secret}","authorization":"Bearer abcdefghijklmnop"}}'
    stored = redact_text(raw)
    assert secret not in stored
    assert "abcdefghijklmnop" not in stored
    with caplog.at_level(logging.INFO):
        with start_agent_run(question="science units", request_id="req-secret", persist=False) as trace:
            AnswerVerifier(
                _ScriptedLLM(_complete(), raw_text=raw),
                settings=_settings(),
            ).verify(_state())
    event = trace.latest_event("agent.verification.payload")
    assert event["raw_response"] == stored
    assert event["raw_response_length"] == len(stored)
    assert event["raw_response_fingerprint"] == fingerprint(stored)
    assert event["fingerprint_basis"] == "redacted_stored_text"
    encoded = json.dumps(trace.to_dict())
    assert secret not in encoded
    assert "abcdefghijklmnop" not in encoded
    assert secret not in caplog.text
    assert "abcdefghijklmnop" not in caplog.text
    assert stored not in caplog.text


def test_missing_diagnostic_event_does_not_change_the_route():
    qa = _state()
    qa.verification = VerificationResult(
        passed=False,
        score=0.0,
        recommendation=VerificationRecommendation.FALLBACK,
    )
    with start_agent_run(question="science units", request_id="req-noraw", persist=False) as trace:
        decision = route_after_verification({"qa": qa}, settings=_settings())
        encoded = json.dumps(trace.to_dict())
    assert decision == "fallback"
    assert trace.latest_event("agent.verification.payload") is None
    assert "sk-" not in encoded
