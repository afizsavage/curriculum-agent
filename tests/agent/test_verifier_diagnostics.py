"""Verifier payload diagnostics stay in the trace and do not change the verdict."""

from __future__ import annotations

import json
import logging
import math
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


def _verdict(payload: dict[str, Any]):
    qa = _state()
    qa.topic = "fractions"
    with start_agent_run(question="science units", request_id="req-strict", persist=False) as trace:
        result = AnswerVerifier(_ScriptedLLM(payload), settings=_settings()).verify(qa)
        qa.verification = result
        decision = route_after_verification({"qa": qa}, settings=_settings())
    return result, decision, trace.latest_event("agent.verification.payload")


def test_string_passed_cannot_become_accept():
    result, decision, event = _verdict(
        _complete(passed="yes", score=0, recommendation="approve")
    )
    assert result.passed is False
    assert result.recommendation == VerificationRecommendation.FALLBACK
    assert result.recommendation != VerificationRecommendation.ACCEPT
    assert decision == "fallback"
    assert event["fields"]["passed"] == "present"
    assert event["parsed_passed"] == "yes"
    assert event["parsed_recommendation"] == "approve"
    assert event["safely_interpretable"] is False
    assert "passed_not_boolean" in event["interpretation_problems"]
    assert "recommendation_not_in_enum" in event["interpretation_problems"]
    assert result.metadata["verifier_result_invalid"] is True


def test_numeric_passed_is_not_a_json_boolean():
    for value in (1, 0):
        result, decision, event = _verdict(_complete(passed=value, recommendation="accept"))
        assert result.passed is False
        assert result.recommendation == VerificationRecommendation.FALLBACK
        assert decision == "fallback"
        assert event["parsed_passed"] == value
        assert "passed_not_boolean" in event["interpretation_problems"]
        assert event["safely_interpretable"] is False


def test_missing_and_null_passed_stay_distinct_from_explicit_false():
    missing, _, missing_event = _verdict(
        {key: value for key, value in _complete(recommendation="accept").items() if key != "passed"}
    )
    null_result, _, null_event = _verdict(_complete(passed=None, recommendation="accept"))
    explicit, explicit_route, explicit_event = _verdict(
        _complete(passed=False, score=0.2, recommendation="accept")
    )
    assert missing_event["fields"]["passed"] == "missing"
    assert null_event["fields"]["passed"] == "null"
    assert explicit_event["fields"]["passed"] == "present"
    assert explicit_event["parsed_passed"] is False
    assert missing_event["parsed_passed"] is None
    assert null_event["parsed_passed"] is None
    assert missing.recommendation == VerificationRecommendation.FALLBACK
    assert null_result.recommendation == VerificationRecommendation.FALLBACK
    assert explicit.passed is False
    assert explicit.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert explicit_route != "finish"
    assert explicit_event["safely_interpretable"] is True
    assert missing_event["safely_interpretable"] is False
    assert null_event["safely_interpretable"] is False


def test_unknown_recommendation_cannot_normalize_to_accept():
    result, decision, event = _verdict(_complete(passed=True, recommendation="approve"))
    assert result.passed is False
    assert result.recommendation == VerificationRecommendation.FALLBACK
    assert decision == "fallback"
    assert event["parsed_passed"] is True
    assert event["parsed_recommendation"] == "approve"
    assert event["recommendation_after_normalize"] == "fallback"
    assert event["safely_interpretable"] is False


def test_valid_accept_and_valid_false_keep_their_routes():
    accepted, accept_route, accept_event = _verdict(_complete(title="VerificationResult"))
    assert accepted.passed is True
    assert accepted.recommendation == VerificationRecommendation.ACCEPT
    assert accepted.score == 1.0
    assert accept_route == "finish"
    assert accept_event["schema_valid"] is False
    assert accept_event["safely_interpretable"] is True

    rejected, reject_route, _event = _verdict(
        _complete(passed=False, score=0.0, recommendation="fallback")
    )
    assert rejected.passed is False
    assert rejected.recommendation == VerificationRecommendation.FALLBACK
    assert reject_route == "fallback"

    retrieve, retrieve_route, _retrieve_event = _verdict(
        _complete(
            passed=False,
            score=0.2,
            recommendation="retrieve_more",
            missing_evidence=[
                {
                    "type": "learning_outcome",
                    "topic": "C4U09",
                    "query": "C4U09 learning objectives",
                }
            ],
        )
    )
    assert retrieve.passed is False
    assert retrieve.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert retrieve_route == "retrieve_more"


def test_explicit_zero_score_stays_zero_and_invalid_scores_do_not_accept():
    zero, _, zero_event = _verdict(_complete(score=0))
    assert zero.passed is True
    assert zero.score == 0.0
    assert zero.recommendation == VerificationRecommendation.ACCEPT
    assert zero_event["fields"]["score"] == "present"
    assert zero_event["parsed_score"] == 0

    omitted = {key: value for key, value in _complete().items() if key != "score"}
    missing_score, _, missing_event = _verdict(omitted)
    assert missing_score.recommendation == VerificationRecommendation.FALLBACK
    assert missing_event["fields"]["score"] == "missing"
    assert missing_event["parsed_score"] is None

    for value in (True, False, "1", float("nan"), float("inf"), float("-inf"), 1.5, -0.1):
        result, decision, event = _verdict(_complete(score=value))
        assert result.passed is False
        assert result.recommendation == VerificationRecommendation.FALLBACK
        assert decision == "fallback"
        assert event["parsed_score"] is value or (
            isinstance(value, float) and math.isnan(value) and math.isnan(event["parsed_score"])
        )
        assert event["safely_interpretable"] is False
        assert event["fields"]["score"] == "present"


def test_malformed_lists_do_not_become_empty_and_hide_a_failure():
    string_issues, _, string_event = _verdict(
        _complete(issues="the answer invents a unit")
    )
    assert string_issues.recommendation == VerificationRecommendation.FALLBACK
    assert string_issues.passed is False
    assert string_issues.issues == []
    assert string_event["fields"]["issues"] == "present"
    assert string_event["parsed_issues"] == "the answer invents a unit"
    assert "issues_not_array" in string_event["interpretation_problems"]

    null_missing, _, null_event = _verdict(_complete(missing_evidence=None))
    assert null_missing.recommendation == VerificationRecommendation.FALLBACK
    assert null_event["fields"]["missing_evidence"] == "null"
    assert null_event["parsed_missing_evidence"] is None

    numbers, _, number_event = _verdict(_complete(unsupported_claims=[1, 2]))
    assert numbers.recommendation == VerificationRecommendation.FALLBACK
    assert numbers.unsupported_claims == []
    assert number_event["parsed_unsupported_claims"] == [1, 2]
    assert "unsupported_claims_item_not_string" in number_event["interpretation_problems"]

    valid_empty, _, empty_event = _verdict(_complete(issues=[]))
    assert valid_empty.recommendation == VerificationRecommendation.ACCEPT
    assert empty_event["fields"]["issues"] == "empty_array"


def test_fallback_with_a_string_passed_cannot_become_accept():
    result, decision, event = _verdict(
        _complete(passed="yes", score=0, recommendation="fallback")
    )
    assert result.passed is False
    assert result.recommendation == VerificationRecommendation.FALLBACK
    assert decision == "fallback"
    assert event["parsed_passed"] == "yes"
    assert event["parsed_recommendation"] == "fallback"
    assert event["recommendation_after_normalize"] == "fallback"
    assert event["safely_interpretable"] is False


# Field order, types, and the echoed title match the five recorded 12H replay
# responses. Claim text and notes are shortened.
_RECORDED_SHAPE = {
    "title": "VerificationResult",
    "passed": True,
    "score": 1.0,
    "issues": [],
    "unsupported_claims": [],
    "missing_evidence": [],
    "incorrect_claims": [],
    "recommendation": "accept",
    "claims": [
        {
            "claim": "Living and Non – living Things — C4-U01",
            "verdict": "supported",
            "evidence_ids": ["unit-1"],
            "notes": "Matches the record.",
        }
    ],
    "clarification": "",
    "notes": "All listed units match the evidence.",
}


def test_recorded_provider_shape_keeps_its_accept():
    result, decision, event = _verdict(dict(_RECORDED_SHAPE))
    assert result.passed is True
    assert result.score == 1.0
    assert result.recommendation == VerificationRecommendation.ACCEPT
    assert len(result.claims) == 1
    assert decision == "finish"
    assert event["schema_valid"] is False
    assert event["extra_fields"] == ["title"]
    assert event["safely_interpretable"] is True
    assert event["verifier_result_invalid"] is False
    assert event["recommendation_after_normalize"] == "accept"
    assert event["final_route"] == "finish"


def test_missing_and_null_required_lists_are_not_read_as_empty():
    for key in ("issues", "unsupported_claims", "incorrect_claims", "missing_evidence"):
        omitted = {name: value for name, value in _complete().items() if name != key}
        missing, missing_route, missing_event = _verdict(omitted)
        null, null_route, null_event = _verdict(_complete(**{key: None}))
        assert missing.recommendation == VerificationRecommendation.FALLBACK
        assert null.recommendation == VerificationRecommendation.FALLBACK
        assert missing_route == null_route == "fallback"
        assert missing_event["fields"][key] == "missing"
        assert null_event["fields"][key] == "null"
        assert f"missing:{key}" in missing_event["interpretation_problems"]
        assert f"{key}_not_array" in null_event["interpretation_problems"]
        for event in (missing_event, null_event):
            assert event["safely_interpretable"] is False
            assert event["verifier_result_invalid"] is True
            assert key in event["verifier_invalid_reason"]


def test_string_and_object_missing_evidence_both_block_accept():
    for request in (
        "C4U09 learning outcomes",
        {"type": "learning_outcome", "topic": "C4U09", "query": "C4U09 learning outcomes"},
    ):
        result, _decision, event = _verdict(_complete(missing_evidence=[request]))
        assert result.passed is False
        assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE
        assert len(result.missing_evidence) == 1
        kept = result.missing_evidence[0]
        if isinstance(request, str):
            assert kept == request
        else:
            assert isinstance(kept, MissingEvidenceRequest)
            assert kept.topic == "C4U09"
        assert event["safely_interpretable"] is True
        assert event["verifier_result_invalid"] is False


def test_malformed_missing_evidence_items_fall_back_and_are_reported():
    for items in ([5], [["C4U09"]], [{"topic": 9}], ["C4U09", None]):
        result, decision, event = _verdict(
            _complete(passed=False, score=0.2, recommendation="retrieve_more", missing_evidence=items)
        )
        assert result.passed is False
        assert result.recommendation == VerificationRecommendation.FALLBACK
        assert decision == "fallback"
        assert event["parsed_missing_evidence"] == items
        assert "missing_evidence_item_invalid" in event["interpretation_problems"]
        assert event["safely_interpretable"] is False
        assert event["verifier_result_invalid"] is True


def test_extra_keys_inside_items_are_schema_warnings_not_rejections():
    payload = _complete(
        passed=False,
        score=0.2,
        recommendation="retrieve_more",
        missing_evidence=[{"topic": "C4U09", "reason": "outcomes absent"}],
        claims=[{"claim": "C4-U01", "verdict": "supported", "source": "x"}],
    )
    result, _decision, event = _verdict(payload)
    assert result.recommendation == VerificationRecommendation.RETRIEVE_MORE
    assert result.missing_evidence[0].topic == "C4U09"
    assert event["schema_valid"] is False
    assert event["safely_interpretable"] is True
    assert event["extra_fields"] == ["missing_evidence.reason", "claims.source"]


def test_parser_and_diagnostics_agree_on_malformed_optional_fields():
    for payload, problem in (
        (_complete(clarification=5), "clarification_not_string"),
        (_complete(notes=["x"]), "notes_not_string"),
        (_complete(claims=["C4-U01"]), "claims_item_invalid"),
        (_complete(claims=[{"claim": 5, "verdict": "supported"}]), "claims_item_invalid"),
        (_complete(claims=[{"claim": "C4-U01", "evidence_ids": "unit-1"}]), "claims_item_invalid"),
        (_complete(claims="C4-U01"), "claims_not_array"),
    ):
        result, decision, event = _verdict(payload)
        assert result.recommendation == VerificationRecommendation.FALLBACK
        assert decision == "fallback"
        assert problem in event["interpretation_problems"]
        assert event["safely_interpretable"] is False
        assert event["verifier_result_invalid"] is True


def test_oversized_integer_score_falls_back_without_raising():
    payload = json.loads(
        '{"passed": true, "score": 1' + "0" * 400 + ', "issues": [], '
        '"unsupported_claims": [], "incorrect_claims": [], '
        '"missing_evidence": [], "recommendation": "accept"}'
    )
    result, decision, event = _verdict(payload)
    assert result.recommendation == VerificationRecommendation.FALLBACK
    assert decision == "fallback"
    assert "score_out_of_range" in event["interpretation_problems"]
    untraced = AnswerVerifier(_ScriptedLLM(payload), settings=_settings()).verify(_state())
    assert untraced.recommendation == VerificationRecommendation.FALLBACK


def test_diagnostic_failure_does_not_change_the_verdict_or_route(monkeypatch):
    import app.agent.verifier as verifier_module

    def broken(_raw):
        raise RuntimeError("diagnostics broke")

    monkeypatch.setattr(verifier_module, "classify_schema", broken)
    accepted, accept_route, _event = _verdict(_complete())
    assert accepted.recommendation == VerificationRecommendation.ACCEPT
    assert accept_route == "finish"
    invalid, invalid_route, _invalid_event = _verdict(_complete(passed="yes"))
    assert invalid.recommendation == VerificationRecommendation.FALLBACK
    assert invalid_route == "fallback"


def _verdict_from(llm: LLMProvider, qa: CurriculumQAState | None = None):
    qa = qa or _state()
    with start_agent_run(question="science units", request_id="req-l", persist=False) as trace:
        result = AnswerVerifier(llm, settings=_settings()).verify(qa)
        qa.verification = result
        qa.pending_missing_evidence = list(result.missing_evidence)
        decision = route_after_verification({"qa": qa}, settings=_settings())
    return result, decision, trace.latest_event("agent.verification.payload")


def test_unreadable_response_cannot_finish_on_deterministic_checks_alone():
    for error, raw_text in (
        (LLMProviderError("Empty structured response from LLM"), None),
        (LLMProviderError("LLM returned invalid JSON"), "not json"),
    ):
        llm = _ScriptedLLM(error=error, raw_text=raw_text, json_extracted=False, record_raw=True)
        result, decision, event = _verdict_from(llm)
        assert result.passed is False
        assert result.recommendation == VerificationRecommendation.FALLBACK
        assert result.metadata["source"] == "deterministic"
        assert result.metadata["verifier_result_invalid"] is True
        assert result.metadata["verifier_invalid_reason"] == "unreadable_response"
        assert decision == "fallback"
        assert event["safely_interpretable"] is False
        assert event["verifier_result_invalid"] is True
        assert event["final_route"] == "fallback"
        assert event["fallback_reason"] == "verification_fallback"


def test_invalid_flag_survives_a_deterministic_hard_failure_merge():
    qa = _state()
    qa.grade = "CLASS_5"
    result, decision, event = _verdict_from(_ScriptedLLM(_complete(passed="yes")), qa)
    assert result.metadata["source"] == "merged"
    assert result.metadata["verifier_result_invalid"] is True
    assert result.recommendation == VerificationRecommendation.FALLBACK
    assert decision == "fallback"
    assert event["verifier_result_invalid"] is True


def test_route_falls_back_for_an_invalid_result_even_if_marked_passed():
    qa = _state()
    qa.verification = VerificationResult(
        passed=True,
        score=1.0,
        recommendation=VerificationRecommendation.ACCEPT,
        metadata={"verifier_result_invalid": True, "verifier_invalid_reason": "uninterpretable:passed"},
    )
    assert route_after_verification({"qa": qa}, settings=_settings()) == "fallback"
    for recommendation in (
        VerificationRecommendation.RETRIEVE_MORE,
        VerificationRecommendation.CLARIFY,
    ):
        qa.verification = VerificationResult(
            passed=False,
            recommendation=recommendation,
            missing_evidence=[MissingEvidenceRequest(type="learning_outcome", topic="C4U09")],
            metadata={"verifier_result_invalid": True},
        )
        assert route_after_verification({"qa": qa}, settings=_settings()) == "fallback"


def test_diagnostics_parser_normalization_and_route_agree():
    without_issues = {key: value for key, value in _complete().items() if key != "issues"}
    cases = [
        ("valid accept", _complete(), True, True, False, "accept", "finish"),
        ("valid fallback", _complete(passed=False, score=0.0, recommendation="fallback"), True, True, False, "fallback", "fallback"),
        ("missing list", without_issues, False, False, True, "fallback", "fallback"),
        ("malformed field", _complete(score="high"), False, False, True, "fallback", "fallback"),
        ("extra top-level", _complete(title="VerificationResult"), False, True, False, "accept", "finish"),
        (
            "extra nested",
            _complete(claims=[{"claim": "C4-U01", "verdict": "supported", "source": "x"}]),
            False,
            True,
            False,
            "accept",
            "finish",
        ),
    ]
    for label, payload, schema_valid, interpretable, invalid, after, route in cases:
        result, decision, event = _verdict(payload)
        assert event["schema_valid"] is schema_valid, label
        assert event["safely_interpretable"] is interpretable, label
        assert event["verifier_result_invalid"] is invalid, label
        assert bool(event["verifier_invalid_reason"]) is invalid, label
        assert event["recommendation_after_normalize"] == after, label
        assert result.recommendation.value == after, label
        assert decision == event["final_route"] == route, label


def test_trace_write_failure_keeps_the_verdict_and_route(monkeypatch):
    import app.agent.verifier as verifier_module

    def broken(self, *args):
        raise RuntimeError("trace write failed")

    for name in (
        "_record_verifier_input",
        "_record_parsed_payload",
        "_record_parser_outcome",
        "_record_normalization",
    ):
        monkeypatch.setattr(verifier_module.AnswerVerifier, name, broken)
    accepted, accept_route, _event = _verdict(_complete())
    assert accepted.recommendation == VerificationRecommendation.ACCEPT
    assert accept_route == "finish"
    invalid, invalid_route, _invalid_event = _verdict(_complete(issues=None))
    assert invalid.recommendation == VerificationRecommendation.FALLBACK
    assert invalid_route == "fallback"


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
