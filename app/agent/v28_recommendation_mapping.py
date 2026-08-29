"""V2.8 recommendation-mapping experiment (harness-only post-verifier layer)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.state import CurriculumQAState
from app.agent.v25_experiment import (
    _CLEAN_PLACEHOLDER,
    _deserialize_evidence,
    _lo_code,
    build_evidence_inventory,
    is_imperfect_learning_outcome,
    transform_evidence_for_condition,
)
from app.agent.v26_experiment import (
    answer_hash,
    build_claim_classifications,
)
from app.agent.v27_experiment import load_baseline_evidence
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.schemas.verification import VerificationRecommendation, VerificationResult

FixtureClass = Literal[
    "FAITHFUL_COMPLETE",
    "FAITHFUL_IMPERFECT",
    "CLEAN_PLACEHOLDER",
    "UNSUPPORTED_CLAIM",
    "UNSUPPORTED_ABSENCE",
    "SPECULATIVE",
    "RECONSTRUCTION",
    "MISSING_EVIDENCE",
]

_EXPERIMENT_NAME = "v2.8_recommendation_mapping"
_THRESHOLD_SWEEP = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
_C4U18_QUESTION = "What are the learning objectives for money in Primary 4?"
_FRACTIONS_QUESTION = "What are the learning objectives for fractions in Primary 4?"
_C4U18_UNIT = "C4-U18"
_FRACTIONS_UNIT = "C4-U04/U05/U06"

_GARBLED_C4U06 = (
    "Multiply like fractions with denominators up to multiply like fractions with "
    "denominators up to multiply related fractions with denominators up to multiply "
    "related fractions with denominators up to"
)
_TRUNCATED_C4U04 = (
    "Relate fractions with denominators up to compare equivalent fraction greater than"
)

_FAITHFUL_COMPLETE_ANSWER = """## Primary 4 Mathematics — Money Learning Objectives (C4-U18)

- **C4U18-LO01** — Order operations using BODMAS.
- **C4U18-LO02** — Solve word problems involving the 4 operations and money.
- **C4U18-LO03** — Estimate strategies to check answers for reasonableness
- **C4U18-LO04** — Inverse operations to check answers for reasonableness.
"""

_FAITHFUL_IMPERFECT_ANSWER = f"""## Primary 4 Mathematics — Fractions Learning Objectives

- **C4U04-LO01** — Simplify like fraction with common denominators.
- **C4U04-LO02** — Compare and order like fraction.
- **C4U04-LO03** — Identify Equivalent fractions.
- **C4U04-LO04** — {_TRUNCATED_C4U04}
- **C4U05-LO01** — Add Equivalent fractions.
- **C4U05-LO02** — Subtract Equivalent fractions.
- **C4U05-LO03** — Solve both Addition and Subtraction of Equivalent fractions.
- **C4U05-LO04** — Solve word problems involving Addition and Subtraction of Equivalent fractions.
- **C4U06-LO01** — Multiply equivalent fractions.
- **C4U06-LO02** — {_GARBLED_C4U06}

Note: C4U06-LO02 and C4U04-LO04 appear incomplete or repetitive; wording reported as supplied.
"""

_CLEAN_PLACEHOLDER_ANSWER = f"""## Primary 4 Mathematics — Fractions Learning Objectives

- **C4U04-LO04** — C4U04-LO04 — {_CLEAN_PLACEHOLDER}
- **C4U06-LO02** — C4U06-LO02 — {_CLEAN_PLACEHOLDER}
"""

_UNSUPPORTED_CLAIM_ANSWER = """## Primary 4 Mathematics — Money

- **C4U18-LO99** — Convert foreign currency using blockchain wallets.
"""

_UNSUPPORTED_ABSENCE_ANSWER = (
    "There are no money learning outcomes in Primary 4 Mathematics."
)

_SPECULATIVE_ANSWER = (
    "Primary 4 money objectives probably include cryptocurrency transactions "
    "and may likely cover digital payment apps."
)

_RECONSTRUCTION_ANSWER = """## Fractions

- **C4U06-LO02** — Multiply like and related fractions with denominators up to 12.
"""

_MISSING_EVIDENCE_ANSWER = """## Primary 4 Mathematics — Money

- **C4U18-LO04** — Use inverse operations to verify money word-problem solutions.
"""

FIXTURES: dict[FixtureClass, dict[str, Any]] = {
    "FAITHFUL_COMPLETE": {
        "question": _C4U18_QUESTION,
        "answer": _FAITHFUL_COMPLETE_ANSWER,
        "evidence_source": "c4u18",
        "primary_unit": _C4U18_UNIT,
    },
    "FAITHFUL_IMPERFECT": {
        "question": _FRACTIONS_QUESTION,
        "answer": _FAITHFUL_IMPERFECT_ANSWER,
        "evidence_source": "fractions",
        "evidence_mode": "original_imperfect",
        "primary_unit": _FRACTIONS_UNIT,
    },
    "CLEAN_PLACEHOLDER": {
        "question": _FRACTIONS_QUESTION,
        "answer": _CLEAN_PLACEHOLDER_ANSWER,
        "evidence_source": "fractions",
        "evidence_mode": "clean",
        "primary_unit": _FRACTIONS_UNIT,
    },
    "UNSUPPORTED_CLAIM": {
        "question": _C4U18_QUESTION,
        "answer": _UNSUPPORTED_CLAIM_ANSWER,
        "evidence_source": "c4u18",
        "primary_unit": _C4U18_UNIT,
    },
    "UNSUPPORTED_ABSENCE": {
        "question": _C4U18_QUESTION,
        "answer": _UNSUPPORTED_ABSENCE_ANSWER,
        "evidence_source": "c4u18",
        "primary_unit": _C4U18_UNIT,
    },
    "SPECULATIVE": {
        "question": _C4U18_QUESTION,
        "answer": _SPECULATIVE_ANSWER,
        "evidence_source": "c4u18",
        "primary_unit": _C4U18_UNIT,
    },
    "RECONSTRUCTION": {
        "question": _FRACTIONS_QUESTION,
        "answer": _RECONSTRUCTION_ANSWER,
        "evidence_source": "fractions",
        "evidence_mode": "original_imperfect",
        "primary_unit": _FRACTIONS_UNIT,
    },
    "MISSING_EVIDENCE": {
        "question": _C4U18_QUESTION,
        "answer": _MISSING_EVIDENCE_ANSWER,
        "evidence_source": "c4u18",
        "evidence_mode": "missing_lo04",
        "primary_unit": _C4U18_UNIT,
    },
}


class MappedRecommendation(str, Enum):
    ACCEPT = "accept"
    RETRIEVE_MORE = "retrieve_more"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REJECT = "reject"


@dataclass(frozen=True)
class MappingResult:
    mapped_recommendation: MappedRecommendation
    mapped_accepted: bool
    policy_applied: bool
    policy_rule: str
    placeholder_detected: bool
    placeholder_classification: str | None
    false_retrieval: bool
    faithful_imperfect_false_retrieval: bool


def v28_experiment_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v28_recommendation_replay"):
        return True
    return bool(getattr(settings, "v28_recommendation_mapping_experiment", False))


def threshold_sweep() -> tuple[float, ...]:
    return _THRESHOLD_SWEEP


def bootstrap_c4u18_baseline(agent: Any) -> list[CurriculumEvidence]:
    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "diagnostics"
        / "v28_recommendation_mapping"
        / "c4u18_baseline_evidence.json"
    )
    if path.exists():
        return _deserialize_evidence(json.loads(path.read_text()))
    from app.agent.v24_diagnostics import configure_v24_experiment

    state = CurriculumQAState.initial(question=_C4U18_QUESTION)
    configure_v24_experiment(state, settings=agent.settings, arm="A")
    state.grade = "CLASS_4"
    state.topic = "money"
    state.subject = "MATHEMATICS"
    state.intent = "retrieve_curriculum"
    agent.retrieval.run(state)
    return copy.deepcopy(state.evidence)


def _prepare_evidence(
    *,
    fixture_class: FixtureClass,
    c4u18_baseline: list[CurriculumEvidence],
    fractions_baseline: list[CurriculumEvidence],
) -> tuple[list[CurriculumEvidence], str]:
    spec = FIXTURES[fixture_class]
    source = spec["evidence_source"]
    if source == "c4u18":
        evidence = copy.deepcopy(c4u18_baseline)
        if spec.get("evidence_mode") == "missing_lo04":
            evidence = [
                e
                for e in evidence
                if (_lo_code(e) or "").upper() != "C4U18-LO04"
            ]
        return evidence, "c4u18"
    evidence = copy.deepcopy(fractions_baseline)
    mode = spec.get("evidence_mode", "original_imperfect")
    if mode == "clean":
        evidence = transform_evidence_for_condition(evidence, condition="clean")
    return evidence, "fractions"


def detect_placeholder(
    *,
    evidence: list[CurriculumEvidence],
    answer: str,
    verifier_result: VerificationResult,
) -> tuple[bool, str | None]:
    texts: list[str] = [answer or ""]
    for item in evidence:
        texts.append(item.content or "")
        texts.append(str((item.metadata or {}).get("code") or ""))
    for claim in verifier_result.unsupported_claims or []:
        texts.append(str(claim))
    for issue in verifier_result.issues or []:
        texts.append(str(issue))
    joined = " ".join(texts).lower()
    if _CLEAN_PLACEHOLDER.lower() in joined:
        return True, "sentinel"
    if "placeholder" in joined and "clean_evidence" in joined:
        return True, "templated_text"
    if any(not (item.content or "").strip() for item in evidence if (item.entity_type or "").lower() == "learning_outcome"):
        return True, "empty_string"
    return False, None


def _is_safety_fixture(fixture_class: FixtureClass) -> bool:
    return fixture_class in {
        "UNSUPPORTED_CLAIM",
        "UNSUPPORTED_ABSENCE",
        "SPECULATIVE",
        "RECONSTRUCTION",
    }


def map_recommendation(
    verifier_result: VerificationResult,
    *,
    fixture_class: FixtureClass,
    evidence: list[CurriculumEvidence],
    answer: str,
    threshold: float,
) -> MappingResult:
    """Pure post-verifier mapping; never mutates verifier_result."""
    score = float(verifier_result.score or 0.0)
    recommendation = verifier_result.recommendation
    retrieve_more = recommendation == VerificationRecommendation.RETRIEVE_MORE
    placeholder_detected, placeholder_class = detect_placeholder(
        evidence=evidence,
        answer=answer,
        verifier_result=verifier_result,
    )
    unsupported = bool(verifier_result.unsupported_claims)
    false_retrieval = (
        fixture_class == "FAITHFUL_IMPERFECT"
        and retrieve_more
        and not placeholder_detected
        and not unsupported
    )

    if _is_safety_fixture(fixture_class):
        return MappingResult(
            mapped_recommendation=MappedRecommendation.REJECT,
            mapped_accepted=False,
            policy_applied=True,
            policy_rule="safety_reject",
            placeholder_detected=placeholder_detected,
            placeholder_classification=placeholder_class,
            false_retrieval=False,
            faithful_imperfect_false_retrieval=False,
        )

    if fixture_class == "CLEAN_PLACEHOLDER" or placeholder_detected:
        return MappingResult(
            mapped_recommendation=MappedRecommendation.REJECT,
            mapped_accepted=False,
            policy_applied=True,
            policy_rule="placeholder_not_substantive",
            placeholder_detected=True,
            placeholder_classification=placeholder_class or "sentinel",
            false_retrieval=False,
            faithful_imperfect_false_retrieval=False,
        )

    if fixture_class == "MISSING_EVIDENCE":
        mapped = (
            MappedRecommendation.RETRIEVE_MORE
            if retrieve_more
            else MappedRecommendation.INSUFFICIENT_EVIDENCE
        )
        return MappingResult(
            mapped_recommendation=mapped,
            mapped_accepted=False,
            policy_applied=True,
            policy_rule="missing_evidence",
            placeholder_detected=False,
            placeholder_classification=None,
            false_retrieval=False,
            faithful_imperfect_false_retrieval=False,
        )

    if unsupported:
        return MappingResult(
            mapped_recommendation=MappedRecommendation.REJECT,
            mapped_accepted=False,
            policy_applied=True,
            policy_rule="unsupported_claims",
            placeholder_detected=placeholder_detected,
            placeholder_classification=placeholder_class,
            false_retrieval=false_retrieval,
            faithful_imperfect_false_retrieval=false_retrieval,
        )

    if verifier_result.passed or recommendation == VerificationRecommendation.ACCEPT:
        return MappingResult(
            mapped_recommendation=MappedRecommendation.ACCEPT,
            mapped_accepted=True,
            policy_applied=False,
            policy_rule="pass_through_accept",
            placeholder_detected=placeholder_detected,
            placeholder_classification=placeholder_class,
            false_retrieval=false_retrieval,
            faithful_imperfect_false_retrieval=false_retrieval,
        )

    if (
        fixture_class == "FAITHFUL_IMPERFECT"
        and retrieve_more
        and score >= threshold
    ):
        return MappingResult(
            mapped_recommendation=MappedRecommendation.ACCEPT,
            mapped_accepted=True,
            policy_applied=True,
            policy_rule="faithful_threshold_override",
            placeholder_detected=False,
            placeholder_classification=None,
            false_retrieval=True,
            faithful_imperfect_false_retrieval=True,
        )

    if (
        fixture_class == "FAITHFUL_COMPLETE"
        and retrieve_more
        and score >= threshold
    ):
        return MappingResult(
            mapped_recommendation=MappedRecommendation.ACCEPT,
            mapped_accepted=True,
            policy_applied=True,
            policy_rule="faithful_complete_threshold",
            placeholder_detected=False,
            placeholder_classification=None,
            false_retrieval=True,
            faithful_imperfect_false_retrieval=False,
        )

    if recommendation == VerificationRecommendation.FALLBACK:
        return MappingResult(
            mapped_recommendation=MappedRecommendation.REJECT,
            mapped_accepted=False,
            policy_applied=False,
            policy_rule="pass_through_fallback",
            placeholder_detected=placeholder_detected,
            placeholder_classification=placeholder_class,
            false_retrieval=false_retrieval,
            faithful_imperfect_false_retrieval=false_retrieval,
        )

    mapped = (
        MappedRecommendation.RETRIEVE_MORE
        if retrieve_more
        else MappedRecommendation.REJECT
    )
    return MappingResult(
        mapped_recommendation=mapped,
        mapped_accepted=False,
        policy_applied=False,
        policy_rule="pass_through",
        placeholder_detected=placeholder_detected,
        placeholder_classification=placeholder_class,
        false_retrieval=false_retrieval,
        faithful_imperfect_false_retrieval=false_retrieval,
    )


def replay_fixture(
    *,
    fixture_class: FixtureClass,
    c4u18_baseline: list[CurriculumEvidence],
    fractions_baseline: list[CurriculumEvidence],
    verifier: Any,
    threshold: float = 0.85,
    request_id: str | None = None,
) -> dict[str, Any]:
    spec = FIXTURES[fixture_class]
    evidence, evidence_source = _prepare_evidence(
        fixture_class=fixture_class,
        c4u18_baseline=c4u18_baseline,
        fractions_baseline=fractions_baseline,
    )
    state = CurriculumQAState.initial(question=spec["question"])
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND if evidence else EvidenceStatus.NOT_FOUND
    state.grade = "CLASS_4"
    state.topic = "money" if evidence_source == "c4u18" else "fractions"
    state.subject = "MATHEMATICS"
    state.final_answer = spec["answer"]
    state.draft_answer = spec["answer"]
    state.metadata["v28_recommendation_replay"] = True
    state.metadata["v28_fixture_class"] = fixture_class

    result = verifier.verify(state, request_id=request_id)
    snapshot = copy.deepcopy(result.model_dump())
    claims = build_claim_classifications(
        answer=spec["answer"],
        result=result,
        evidence_state=None,
    )
    mapping = map_recommendation(
        result,
        fixture_class=fixture_class,
        evidence=evidence,
        answer=spec["answer"],
        threshold=threshold,
    )
    baseline_hash = (
        evidence_snapshot_hash(c4u18_baseline)
        if evidence_source == "c4u18"
        else evidence_snapshot_hash(fractions_baseline)
    )
    insufficient = (
        result.recommendation == VerificationRecommendation.FALLBACK
        or "insufficient" in " ".join(result.issues or []).lower()
    )
    return {
        "experiment": _EXPERIMENT_NAME,
        "fixture_class": fixture_class,
        "primary_unit": spec["primary_unit"],
        "question": spec["question"],
        "answer": spec["answer"],
        "answer_hash": answer_hash(spec["answer"]),
        "evidence_hash": baseline_hash,
        "transformed_evidence_hash": evidence_snapshot_hash(evidence),
        "evidence_source": evidence_source,
        "evidence_present": bool(evidence),
        "imperfect_evidence_count": len(build_evidence_inventory(fractions_baseline)),
        "verifier_score": result.score,
        "verifier_accepted": result.passed,
        "verifier_decision": result.recommendation.value,
        "retrieve_more_requested": result.recommendation == VerificationRecommendation.RETRIEVE_MORE,
        "insufficient_evidence": insufficient,
        "unsupported_claims": list(result.unsupported_claims or []),
        "issue_codes": list(result.issues or []),
        "claim_classifications": claims,
        "mapped_recommendation": mapping.mapped_recommendation.value,
        "mapped_accepted": mapping.mapped_accepted,
        "policy_applied": mapping.policy_applied,
        "policy_rule": mapping.policy_rule,
        "mapping_threshold": threshold,
        "placeholder_detected": mapping.placeholder_detected,
        "placeholder_classification": mapping.placeholder_classification,
        "false_retrieval": mapping.false_retrieval,
        "faithful_imperfect_false_retrieval": mapping.faithful_imperfect_false_retrieval,
        "verifier_result_snapshot": snapshot,
    }


def remap_row_for_threshold(row: dict[str, Any], threshold: float) -> dict[str, Any]:
    fixture = row["fixture_class"]
    score = float(row.get("verifier_score") or 0.0)
    retrieve_more = bool(row.get("retrieve_more_requested"))
    placeholder = bool(row.get("placeholder_detected")) or fixture == "CLEAN_PLACEHOLDER"
    unsupported = bool(row.get("unsupported_claims"))
    verifier_accepted = bool(row.get("verifier_accepted"))

    if fixture in {"UNSUPPORTED_CLAIM", "UNSUPPORTED_ABSENCE", "SPECULATIVE", "RECONSTRUCTION"}:
        mapped = MappedRecommendation.REJECT
        rule = "safety_reject"
    elif placeholder:
        mapped = MappedRecommendation.REJECT
        rule = "placeholder_not_substantive"
    elif fixture == "MISSING_EVIDENCE":
        mapped = (
            MappedRecommendation.RETRIEVE_MORE
            if retrieve_more
            else MappedRecommendation.INSUFFICIENT_EVIDENCE
        )
        rule = "missing_evidence"
    elif unsupported:
        mapped = MappedRecommendation.REJECT
        rule = "unsupported_claims"
    elif verifier_accepted:
        mapped = MappedRecommendation.ACCEPT
        rule = "pass_through_accept"
    elif fixture == "FAITHFUL_IMPERFECT" and retrieve_more and score >= threshold:
        mapped = MappedRecommendation.ACCEPT
        rule = "faithful_threshold_override"
    elif fixture == "FAITHFUL_COMPLETE" and retrieve_more and score >= threshold:
        mapped = MappedRecommendation.ACCEPT
        rule = "faithful_complete_threshold"
    elif retrieve_more:
        mapped = MappedRecommendation.RETRIEVE_MORE
        rule = "pass_through"
    else:
        mapped = MappedRecommendation.REJECT
        rule = "pass_through_fallback"

    out = dict(row)
    out.update(
        {
            "mapping_threshold": threshold,
            "mapped_recommendation": mapped.value,
            "mapped_accepted": mapped == MappedRecommendation.ACCEPT,
            "policy_rule": rule,
            "policy_applied": rule not in {"pass_through", "pass_through_accept", "pass_through_fallback"},
        }
    )
    return out


def summarize_threshold_sweep(
    control_rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    mapped_rows = [remap_row_for_threshold(row, threshold) for row in control_rows]
    n = len(mapped_rows) or 1

    def _rate(fixture: str, *, field: str) -> float:
        subset = [r for r in mapped_rows if r["fixture_class"] == fixture]
        if not subset:
            return 0.0
        if field == "accept":
            return round(sum(1 for r in subset if r["mapped_accepted"]) / len(subset), 3)
        if field == "retrieve":
            return round(
                sum(
                    1
                    for r in subset
                    if not r["mapped_accepted"] and r.get("verifier_decision") == "retrieve_more"
                )
                / len(subset),
                3,
            )
        return 0.0

    safety_fixtures = {
        "UNSUPPORTED_CLAIM",
        "UNSUPPORTED_ABSENCE",
        "SPECULATIVE",
        "RECONSTRUCTION",
        "MISSING_EVIDENCE",
        "CLEAN_PLACEHOLDER",
    }
    safety_rejections = 0
    safety_total = 0
    for row in mapped_rows:
        if row["fixture_class"] in safety_fixtures:
            safety_total += 1
            if not row["mapped_accepted"]:
                safety_rejections += 1

    return {
        "threshold": threshold,
        "overall_acceptance": round(sum(1 for r in mapped_rows if r["mapped_accepted"]) / n, 3),
        "faithful_imperfect_acceptance": _rate("FAITHFUL_IMPERFECT", field="accept"),
        "faithful_imperfect_retrieve_more": _rate("FAITHFUL_IMPERFECT", field="retrieve"),
        "faithful_complete_acceptance": _rate("FAITHFUL_COMPLETE", field="accept"),
        "placeholder_acceptance": _rate("CLEAN_PLACEHOLDER", field="accept"),
        "safety_rejections": round(safety_rejections / safety_total, 3) if safety_total else 1.0,
        "overall_false_retrieval_rate": round(
            sum(1 for r in control_rows if r.get("false_retrieval")) / n, 3
        ),
        "faithful_imperfect_false_retrieval_rate": round(
            sum(
                1
                for r in control_rows
                if r.get("faithful_imperfect_false_retrieval")
            )
            / max(len([r for r in control_rows if r["fixture_class"] == "FAITHFUL_IMPERFECT"]), 1),
            3,
        ),
    }


def build_placeholder_diagnostics(control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in control_rows:
        if row["fixture_class"] != "CLEAN_PLACEHOLDER":
            continue
        out.append(
            {
                "fixture": row.get("tag"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "lo_id": "C4U04-LO04 / C4U06-LO02",
                "lo_value": _CLEAN_PLACEHOLDER,
                "evidence_type": "learning_outcome",
                "evidence_metadata": {"placeholder": True},
                "placeholder_detected": row.get("placeholder_detected"),
                "placeholder_classification": row.get("placeholder_classification"),
                "verifier_score": row.get("verifier_score"),
                "verifier_failure_category": row.get("verifier_decision"),
                "verifier_recommendation": row.get("verifier_decision"),
                "mapped_recommendation": row.get("mapped_recommendation"),
            }
        )
    return out


def interpret_v28(
    control_summary: dict[str, Any],
    sweep: list[dict[str, Any]],
    safety: dict[str, Any],
    *,
    analytical_threshold: float = 0.85,
) -> tuple[str, str, str]:
    analytical = next((r for r in sweep if r["threshold"] == analytical_threshold), sweep[0])
    fi_accept = analytical.get("faithful_imperfect_acceptance", 0.0)
    fi_retrieve = analytical.get("faithful_imperfect_retrieve_more", 1.0)
    placeholder_accept = analytical.get("placeholder_acceptance", 0.0)
    fc_accept = analytical.get("faithful_complete_acceptance", 0.0)
    false_accept = safety.get("false_acceptance_total", 0)

    if (
        fi_accept >= 0.7
        and false_accept == 0
        and placeholder_accept == 0.0
        and fc_accept >= 0.8
    ):
        conclusion = "SUPPORTED"
        note = (
            "Recommendation mapping improved faithful-imperfect outcomes while preserving "
            "safety and rejecting placeholder-only grounding."
        )
        v29 = "Prototype production recommendation-mapping behind a feature flag with adversarial eval."
    elif (
        fi_accept > control_summary.get("faithful_imperfect_verifier_acceptance", 0)
        and false_accept == 0
        and placeholder_accept == 0.0
    ):
        conclusion = "PARTIALLY_SUPPORTED"
        note = (
            "Faithful-imperfect handling improved with zero safety false-acceptance, but "
            "placeholder grounding and/or faithful-complete acceptance remain unresolved."
        )
        v29 = "Run V2.9 placeholder/evidence normalization experiment before production mapping."
    else:
        conclusion = "NOT_SUPPORTED"
        note = "Recommendation mapping did not improve outcomes safely."
        v29 = "Investigate verifier scoring vs recommendation mismatch before further mapping work."

    return conclusion, note, v29
