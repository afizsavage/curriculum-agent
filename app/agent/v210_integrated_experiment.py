"""V2.10 integrated grounding + recommendation safety experiment (harness-only)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.state import CurriculumQAState
from app.agent.v25_experiment import _CLEAN_PLACEHOLDER, _lo_code, build_evidence_inventory
from app.agent.v26_experiment import answer_hash, build_claim_classifications
from app.agent.v28_recommendation_mapping import (
    FIXTURES as V28_FIXTURES,
    FixtureClass as V28FixtureClass,
    _prepare_evidence,
    bootstrap_c4u18_baseline,
    map_recommendation,
    remap_row_for_threshold,
    threshold_sweep,
)
from app.agent.v29_evidence_normalization import (
    NormalizationVariant,
    _build_unit_lookup,
    _evidence_chain_for_fc,
    detect_placeholder_content,
    normalize_evidence,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.schemas.verification import VerificationRecommendation, VerificationResult

_EXPERIMENT_NAME = "v2.10_integrated_experiment"
_C4U18_HASH = "be3e342763f1faac"
_FRACTIONS_HASH = "977b259fcfb4b282"
_ANALYTICAL_THRESHOLD = 0.85

Attribution = Literal[
    "NO_CHANGE",
    "NORMALIZATION_FIXED_GROUNDING",
    "MAPPER_CHANGED_DECISION",
    "NORMALIZATION_PLUS_MAPPER",
    "SAFETY_BLOCK",
    "PLACEHOLDER_BLOCK",
    "OTHER",
]

IntegrationFixtureClass = Literal[
    "FAITHFUL_COMPLETE",
    "FAITHFUL_IMPERFECT",
    "CLEAN_PLACEHOLDER",
    "UNSUPPORTED_CLAIM",
    "UNSUPPORTED_ABSENCE",
    "SPECULATIVE",
    "RECONSTRUCTION",
    "MISSING_EVIDENCE",
    "HIGH_SCORE_UNSUPPORTED",
    "NORMALIZATION_ONLY_GROUNDING",
    "NORMALIZATION_MUST_NOT_INVENT",
    "PLACEHOLDER_PLUS_HIGH_SCORE",
]

AdversarialFixtureClass = Literal[
    "ADV_FAKE_PARENT",
    "ADV_CONFLICTING_PARENT",
    "ADV_PLACEHOLDER_PARENT",
    "ADV_WRONG_SUBJECT",
    "ADV_WRONG_GRADE",
    "ADV_HIGH_SCORE_SAFETY",
    "ADV_HIGH_SCORE_PLACEHOLDER",
    "ADV_MISSING_AFTER_NORM",
]

FixtureClass = IntegrationFixtureClass | AdversarialFixtureClass

_UNRESOLVABLE_TOPIC_UUID = "00000000-0000-0000-0000-000000000099"

_HIGH_SCORE_UNSUPPORTED_ANSWER = """## Primary 4 Mathematics — Money

- **C4U18-LO99** — Convert foreign currency using blockchain wallets.
"""

_NORMALIZATION_MUST_NOT_INVENT_ANSWER = """## Primary 4 Mathematics — Money

- **C4U18-LO01** — Order operations using BODMAS.
"""

_INTEGRATION_FIXTURES: dict[str, dict[str, Any]] = {
    **{k: dict(v) for k, v in V28_FIXTURES.items()},
    "HIGH_SCORE_UNSUPPORTED": {
        "question": V28_FIXTURES["UNSUPPORTED_CLAIM"]["question"],
        "answer": _HIGH_SCORE_UNSUPPORTED_ANSWER,
        "evidence_source": "c4u18",
        "primary_unit": "C4-U18",
        "mapper_fixture": "UNSUPPORTED_CLAIM",
        "expect_safety_block": True,
    },
    "NORMALIZATION_ONLY_GROUNDING": {
        **V28_FIXTURES["FAITHFUL_COMPLETE"],
        "mapper_fixture": "FAITHFUL_COMPLETE",
        "requires_normalization": True,
    },
    "NORMALIZATION_MUST_NOT_INVENT": {
        "question": V28_FIXTURES["FAITHFUL_COMPLETE"]["question"],
        "answer": _NORMALIZATION_MUST_NOT_INVENT_ANSWER,
        "evidence_source": "c4u18",
        "evidence_mode": "unresolvable_topic",
        "primary_unit": "C4-U18",
        "mapper_fixture": "FAITHFUL_COMPLETE",
        "expect_no_invention": True,
    },
    "PLACEHOLDER_PLUS_HIGH_SCORE": {
        **V28_FIXTURES["CLEAN_PLACEHOLDER"],
        "mapper_fixture": "CLEAN_PLACEHOLDER",
        "expect_placeholder_block": True,
    },
}

_ADVERSARIAL_FIXTURES: dict[str, dict[str, Any]] = {
    "ADV_FAKE_PARENT": {
        "question": V28_FIXTURES["FAITHFUL_COMPLETE"]["question"],
        "answer": _NORMALIZATION_MUST_NOT_INVENT_ANSWER,
        "evidence_source": "c4u18",
        "evidence_mode": "unresolvable_topic",
        "primary_unit": "C4-U18",
        "mapper_fixture": "FAITHFUL_COMPLETE",
        "adversarial": "fake_parent",
    },
    "ADV_CONFLICTING_PARENT": {
        "question": V28_FIXTURES["FAITHFUL_COMPLETE"]["question"],
        "answer": V28_FIXTURES["FAITHFUL_COMPLETE"]["answer"],
        "evidence_source": "c4u18",
        "evidence_mode": "conflicting_parent",
        "primary_unit": "C4-U18",
        "mapper_fixture": "FAITHFUL_COMPLETE",
        "adversarial": "conflicting_parent",
    },
    "ADV_PLACEHOLDER_PARENT": {
        "question": V28_FIXTURES["FAITHFUL_COMPLETE"]["question"],
        "answer": V28_FIXTURES["FAITHFUL_COMPLETE"]["answer"],
        "evidence_source": "c4u18",
        "evidence_mode": "placeholder_parent",
        "primary_unit": "C4-U18",
        "mapper_fixture": "FAITHFUL_COMPLETE",
        "adversarial": "placeholder_parent",
    },
    "ADV_WRONG_SUBJECT": {
        "question": V28_FIXTURES["FAITHFUL_COMPLETE"]["question"],
        "answer": V28_FIXTURES["FAITHFUL_COMPLETE"]["answer"],
        "evidence_source": "c4u18",
        "evidence_mode": "wrong_subject",
        "primary_unit": "C4-U18",
        "mapper_fixture": "FAITHFUL_COMPLETE",
        "adversarial": "wrong_subject",
    },
    "ADV_WRONG_GRADE": {
        "question": V28_FIXTURES["FAITHFUL_COMPLETE"]["question"],
        "answer": V28_FIXTURES["FAITHFUL_COMPLETE"]["answer"],
        "evidence_source": "c4u18",
        "evidence_mode": "wrong_grade",
        "primary_unit": "C4-U18",
        "mapper_fixture": "FAITHFUL_COMPLETE",
        "adversarial": "wrong_grade",
    },
    "ADV_HIGH_SCORE_SAFETY": {
        **_INTEGRATION_FIXTURES["HIGH_SCORE_UNSUPPORTED"],
        "adversarial": "high_score_safety",
    },
    "ADV_HIGH_SCORE_PLACEHOLDER": {
        **_INTEGRATION_FIXTURES["PLACEHOLDER_PLUS_HIGH_SCORE"],
        "adversarial": "high_score_placeholder",
    },
    "ADV_MISSING_AFTER_NORM": {
        **V28_FIXTURES["MISSING_EVIDENCE"],
        "mapper_fixture": "MISSING_EVIDENCE",
        "adversarial": "missing_after_norm",
    },
}

ALL_FIXTURES: dict[str, dict[str, Any]] = {**_INTEGRATION_FIXTURES, **_ADVERSARIAL_FIXTURES}

INTEGRATION_FIXTURE_CLASSES: tuple[str, ...] = tuple(_INTEGRATION_FIXTURES.keys())
ADVERSARIAL_FIXTURE_CLASSES: tuple[str, ...] = tuple(_ADVERSARIAL_FIXTURES.keys())


class Pipeline(str, Enum):
    A_RAW_VERIFIER = "A_RAW_VERIFIER"
    B_NORMALIZED_VERIFIER = "B_NORMALIZED_VERIFIER"
    C_RAW_VERIFIER_MAPPER = "C_RAW_VERIFIER_MAPPER"
    D_NORMALIZED_VERIFIER_MAPPER = "D_NORMALIZED_VERIFIER_MAPPER"


PIPELINES: tuple[Pipeline, ...] = (
    Pipeline.A_RAW_VERIFIER,
    Pipeline.B_NORMALIZED_VERIFIER,
    Pipeline.C_RAW_VERIFIER_MAPPER,
    Pipeline.D_NORMALIZED_VERIFIER_MAPPER,
)


def v210_experiment_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v210_integrated_replay"):
        return True
    return bool(getattr(settings, "v210_integrated_experiment", False))


def get_threshold_sweep() -> tuple[float, ...]:
    return threshold_sweep()


def _mapper_fixture(fixture_class: str) -> V28FixtureClass:
    spec = ALL_FIXTURES[fixture_class]
    mapped = spec.get("mapper_fixture", fixture_class)
    if mapped in V28_FIXTURES:
        return mapped  # type: ignore[return-value]
    return fixture_class  # type: ignore[return-value]


def _prepare_integrated_evidence(
    *,
    fixture_class: str,
    c4u18_baseline: list[CurriculumEvidence],
    fractions_baseline: list[CurriculumEvidence],
) -> tuple[list[CurriculumEvidence], str]:
    if fixture_class in {
        "HIGH_SCORE_UNSUPPORTED",
        "ADV_HIGH_SCORE_SAFETY",
    }:
        base_fixture = "UNSUPPORTED_CLAIM"
    elif fixture_class in {"NORMALIZATION_ONLY_GROUNDING"}:
        base_fixture = "FAITHFUL_COMPLETE"
    elif fixture_class in {"NORMALIZATION_MUST_NOT_INVENT", "ADV_FAKE_PARENT"}:
        base_fixture = "FAITHFUL_COMPLETE"
    elif fixture_class in {"PLACEHOLDER_PLUS_HIGH_SCORE", "ADV_HIGH_SCORE_PLACEHOLDER"}:
        base_fixture = "CLEAN_PLACEHOLDER"
    elif fixture_class == "ADV_MISSING_AFTER_NORM":
        base_fixture = "MISSING_EVIDENCE"
    elif fixture_class.startswith("ADV_"):
        base_fixture = "FAITHFUL_COMPLETE"
    else:
        base_fixture = fixture_class

    evidence, source = _prepare_evidence(
        fixture_class=base_fixture,  # type: ignore[arg-type]
        c4u18_baseline=c4u18_baseline,
        fractions_baseline=fractions_baseline,
    )
    evidence = copy.deepcopy(evidence)
    mode = ALL_FIXTURES[fixture_class].get("evidence_mode")

    if mode == "unresolvable_topic":
        for item in evidence:
            if (_lo_code(item) or "").upper() == "C4U18-LO01":
                item.topic = _UNRESOLVABLE_TOPIC_UUID
                meta = dict(item.metadata or {})
                meta.pop("parent_content_name", None)
                meta.pop("parent_content_code", None)
                item.metadata = meta
    elif mode == "conflicting_parent":
        for item in evidence:
            if (_lo_code(item) or "").upper() == "C4U18-LO01":
                meta = dict(item.metadata or {})
                meta["parent_content_name"] = "Conflicting Unit A"
                item.metadata = meta
        unit = next((e for e in evidence if (e.entity_type or "").lower() == "unit"), None)
        if unit:
            conflict_unit = copy.deepcopy(unit)
            conflict_unit.entity_id = "conflict-unit-b"
            conflict_unit.name = "Conflicting Unit B"
            conflict_unit.content = "Conflicting Unit B"
            conflict_unit.metadata = {"code": "C4-U99"}
            evidence.append(conflict_unit)
    elif mode == "placeholder_parent":
        for item in evidence:
            if (item.entity_type or "").lower() == "unit":
                item.content = _CLEAN_PLACEHOLDER
                item.name = _CLEAN_PLACEHOLDER
    elif mode == "wrong_subject":
        for item in evidence:
            if (_lo_code(item) or "").upper() == "C4U18-LO01":
                item.subject = "ENGLISH"
    elif mode == "wrong_grade":
        for item in evidence:
            if (_lo_code(item) or "").upper() == "C4U18-LO01":
                item.grade = "CLASS_5"

    return evidence, source


def _apply_normalization(
    raw_evidence: list[CurriculumEvidence],
    *,
    pipeline: Pipeline,
) -> tuple[list[CurriculumEvidence], dict[str, Any] | None]:
    if pipeline in {Pipeline.A_RAW_VERIFIER, Pipeline.C_RAW_VERIFIER_MAPPER}:
        return copy.deepcopy(raw_evidence), None
    normalized = normalize_evidence(raw_evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION)
    return normalized.evidence, normalized.diagnostics.to_dict()


def _final_accepted(
    *,
    pipeline: Pipeline,
    verifier_result: VerificationResult,
    mapping: Any | None,
) -> bool:
    if pipeline in {Pipeline.C_RAW_VERIFIER_MAPPER, Pipeline.D_NORMALIZED_VERIFIER_MAPPER}:
        return bool(mapping and mapping.mapped_accepted)
    return bool(verifier_result.passed)


def _final_recommendation(
    *,
    pipeline: Pipeline,
    verifier_result: VerificationResult,
    mapping: Any | None,
) -> str:
    if pipeline in {Pipeline.C_RAW_VERIFIER_MAPPER, Pipeline.D_NORMALIZED_VERIFIER_MAPPER}:
        return mapping.mapped_recommendation.value if mapping else verifier_result.recommendation.value
    return verifier_result.recommendation.value


def compute_attribution(
    *,
    pipeline: Pipeline,
    fixture_class: str,
    raw_verifier_accepted: bool,
    normalized_verifier_accepted: bool,
    final_accepted: bool,
    mapping_applied: bool,
    unsupported: bool,
    placeholder: bool,
) -> Attribution:
    if placeholder and not final_accepted:
        return "PLACEHOLDER_BLOCK"
    if unsupported and not final_accepted:
        return "SAFETY_BLOCK"
    if pipeline == Pipeline.A_RAW_VERIFIER:
        return "NO_CHANGE"
    if pipeline == Pipeline.B_NORMALIZED_VERIFIER:
        if not raw_verifier_accepted and normalized_verifier_accepted:
            return "NORMALIZATION_FIXED_GROUNDING"
        return "NO_CHANGE" if raw_verifier_accepted == normalized_verifier_accepted else "OTHER"
    if pipeline == Pipeline.C_RAW_VERIFIER_MAPPER:
        if mapping_applied and final_accepted and not raw_verifier_accepted:
            return "MAPPER_CHANGED_DECISION"
        return "NO_CHANGE"
    if pipeline == Pipeline.D_NORMALIZED_VERIFIER_MAPPER:
        if mapping_applied and final_accepted:
            if not raw_verifier_accepted and normalized_verifier_accepted:
                return "NORMALIZATION_PLUS_MAPPER" if not mapping_applied else "MAPPER_CHANGED_DECISION"
            if not raw_verifier_accepted and not normalized_verifier_accepted and final_accepted:
                return "NORMALIZATION_PLUS_MAPPER"
            if not raw_verifier_accepted and normalized_verifier_accepted:
                return "NORMALIZATION_FIXED_GROUNDING"
            return "MAPPER_CHANGED_DECISION"
        if not raw_verifier_accepted and normalized_verifier_accepted:
            return "NORMALIZATION_FIXED_GROUNDING"
    return "OTHER"


@dataclass(frozen=True)
class IntegratedPipelineResult:
    pipeline: str
    fixture_class: str
    raw_verifier_accepted: bool
    verifier_accepted: bool
    final_accepted: bool
    final_recommendation: str
    attribution: str
    mapping_applied: bool


def run_pipeline(
    *,
    fixture_class: str,
    pipeline: Pipeline,
    c4u18_baseline: list[CurriculumEvidence],
    fractions_baseline: list[CurriculumEvidence],
    verifier: Any,
    threshold: float = _ANALYTICAL_THRESHOLD,
    request_id: str | None = None,
    raw_baseline_row: dict[str, Any] | None = None,
    normalized_baseline_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute normalize → verify → map (when applicable)."""
    spec = ALL_FIXTURES[fixture_class]
    raw_evidence, evidence_source = _prepare_integrated_evidence(
        fixture_class=fixture_class,
        c4u18_baseline=c4u18_baseline,
        fractions_baseline=fractions_baseline,
    )
    verify_evidence, norm_diag = _apply_normalization(raw_evidence, pipeline=pipeline)

    state = CurriculumQAState.initial(question=spec["question"])
    state.evidence = verify_evidence
    state.evidence_status = EvidenceStatus.FOUND if verify_evidence else EvidenceStatus.NOT_FOUND
    state.grade = "CLASS_4"
    state.topic = "money" if evidence_source == "c4u18" else "fractions"
    state.subject = "MATHEMATICS"
    state.final_answer = spec["answer"]
    state.draft_answer = spec["answer"]
    state.metadata["v210_integrated_replay"] = True
    state.metadata["v210_fixture_class"] = fixture_class
    state.metadata["v210_pipeline"] = pipeline.value

    verifier_result = verifier.verify(state, request_id=request_id)
    verifier_snapshot = copy.deepcopy(verifier_result.model_dump())

    mapping = None
    if pipeline in {Pipeline.C_RAW_VERIFIER_MAPPER, Pipeline.D_NORMALIZED_VERIFIER_MAPPER}:
        mapping = map_recommendation(
            verifier_result,
            fixture_class=_mapper_fixture(fixture_class),
            evidence=verify_evidence,
            answer=spec["answer"],
            threshold=threshold,
        )

    raw_verifier_accepted = bool((raw_baseline_row or {}).get("verifier_accepted", False))
    normalized_baseline_accepted = bool(
        (normalized_baseline_row or {}).get("verifier_accepted", verifier_result.passed)
    )

    placeholder_detected = any(
        detect_placeholder_content(item.content)[0] for item in raw_evidence
    )
    unsupported = bool(verifier_result.unsupported_claims)
    final_accepted = _final_accepted(
        pipeline=pipeline, verifier_result=verifier_result, mapping=mapping
    )
    attribution = compute_attribution(
        pipeline=pipeline,
        fixture_class=fixture_class,
        raw_verifier_accepted=raw_verifier_accepted,
        normalized_verifier_accepted=normalized_baseline_accepted,
        final_accepted=final_accepted,
        mapping_applied=bool(mapping and mapping.policy_applied),
        unsupported=unsupported,
        placeholder=placeholder_detected or fixture_class in {
            "CLEAN_PLACEHOLDER",
            "PLACEHOLDER_PLUS_HIGH_SCORE",
        },
    )

    claims = build_claim_classifications(
        answer=spec["answer"], result=verifier_result, evidence_state=None
    )
    baseline_hash = _C4U18_HASH if evidence_source == "c4u18" else _FRACTIONS_HASH
    insufficient = (
        verifier_result.recommendation == VerificationRecommendation.FALLBACK
        or bool(verifier_result.missing_evidence)
        or "insufficient" in " ".join(verifier_result.issues or []).lower()
    )

    row: dict[str, Any] = {
        "experiment": _EXPERIMENT_NAME,
        "pipeline": pipeline.value,
        "fixture_class": fixture_class,
        "primary_unit": spec.get("primary_unit"),
        "question": spec["question"],
        "answer": spec["answer"],
        "answer_hash": answer_hash(spec["answer"]),
        "evidence_hash": baseline_hash,
        "raw_evidence_hash": evidence_snapshot_hash(raw_evidence),
        "normalized_evidence_hash": evidence_snapshot_hash(verify_evidence),
        "evidence_source": evidence_source,
        "normalization_applied": pipeline
        in {Pipeline.B_NORMALIZED_VERIFIER, Pipeline.D_NORMALIZED_VERIFIER_MAPPER},
        "mapping_applied": pipeline
        in {Pipeline.C_RAW_VERIFIER_MAPPER, Pipeline.D_NORMALIZED_VERIFIER_MAPPER},
        "normalization_diagnostics": norm_diag,
        "mapping_threshold": threshold if mapping else None,
        "verifier_score": verifier_result.score,
        "verifier_accepted": verifier_result.passed,
        "verifier_decision": verifier_result.recommendation.value,
        "retrieve_more_requested": verifier_result.recommendation
        == VerificationRecommendation.RETRIEVE_MORE,
        "insufficient_evidence": insufficient,
        "unsupported_claims": list(verifier_result.unsupported_claims or []),
        "issue_codes": list(verifier_result.issues or []),
        "claim_classifications": claims,
        "raw_verifier_accepted": raw_verifier_accepted,
        "final_accepted": final_accepted,
        "final_recommendation": _final_recommendation(
            pipeline=pipeline, verifier_result=verifier_result, mapping=mapping
        ),
        "attribution": attribution,
        "placeholder_detected": placeholder_detected,
        "verifier_result_snapshot": verifier_snapshot,
    }
    if mapping:
        row.update(
            {
                "mapped_recommendation": mapping.mapped_recommendation.value,
                "mapped_accepted": mapping.mapped_accepted,
                "policy_applied": mapping.policy_applied,
                "policy_rule": mapping.policy_rule,
                "mapper_input": {
                    "score": verifier_result.score,
                    "recommendation": verifier_result.recommendation.value,
                    "unsupported_claims": list(verifier_result.unsupported_claims or []),
                },
                "mapper_output": mapping.mapped_recommendation.value,
            }
        )

    if fixture_class in {"FAITHFUL_COMPLETE", "NORMALIZATION_ONLY_GROUNDING"}:
        normalized = normalize_evidence(
            raw_evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION
        )
        row["c4u18_trace"] = _build_c4u18_trace(
            spec=spec,
            raw_evidence=raw_evidence,
            normalized_evidence=normalized.evidence,
            norm_diag=norm_diag or normalized.diagnostics.to_dict(),
            verifier_result=verifier_result,
            mapping=mapping,
            row=row,
        )
    return row


def _build_c4u18_trace(
    *,
    spec: dict[str, Any],
    raw_evidence: list[CurriculumEvidence],
    normalized_evidence: list[CurriculumEvidence],
    norm_diag: dict[str, Any],
    verifier_result: VerificationResult,
    mapping: Any | None,
    row: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_evidence(
        raw_evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION
    )
    chain = _evidence_chain_for_fc(
        raw_evidence=raw_evidence,
        normalized=normalized,
        result=verifier_result,
        answer=spec["answer"],
    )
    lo01 = next((c for c in chain if c.get("lo_id") == "C4U18-LO01"), chain[0] if chain else {})
    return {
        "question": spec["question"],
        "answer": spec["answer"],
        "raw_evidence_count": len(raw_evidence),
        "normalized_evidence_count": len(normalized_evidence),
        "normalization_transformations": norm_diag.get("operations", []),
        "lo_id": lo01.get("lo_id"),
        "raw_topic_value": lo01.get("topic_raw"),
        "resolved_topic_value": lo01.get("topic_resolved"),
        "verifier_claims": [c.model_dump() for c in (verifier_result.claims or [])],
        "verifier_score": verifier_result.score,
        "verifier_recommendation": verifier_result.recommendation.value,
        "mapper_input": row.get("mapper_input"),
        "mapper_output": row.get("mapper_output"),
        "final_recommendation": row.get("final_recommendation"),
        "evidence_chain": chain,
    }


def summarize_pipeline_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1

    def rate(fixture: str, field: str = "final_accepted") -> float:
        subset = [r for r in rows if r["fixture_class"] == fixture]
        if not subset:
            return 0.0
        if field == "final_accepted":
            return round(sum(1 for r in subset if r.get("final_accepted")) / len(subset), 3)
        if field == "verifier_accepted":
            return round(sum(1 for r in subset if r.get("verifier_accepted")) / len(subset), 3)
        return 0.0

    safety_fixtures = {
        "UNSUPPORTED_CLAIM",
        "UNSUPPORTED_ABSENCE",
        "SPECULATIVE",
        "RECONSTRUCTION",
        "MISSING_EVIDENCE",
        "HIGH_SCORE_UNSUPPORTED",
    }
    safety_false = sum(
        1
        for r in rows
        if r["fixture_class"] in safety_fixtures and r.get("final_accepted")
    )
    placeholder_false = sum(
        1
        for r in rows
        if r["fixture_class"] in {"CLEAN_PLACEHOLDER", "PLACEHOLDER_PLUS_HIGH_SCORE"}
        and r.get("final_accepted")
    )
    missing_false = sum(
        1
        for r in rows
        if r["fixture_class"] == "MISSING_EVIDENCE" and r.get("final_accepted")
    )
    scores = [float(r.get("verifier_score") or 0) for r in rows]

    return {
        "n": len(rows),
        "acceptance_rate": round(sum(1 for r in rows if r.get("final_accepted")) / n, 3),
        "verifier_acceptance_rate": round(
            sum(1 for r in rows if r.get("verifier_accepted")) / n, 3
        ),
        "faithful_complete_acceptance": rate("FAITHFUL_COMPLETE"),
        "faithful_imperfect_acceptance": rate("FAITHFUL_IMPERFECT"),
        "placeholder_acceptance": rate("CLEAN_PLACEHOLDER"),
        "placeholder_plus_acceptance": rate("PLACEHOLDER_PLUS_HIGH_SCORE"),
        "safety_false_acceptance": safety_false,
        "placeholder_false_acceptance": placeholder_false,
        "missing_evidence_false_acceptance": missing_false,
        "avg_verifier_score": round(sum(scores) / len(scores), 3) if scores else None,
    }


def build_integration_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixtures = INTEGRATION_FIXTURE_CLASSES
    pipeline_map = {
        Pipeline.A_RAW_VERIFIER.value: "raw_verifier",
        Pipeline.B_NORMALIZED_VERIFIER.value: "normalized_verifier",
        Pipeline.C_RAW_VERIFIER_MAPPER.value: "raw_mapper",
        Pipeline.D_NORMALIZED_VERIFIER_MAPPER.value: "normalized_mapper",
    }
    out: list[dict[str, Any]] = []
    for fixture in fixtures:
        entry: dict[str, Any] = {"fixture": fixture}
        for pipeline_value, key in pipeline_map.items():
            subset = [
                r
                for r in rows
                if r["fixture_class"] == fixture and r["pipeline"] == pipeline_value
            ]
            if not subset:
                entry[key] = None
                continue
            if pipeline_value in {
                Pipeline.A_RAW_VERIFIER.value,
                Pipeline.B_NORMALIZED_VERIFIER.value,
            }:
                rate_val = sum(1 for r in subset if r.get("verifier_accepted")) / len(subset)
            else:
                rate_val = sum(1 for r in subset if r.get("final_accepted")) / len(subset)
            entry[key] = round(rate_val, 3)
        out.append(entry)
    return out


def summarize_pipeline_d_threshold(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Threshold sweep for Pipeline D using stored verifier outputs."""
    d_rows = [r for r in rows if r["pipeline"] == Pipeline.D_NORMALIZED_VERIFIER_MAPPER.value]
    remapped = [remap_row_for_threshold(r, threshold) for r in d_rows]
    n = len(remapped) or 1

    def rate(fixture: str) -> float:
        subset = [r for r in remapped if r["fixture_class"] == fixture]
        if not subset:
            return 0.0
        return round(sum(1 for r in subset if r.get("mapped_accepted")) / len(subset), 3)

    safety_fixtures = {
        "UNSUPPORTED_CLAIM",
        "UNSUPPORTED_ABSENCE",
        "SPECULATIVE",
        "RECONSTRUCTION",
        "MISSING_EVIDENCE",
        "HIGH_SCORE_UNSUPPORTED",
    }
    safety_false = sum(
        1
        for r in remapped
        if r["fixture_class"] in safety_fixtures and r.get("mapped_accepted")
    )
    missing_false = sum(
        1
        for r in remapped
        if r["fixture_class"] == "MISSING_EVIDENCE" and r.get("mapped_accepted")
    )

    return {
        "threshold": threshold,
        "faithful_complete_acceptance": rate("FAITHFUL_COMPLETE"),
        "faithful_imperfect_acceptance": rate("FAITHFUL_IMPERFECT"),
        "placeholder_acceptance": max(rate("CLEAN_PLACEHOLDER"), rate("PLACEHOLDER_PLUS_HIGH_SCORE")),
        "safety_false_acceptance": safety_false,
        "missing_evidence_false_acceptance": missing_false,
        "overall_acceptance": round(sum(1 for r in remapped if r.get("mapped_accepted")) / n, 3),
    }


def build_safety_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "unsupported_claim": ["UNSUPPORTED_CLAIM", "HIGH_SCORE_UNSUPPORTED"],
        "unsupported_absence": ["UNSUPPORTED_ABSENCE"],
        "speculative": ["SPECULATIVE"],
        "reconstruction": ["RECONSTRUCTION"],
        "missing_evidence": ["MISSING_EVIDENCE"],
        "clean_placeholder": ["CLEAN_PLACEHOLDER", "PLACEHOLDER_PLUS_HIGH_SCORE"],
    }
    out: dict[str, Any] = {}
    false_accept = 0
    for key, fixtures in groups.items():
        subset = [r for r in rows if r["fixture_class"] in fixtures]
        accepted = sum(1 for r in subset if r.get("final_accepted"))
        rejected = len(subset) - accepted
        if key != "clean_placeholder":
            false_accept += accepted
        out[key] = {
            "cases": len(subset),
            "correctly_rejected": rejected,
            "incorrectly_accepted": accepted,
        }
    out["false_acceptance_total"] = false_accept
    return out


def build_adversarial_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    adv_rows = [r for r in rows if r["fixture_class"].startswith("ADV_")]
    out: dict[str, Any] = {}
    for fixture in ADVERSARIAL_FIXTURE_CLASSES:
        subset = [r for r in adv_rows if r["fixture_class"] == fixture]
        if not subset:
            continue
        out[fixture] = {
            "pipelines": {
                r["pipeline"]: {
                    "final_accepted": r.get("final_accepted"),
                    "final_recommendation": r.get("final_recommendation"),
                    "verifier_score": r.get("verifier_score"),
                    "attribution": r.get("attribution"),
                }
                for r in subset
            },
            "any_false_acceptance": any(r.get("final_accepted") for r in subset),
        }
    out["total_false_acceptance"] = sum(
        1 for r in adv_rows if r.get("final_accepted")
    )
    return out


def interpret_v210(
    *,
    integration_comparison: list[dict[str, Any]],
    pipeline_d_summary: dict[str, Any],
    safety: dict[str, Any],
    adversarial: dict[str, Any],
    threshold_sweep_d: list[dict[str, Any]],
) -> tuple[str, str, str, str]:
    d_row = next(
        (r for r in integration_comparison if r["fixture"] == "FAITHFUL_COMPLETE"),
        {},
    )
    fi_row = next(
        (r for r in integration_comparison if r["fixture"] == "FAITHFUL_IMPERFECT"),
        {},
    )
    fc_d = d_row.get("normalized_mapper")
    fi_d = fi_row.get("normalized_mapper")
    analytical = next(
        (r for r in threshold_sweep_d if r["threshold"] == _ANALYTICAL_THRESHOLD),
        threshold_sweep_d[0] if threshold_sweep_d else {},
    )
    false_accept = safety.get("false_acceptance_total", 0)
    adv_false = adversarial.get("total_false_acceptance", 0)
    placeholder_accept = analytical.get("placeholder_acceptance", 1.0)
    fi_accept = analytical.get("faithful_imperfect_acceptance", 0.0)
    fc_accept = analytical.get("faithful_complete_acceptance", 0.0)

    arch_answer = (
        "Yes — experimentally validated harness architecture: "
        "Evidence Normalization → Verifier → Recommendation Mapping → Routing. "
        "Normalization and mapper remain experiment-only; production-hardening, "
        "adversarial stress testing, and shadow evaluation still required."
    )

    core_targets_met = (
        fc_accept >= 0.9
        and fi_accept >= 0.75
        and false_accept == 0
        and placeholder_accept == 0.0
    )

    if core_targets_met and adv_false == 0:
        conclusion = "SUPPORTED"
        note = (
            "Integrated pipeline reproduces V2.9 FC grounding and V2.8 FI mapping "
            "with zero safety/placeholder/adversarial false acceptance."
        )
        v211 = "controlled production-shadow evaluation"
    elif core_targets_met and adv_false > 0:
        conclusion = "PARTIALLY_SUPPORTED"
        note = (
            "Core integration fixtures meet FC/FI/safety/placeholder targets, but "
            f"{adv_false} adversarial metadata-corruption cases were verifier-accepted "
            "(verifier does not enforce subject/grade/parent integrity)."
        )
        v211 = "adversarial grounding stress test"
        arch_answer = (
            "Partially — harness validates Normalization → Verifier → Recommendation "
            "Mapping for primary fixtures (C4-U18 FC, FI mapping, safety, placeholders). "
            "Production-hardening required for metadata-integrity adversarial cases "
            "(wrong subject/grade, conflicting parent, placeholder parent)."
        )
    elif false_accept == 0 and placeholder_accept == 0.0:
        conclusion = "PARTIALLY_SUPPORTED"
        note = (
            "Layers compose without core safety false-acceptance, but integrated metrics "
            "show residual gaps versus individual experiment targets."
        )
        v211 = "recommendation-mapper calibration experiment"
    else:
        conclusion = "NOT_SUPPORTED"
        note = "Integration introduced unsafe acceptance on core safety fixtures."
        v211 = "adversarial grounding stress test"
        arch_answer = (
            "No — integration failed core safety invariants. "
            "Unresolved boundary: combined normalization + mapping under adversarial conditions."
        )

    return conclusion, note, v211, arch_answer
