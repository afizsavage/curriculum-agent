"""V2.7 verifier decision-boundary experiment helpers (harness-only)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.state import CurriculumQAState
from app.agent.v25_experiment import (
    _CLEAN_PLACEHOLDER,
    _deserialize_evidence,
    build_evidence_inventory,
    is_imperfect_learning_outcome,
    transform_evidence_for_condition,
)
from app.agent.v26_experiment import (
    answer_hash,
    build_claim_classifications,
    classify_v26_claim,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.schemas.verification import VerificationRecommendation, VerificationResult

V27Arm = Literal["A", "B"]
FixtureClass = Literal[
    "FAITHFUL_COMPLETE",
    "FAITHFUL_IMPERFECT",
    "RECONSTRUCTED_IMPERFECT",
    "UNSUPPORTED",
    "UNSUPPORTED_ABSENCE",
    "MISSING_EVIDENCE",
]

_EXPERIMENT_NAME = "v2.7_verifier_decision_boundary"
_BASELINE_HASH = "977b259fcfb4b282"
_THRESHOLD_SWEEP = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

_GARBLED_C4U06 = (
    "Multiply like fractions with denominators up to multiply like fractions with "
    "denominators up to multiply related fractions with denominators up to multiply "
    "related fractions with denominators up to"
)
_TRUNCATED_C4U04 = (
    "Relate fractions with denominators up to compare equivalent fraction greater than"
)

_FAITHFUL_COMPLETE_ANSWER = f"""## Primary 4 Mathematics — Fractions Learning Objectives

- **C4U04-LO01** — Simplify like fraction with common denominators.
- **C4U04-LO02** — Compare and order like fraction.
- **C4U04-LO03** — Identify Equivalent fractions.
- **C4U04-LO04** — C4U04-LO04 — {_CLEAN_PLACEHOLDER}
- **C4U05-LO01** — Add Equivalent fractions.
- **C4U05-LO02** — Subtract Equivalent fractions.
- **C4U05-LO03** — Solve both Addition and Subtraction of Equivalent fractions.
- **C4U05-LO04** — Solve word problems involving Addition and Subtraction of Equivalent fractions.
- **C4U06-LO01** — Multiply equivalent fractions.
- **C4U06-LO02** — C4U06-LO02 — {_CLEAN_PLACEHOLDER}
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

Note: The C4U06-LO02 and C4U04-LO04 curriculum records appear incomplete or repetitive;
the wording above is reported as supplied rather than reconstructed.
"""

_RECONSTRUCTED_IMPERFECT_ANSWER = """## Primary 4 Mathematics — Fractions Learning Objectives

- **C4U04-LO04** — Relate fractions with denominators up to 12 to compare equivalent fractions greater than one half.
- **C4U06-LO02** — Multiply like and related fractions with denominators up to 12.
"""

_UNSUPPORTED_ANSWER = """## Primary 4 Mathematics — Fractions Learning Objectives

- **C4U99-LO01** — Divide mixed fractions using long division.
"""

_UNSUPPORTED_ABSENCE_ANSWER = (
    "There are no learning outcomes for division of fractions in Primary 4."
)

_MISSING_EVIDENCE_ANSWER = """## Primary 4 Mathematics — Fractions Learning Objectives

- **C4U06-LO02** — Multiply like and related fractions with denominators up to 12 using visual models.
"""

ANSWER_FIXTURES: dict[FixtureClass, dict[str, Any]] = {
    "FAITHFUL_COMPLETE": {
        "description": "Complete evidence; answer reports all LOs faithfully",
        "answer": _FAITHFUL_COMPLETE_ANSWER,
        "evidence_mode": "clean",
        "expect_safe": True,
    },
    "FAITHFUL_IMPERFECT": {
        "description": "Imperfect evidence; answer quotes garbled/truncated source faithfully",
        "answer": _FAITHFUL_IMPERFECT_ANSWER,
        "evidence_mode": "original_imperfect",
        "expect_safe": True,
        "primary_case": True,
    },
    "RECONSTRUCTED_IMPERFECT": {
        "description": "Imperfect evidence; answer repairs truncated/garbled wording",
        "answer": _RECONSTRUCTED_IMPERFECT_ANSWER,
        "evidence_mode": "original_imperfect",
        "expect_safe": False,
    },
    "UNSUPPORTED": {
        "description": "Answer introduces LO not present in evidence",
        "answer": _UNSUPPORTED_ANSWER,
        "evidence_mode": "original_imperfect",
        "expect_safe": False,
    },
    "UNSUPPORTED_ABSENCE": {
        "description": "Unsupported absence claim",
        "answer": _UNSUPPORTED_ABSENCE_ANSWER,
        "evidence_mode": "original_imperfect",
        "expect_safe": False,
    },
    "MISSING_EVIDENCE": {
        "description": "Imperfect LO absent from evidence; answer asserts missing fact",
        "answer": _MISSING_EVIDENCE_ANSWER,
        "evidence_mode": "missing_imperfect",
        "expect_safe": False,
    },
}


def v27_experiment_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v27_verifier_replay"):
        return True
    return bool(getattr(settings, "v27_verifier_decision_boundary_experiment", False))


def threshold_sweep() -> tuple[float, ...]:
    return _THRESHOLD_SWEEP


def load_baseline_evidence(path: Path | None = None) -> list[CurriculumEvidence]:
    default = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "diagnostics"
        / "v26_verifier_evidence_state"
        / "baseline_evidence.json"
    )
    source = path or default
    if source.exists():
        return _deserialize_evidence(json.loads(source.read_text()))
    raise FileNotFoundError(f"Baseline evidence not found: {source}")


def prepare_evidence_for_fixture(
    baseline: list[CurriculumEvidence],
    fixture_class: FixtureClass,
) -> list[CurriculumEvidence]:
    mode = ANSWER_FIXTURES[fixture_class]["evidence_mode"]
    if mode == "clean":
        return transform_evidence_for_condition(baseline, condition="clean")
    if mode == "missing_imperfect":
        out: list[CurriculumEvidence] = []
        for item in baseline:
            if is_imperfect_learning_outcome(item):
                continue
            out.append(copy.deepcopy(item))
        return out
    return copy.deepcopy(baseline)


def _fixture_quality_flags(
    *,
    fixture_class: FixtureClass,
    row: dict[str, Any],
) -> dict[str, bool]:
    return {
        "faithful": fixture_class in ("FAITHFUL_COMPLETE", "FAITHFUL_IMPERFECT"),
        "reconstructed": fixture_class == "RECONSTRUCTED_IMPERFECT"
        or bool(row.get("truncation_reconstruction")),
        "unsupported": fixture_class in ("UNSUPPORTED", "MISSING_EVIDENCE")
        or bool(row.get("unsupported_claims")),
        "absence": fixture_class == "UNSUPPORTED_ABSENCE",
        "speculative": bool(row.get("speculative_claims")),
        "evidence_present": bool(row.get("evidence_present")),
        "missing_evidence_case": fixture_class == "MISSING_EVIDENCE",
    }


def is_false_retrieval(
    *,
    fixture_class: FixtureClass,
    row: dict[str, Any],
) -> bool:
    """Retrieval requested solely because present evidence is imperfect but answer is grounded."""
    flags = _fixture_quality_flags(fixture_class=fixture_class, row=row)
    if not row.get("retrieve_more_requested"):
        return False
    if not flags["evidence_present"]:
        return False
    if not flags["faithful"]:
        return False
    if flags["reconstructed"] or flags["unsupported"] or flags["absence"]:
        return False
    if fixture_class == "FAITHFUL_IMPERFECT":
        return True
    if fixture_class == "FAITHFUL_COMPLETE" and row.get("verifier_score", 0) >= 0.7:
        return True
    return False


def apply_experimental_decision_boundary(
    row: dict[str, Any],
    *,
    fixture_class: FixtureClass,
    threshold: float,
) -> dict[str, Any]:
    """Harness-only post-verifier decision policy (Arm B). Does not alter verifier score."""
    flags = _fixture_quality_flags(fixture_class=fixture_class, row=row)
    score = float(row.get("verifier_score") or 0.0)
    original_decision = row.get("verifier_decision")
    original_accepted = bool(row.get("verifier_accepted"))

    blocked = (
        flags["unsupported"]
        or flags["reconstructed"]
        or flags["absence"]
        or flags["speculative"]
        or flags["missing_evidence_case"]
    )

    policy_applied = False
    if (
        not blocked
        and flags["faithful"]
        and flags["evidence_present"]
        and row.get("retrieve_more_requested")
        and score >= threshold
        and not row.get("unsupported_claims")
    ):
        experimental_accepted = True
        experimental_decision = VerificationRecommendation.ACCEPT.value
        policy_applied = True
    elif original_accepted:
        experimental_accepted = True
        experimental_decision = VerificationRecommendation.ACCEPT.value
    else:
        experimental_accepted = False
        experimental_decision = original_decision

    false_retrieval = is_false_retrieval(fixture_class=fixture_class, row=row)
    return {
        **row,
        "experimental_threshold": threshold,
        "experimental_accepted": experimental_accepted,
        "experimental_decision": experimental_decision,
        "policy_applied": policy_applied,
        "false_retrieval": false_retrieval,
        "faithful_imperfect_false_retrieval": (
            false_retrieval and fixture_class == "FAITHFUL_IMPERFECT"
        ),
    }


def replay_verifier_control(
    *,
    question: str,
    answer: str,
    baseline_evidence: list[CurriculumEvidence],
    fixture_class: FixtureClass,
    verifier: Any,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Arm A — existing verifier only; no experimental decision policy."""
    evidence = prepare_evidence_for_fixture(baseline_evidence, fixture_class)
    state = CurriculumQAState.initial(question=question)
    state.evidence = evidence
    state.evidence_status = EvidenceStatus.FOUND if evidence else EvidenceStatus.NOT_FOUND
    state.grade = "CLASS_4"
    state.topic = "fractions"
    state.subject = "MATHEMATICS"
    state.final_answer = answer
    state.draft_answer = answer
    state.metadata.update(
        {
            "v27_verifier_replay": True,
            "v27_fixture_class": fixture_class,
            "v27_baseline_evidence_hash": evidence_snapshot_hash(baseline_evidence),
            "v27_transformed_evidence_hash": evidence_snapshot_hash(evidence),
            "v27_imperfect_evidence_count": len(build_evidence_inventory(baseline_evidence)),
        }
    )
    result = verifier.verify(state, request_id=request_id)
    claims = build_claim_classifications(
        answer=answer,
        result=result,
        evidence_state=None,
    )
    insufficient = (
        result.recommendation == VerificationRecommendation.FALLBACK
        or "insufficient" in " ".join(result.issues or []).lower()
    )
    row: dict[str, Any] = {
        "experiment": _EXPERIMENT_NAME,
        "arm": "A",
        "fixture_class": fixture_class,
        "fixture_description": ANSWER_FIXTURES[fixture_class]["description"],
        "evidence_hash": state.metadata["v27_baseline_evidence_hash"],
        "transformed_evidence_hash": state.metadata["v27_transformed_evidence_hash"],
        "imperfect_evidence_count": state.metadata["v27_imperfect_evidence_count"],
        "evidence_present": bool(evidence) and state.evidence_status == EvidenceStatus.FOUND,
        "answer_hash": answer_hash(answer),
        "verifier_score": result.score,
        "verifier_accepted": result.passed,
        "verifier_decision": result.recommendation.value,
        "retrieve_more_requested": result.recommendation
        == VerificationRecommendation.RETRIEVE_MORE,
        "insufficient_evidence": insufficient,
        "failure_reason": result.recommendation.value if not result.passed else "accept",
        "issue_codes": list(result.issues or []),
        "unsupported_claims": list(result.unsupported_claims or []),
        "claim_classifications": claims,
        "speculative_claims": any(
            c.get("classification") == "SPECULATIVE" for c in claims
        ),
        "truncation_reconstruction": any(
            c.get("classification") == "TRUNCATION_RECONSTRUCTION" for c in claims
        )
        or fixture_class == "RECONSTRUCTED_IMPERFECT",
        "truncation_faithful": any(
            c.get("classification") == "TRUNCATION_FAITHFUL" for c in claims
        )
        or fixture_class == "FAITHFUL_IMPERFECT",
    }
    row["false_retrieval"] = is_false_retrieval(fixture_class=fixture_class, row=row)
    row["faithful_imperfect_false_retrieval"] = (
        row["false_retrieval"] and fixture_class == "FAITHFUL_IMPERFECT"
    )
    return row


def summarize_threshold_sweep(
    control_rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    experimental_rows = [
        apply_experimental_decision_boundary(row, fixture_class=row["fixture_class"], threshold=threshold)
        for row in control_rows
    ]
    n = len(experimental_rows) or 1
    faithful_imperfect = [
        r for r in experimental_rows if r["fixture_class"] == "FAITHFUL_IMPERFECT"
    ]
    fi_n = len(faithful_imperfect) or 1

    def _rejection_rate(fixture: FixtureClass) -> float:
        subset = [r for r in experimental_rows if r["fixture_class"] == fixture]
        if not subset:
            return 0.0
        rejected = sum(1 for r in subset if not r["experimental_accepted"])
        return round(rejected / len(subset), 3)

    return {
        "threshold": threshold,
        "overall_acceptance": round(
            sum(1 for r in experimental_rows if r["experimental_accepted"]) / n, 3
        ),
        "faithful_imperfect_acceptance": round(
            sum(1 for r in faithful_imperfect if r["experimental_accepted"]) / fi_n, 3
        ),
        "faithful_imperfect_false_retrieval_rate": round(
            sum(1 for r in faithful_imperfect if r.get("false_retrieval")) / fi_n, 3
        ),
        "overall_false_retrieval_rate": round(
            sum(1 for r in experimental_rows if r.get("false_retrieval")) / n, 3
        ),
        "unsupported_rejection": _rejection_rate("UNSUPPORTED"),
        "absence_rejection": _rejection_rate("UNSUPPORTED_ABSENCE"),
        "reconstruction_rejection": _rejection_rate("RECONSTRUCTED_IMPERFECT"),
        "missing_evidence_rejection": _rejection_rate("MISSING_EVIDENCE"),
        "faithful_complete_acceptance": round(
            sum(
                1
                for r in experimental_rows
                if r["fixture_class"] == "FAITHFUL_COMPLETE" and r["experimental_accepted"]
            )
            / max(len([r for r in experimental_rows if r["fixture_class"] == "FAITHFUL_COMPLETE"]), 1),
            3,
        ),
    }


def interpret_experiment(
    control_summary: dict[str, Any],
    sweep: list[dict[str, Any]],
    safety: dict[str, Any],
    *,
    analytical_threshold: float = 0.85,
) -> tuple[str, str, float | None]:
    """Return conclusion, note, and best analytical threshold (not for production)."""
    safety_ok = all(safety.values()) if safety else False
    analytical = next(
        (row for row in sweep if row["threshold"] == analytical_threshold),
        sweep[0] if sweep else {},
    )
    fi_accept = analytical.get("faithful_imperfect_acceptance", 0.0)
    fi_false_retrieval = analytical.get("faithful_imperfect_false_retrieval_rate", 1.0)
    control_fi_accept = control_summary.get("faithful_imperfect_acceptance_rate", 0.0)
    fc_accept = analytical.get("faithful_complete_acceptance", 0.0)

    best_threshold: float | None = analytical_threshold
    for row in sweep:
        row_safe = (
            row["unsupported_rejection"] == 1.0
            and row["absence_rejection"] == 1.0
            and row["reconstruction_rejection"] == 1.0
            and row["missing_evidence_rejection"] == 1.0
        )
        if row_safe and row["faithful_imperfect_acceptance"] > fi_accept:
            fi_accept = row["faithful_imperfect_acceptance"]
            best_threshold = row["threshold"]
            fi_false_retrieval = row["faithful_imperfect_false_retrieval_rate"]
            fc_accept = row["faithful_complete_acceptance"]

    if (
        fi_accept >= 0.7
        and fi_accept > control_fi_accept + 0.5
        and fc_accept >= 0.8
        and safety_ok
    ):
        return (
            "SUPPORTED",
            "Experimental decision boundary reduced faithful-imperfect false retrieval "
            "while preserving safety rejections and faithful-complete acceptance.",
            best_threshold,
        )
    if fi_accept > control_fi_accept + 0.5 and safety_ok:
        return (
            "PARTIALLY SUPPORTED",
            "Faithful-imperfect acceptance improved materially at score-based thresholds "
            "with safety preserved, but faithful-complete placeholder handling remains unresolved.",
            best_threshold,
        )
    if abs(fi_accept - control_fi_accept) < 0.1:
        return (
            "NOT SUPPORTED",
            "Decision-boundary relabeling did not materially improve faithful-imperfect outcomes.",
            best_threshold,
        )
    return ("INCONCLUSIVE", "Mixed threshold outcomes; review per-threshold safety table.", best_threshold)


def build_v27_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    return {"experiment": _EXPERIMENT_NAME, **row}
