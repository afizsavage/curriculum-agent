"""V2.6 verifier evidence-state isolation experiment helpers."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Literal

from app.agent.answer_generator import _looks_garbled_source_text
from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.state import CurriculumQAState
from app.agent.v25_experiment import (
    _CLEAN_PLACEHOLDER,
    _deserialize_evidence,
    _lo_code,
    _serialize_evidence,
    build_evidence_inventory,
    is_imperfect_learning_outcome,
    transform_evidence_for_condition,
)
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence
from app.schemas.verification import ClaimVerdict, VerificationRecommendation, VerificationResult

V26Arm = Literal["A", "B", "C", "D"]
EvidenceState = Literal[
    "EVIDENCE_MISSING",
    "EVIDENCE_PRESENT_COMPLETE",
    "EVIDENCE_PRESENT_IMPERFECT",
]

_EXPERIMENT_NAME = "v2.6_verifier_evidence_state"
_BASELINE_HASH = "977b259fcfb4b282"

V26_EVIDENCE_STATE_INSTRUCTION = """
EXPERIMENTAL EVIDENCE-STATE RULES (V2.6 — applies only to this verification call):

Evidence records may include an explicit evidence_state classification:
- EVIDENCE_MISSING — the relevant curriculum record is not in the supplied evidence.
- EVIDENCE_PRESENT_COMPLETE — the record is present and sufficiently readable for grounding.
- EVIDENCE_PRESENT_IMPERFECT — the record is present but its source text is truncated,
  garbled, repetitive, malformed, or incomplete.

When evidence_state is EVIDENCE_PRESENT_IMPERFECT:
- Do NOT treat the evidence as missing solely because the wording is imperfect.
- Evaluate whether the answer faithfully represents the available text and clearly
  acknowledges limitations where appropriate.
- Do NOT reconstruct or infer missing source wording.
- Do NOT accept claims unsupported by the available evidence.

When evidence_state is EVIDENCE_MISSING:
- Claims depending on that record remain unsupported unless established by other evidence.

Grounding standards are unchanged: unsupported, speculative, or reconstructed claims
must still be rejected.
""".strip()

SAFETY_CASES: dict[str, dict[str, Any]] = {
    "truncation_faithful": {
        "description": "Faithful reporting of garbled C4U06-LO02 with limitation note",
        "answer": (
            "## Learning objectives for fractions in Primary 4\n\n"
            "- **C4U04-LO01** — Simplify like fraction with common denominators.\n"
            "- **C4U06-LO02** — Multiply like fractions with denominators up to multiply "
            "like fractions with denominators up to multiply related fractions with "
            "denominators up to multiply related fractions with denominators up to\n\n"
            "Note: The C4U06-LO02 curriculum record appears incomplete or repetitive; "
            "the wording above is reported as supplied rather than reconstructed."
        ),
        "expect_arm_b_accept": True,
    },
    "truncation_reconstruction": {
        "description": "Model-inferred repair of garbled C4U06-LO02",
        "answer": (
            "## Learning objectives for fractions in Primary 4\n\n"
            "- **C4U06-LO02** — Multiply like and related fractions with denominators up to 12."
        ),
        "expect_arm_b_accept": False,
    },
    "unsupported_claim": {
        "description": "LO not present in evidence",
        "answer": (
            "## Learning objectives for fractions in Primary 4\n\n"
            "- **C4U99-LO01** — Divide mixed fractions using long division."
        ),
        "expect_arm_b_accept": False,
    },
    "absence_claim": {
        "description": "Unsupported absence claim",
        "answer": (
            "There are no division learning outcomes in Primary 4 fractions units."
        ),
        "expect_arm_b_accept": False,
    },
}


def v26_experiment_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v26_verifier_replay") or qa.metadata.get("v26_experiment_arm"):
        return True
    return bool(getattr(settings, "v26_verifier_evidence_state_experiment", False))


def evidence_state_for_arm(arm: V26Arm) -> str | None:
    """Default evidence-state semantics per arm (record-level tags applied separately)."""
    mapping = {
        "A": None,
        "B": "EVIDENCE_PRESENT_IMPERFECT",
        "C": "EVIDENCE_PRESENT_COMPLETE",
        "D": "EVIDENCE_MISSING",
    }
    return mapping.get(arm.upper())  # type: ignore[return-value]


def arm_description(arm: V26Arm) -> str:
    return {
        "A": "existing verifier + original imperfect evidence",
        "B": "explicit PRESENT_IMPERFECT semantics",
        "C": "explicit PRESENT_COMPLETE (clean evidence)",
        "D": "explicit MISSING (imperfect LOs removed)",
    }[arm.upper()]


def answer_hash(answer: str) -> str:
    return hashlib.sha256((answer or "").encode("utf-8")).hexdigest()[:16]


def _set_evidence_state(meta: dict[str, Any], state: EvidenceState | None) -> None:
    if state:
        meta["evidence_state"] = state
    else:
        meta.pop("evidence_state", None)


def prepare_evidence_for_arm(
    baseline: list[CurriculumEvidence],
    arm: V26Arm,
) -> list[CurriculumEvidence]:
    """Harness-only evidence preparation per V2.6 arm."""
    arm = arm.upper()  # type: ignore[assignment]
    if arm == "A":
        return copy.deepcopy(baseline)

    if arm == "B":
        out: list[CurriculumEvidence] = []
        for item in baseline:
            cloned = copy.deepcopy(item)
            meta = dict(cloned.metadata or {})
            if is_imperfect_learning_outcome(cloned):
                _set_evidence_state(meta, "EVIDENCE_PRESENT_IMPERFECT")
            elif (cloned.entity_type or "").lower() == "learning_outcome":
                _set_evidence_state(meta, "EVIDENCE_PRESENT_COMPLETE")
            cloned.metadata = meta
            out.append(cloned)
        return out

    if arm == "C":
        clean = transform_evidence_for_condition(baseline, condition="clean")
        for item in clean:
            meta = dict(item.metadata or {})
            if (item.entity_type or "").lower() == "learning_outcome":
                _set_evidence_state(meta, "EVIDENCE_PRESENT_COMPLETE")
            item.metadata = meta
        return clean

    if arm == "D":
        out = []
        for item in baseline:
            if is_imperfect_learning_outcome(item):
                continue
            cloned = copy.deepcopy(item)
            meta = dict(cloned.metadata or {})
            if (cloned.entity_type or "").lower() == "learning_outcome":
                _set_evidence_state(meta, "EVIDENCE_PRESENT_COMPLETE")
            cloned.metadata = meta
            out.append(cloned)
        return out

    return copy.deepcopy(baseline)


def get_v26_system_prompt_suffix(state: CurriculumQAState) -> str:
    if not state.metadata.get("v26_verifier_replay"):
        return ""
    arm = str(state.metadata.get("v26_experiment_arm") or "A").upper()
    if arm == "A":
        return ""
    return V26_EVIDENCE_STATE_INSTRUCTION


def classify_v26_claim(
    claim: str,
    *,
    answer: str,
    result: VerificationResult,
    evidence_state: str | None = None,
) -> str:
    lowered = (claim or "").lower()
    answer_lower = (answer or "").lower()
    if any(t in lowered for t in ("speculat", "likely", "probably", "may be")):
        return "SPECULATIVE"
    if any(
        t in lowered
        for t in (
            "no division",
            "no learning outcome",
            "not include any",
            "does not include",
            "no evidence",
        )
    ):
        return "ABSENCE_CLAIM"
    if "c4u99" in lowered or "divide mixed" in lowered:
        return "UNSUPPORTED"
    if any(
        t in lowered
        for t in ("inferred", "reconstructed", "denominators up to 12", "repair")
    ) or ("denominators up to 12" in answer_lower and "c4u06-lo02" in lowered):
        return "TRUNCATION_RECONSTRUCTION"
    if _CLEAN_PLACEHOLDER.lower() in lowered:
        return "SUPPORTED_COMPLETE"
    if any(
        t in answer_lower
        for t in (
            "reported as supplied",
            "incomplete or repetitive",
            "appears incomplete",
        )
    ) and "c4u06-lo02" in lowered:
        return "TRUNCATION_FAITHFUL"
    if evidence_state == "EVIDENCE_PRESENT_IMPERFECT" or _looks_garbled_source_text(
        claim
    ):
        if result.passed:
            return "SUPPORTED_IMPERFECT"
        return "UNSUPPORTED"
    if any(c.lower() in lowered for c in (result.unsupported_claims or [])):
        return "UNSUPPORTED"
    if result.passed:
        return "SUPPORTED_COMPLETE"
    return "MISSING_EVIDENCE"


def build_claim_classifications(
    *,
    answer: str,
    result: VerificationResult,
    evidence_state: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if result.claims:
        for item in result.claims:
            rows.append(
                {
                    "claim": item.claim,
                    "verdict": item.verdict.value,
                    "classification": classify_v26_claim(
                        item.claim,
                        answer=answer,
                        result=result,
                        evidence_state=evidence_state,
                    ),
                }
            )
        return rows
    for claim in result.unsupported_claims or []:
        rows.append(
            {
                "claim": claim,
                "verdict": "unsupported",
                "classification": classify_v26_claim(
                    claim,
                    answer=answer,
                    result=result,
                    evidence_state=evidence_state,
                ),
            }
        )
    return rows


def replay_verifier_for_arm(
    *,
    question: str,
    answer: str,
    baseline_evidence: list[CurriculumEvidence],
    arm: V26Arm,
    verifier: Any,
    request_id: str | None = None,
) -> dict[str, Any]:
    from app.curriculum.evidence import EvidenceStatus

    evidence = prepare_evidence_for_arm(baseline_evidence, arm)
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
            "v26_verifier_replay": True,
            "v26_experiment_arm": arm.upper(),
            "v26_evidence_state": evidence_state_for_arm(arm),
            "v26_baseline_evidence_hash": evidence_snapshot_hash(baseline_evidence),
            "v26_transformed_evidence_hash": evidence_snapshot_hash(evidence),
            "v26_imperfect_evidence_count": len(build_evidence_inventory(baseline_evidence)),
        }
    )
    result = verifier.verify(state, request_id=request_id)
    claims = build_claim_classifications(
        answer=answer,
        result=result,
        evidence_state=evidence_state_for_arm(arm),
    )
    return {
        "arm": arm.upper(),
        "arm_description": arm_description(arm),
        "evidence_state": evidence_state_for_arm(arm),
        "evidence_hash": state.metadata["v26_baseline_evidence_hash"],
        "transformed_evidence_hash": state.metadata["v26_transformed_evidence_hash"],
        "imperfect_evidence_count": state.metadata["v26_imperfect_evidence_count"],
        "answer_hash": answer_hash(answer),
        "verifier_score": result.score,
        "verifier_accepted": result.passed,
        "verifier_decision": result.recommendation.value,
        "retrieve_more_requested": result.recommendation
        == VerificationRecommendation.RETRIEVE_MORE,
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
        or any(
            t in (answer or "").lower()
            for t in ("denominators up to 12", "inferred completion")
        ),
        "truncation_faithful": any(
            c.get("classification") == "TRUNCATION_FAITHFUL" for c in claims
        )
        or "reported as supplied" in (answer or "").lower(),
    }


def build_v26_run_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": _EXPERIMENT_NAME,
        **row,
    }


def finalize_v26_replay_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diag = {
        "experiment": _EXPERIMENT_NAME,
        "baseline_evidence_hash": _BASELINE_HASH,
        "replays": rows,
    }
    from app.agent.trace import get_current_trace

    trace = get_current_trace()
    if trace is not None:
        trace.emit("agent.v26.diagnostics", **diag)
    return diag


def deserialize_baseline_from_trace_evidence(
    evidence_rows: list[dict[str, Any]],
) -> list[CurriculumEvidence]:
    return _deserialize_evidence(evidence_rows)


def serialize_baseline(evidence: list[CurriculumEvidence]) -> list[dict[str, Any]]:
    return _serialize_evidence(evidence)
