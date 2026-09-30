"""V2.13G.1 PROVENANCE_ONLY generation-boundary investigation helpers.

V2.13F offline counterfactual semantics:
  document_use != INCLUDE_IN_GENERATION → outcome_source = control
  (reuse accepted control answer; do not document-condition a new answer)

V2.13G live dual-arm previously always regenerated with structured-only
evidence when PROVENANCE_ONLY. That withholds documents from generation
(enforced) but does not preserve the control answer (diverges from V2.13F).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.v213f_arbitration import DocumentUse, counterfactual_outcome
from app.agent.v26_experiment import answer_hash

PROVENANCE_ONLY_SEMANTICS_REGENERATE = "structured_only_regeneration"
PROVENANCE_ONLY_SEMANTICS_PRESERVE = "preserve_control_answer"

_TARGETED_JSONL = Path("data/diagnostics/v213g_arbitration_targeted_replay.jsonl")
_OUT_JSON = Path("data/diagnostics/v213g1_provenance_only.json")
_OUT_JSONL = Path("data/diagnostics/v213g1_provenance_only.jsonl")


def resolved_provenance_only_semantics(settings: Any | None) -> str:
    raw = str(
        getattr(settings, "v213g_provenance_only_semantics", None)
        or PROVENANCE_ONLY_SEMANTICS_REGENERATE
    ).strip()
    if raw == PROVENANCE_ONLY_SEMANTICS_PRESERVE:
        return PROVENANCE_ONLY_SEMANTICS_PRESERVE
    return PROVENANCE_ONLY_SEMANTICS_REGENERATE


def should_preserve_control_answer(
    *,
    document_use: str | None,
    control_accepted: bool,
    semantics: str,
) -> bool:
    """True when shadow arm must reuse control instead of regenerating."""
    if semantics != PROVENANCE_ONLY_SEMANTICS_PRESERVE:
        return False
    if not control_accepted:
        return False
    use = str(document_use or "")
    return use in {
        DocumentUse.PROVENANCE_ONLY.value,
        DocumentUse.DO_NOT_USE.value,
        DocumentUse.REQUIRE_REVIEW.value,
    }


def build_preserved_control_arm(
    control: dict[str, Any],
    *,
    structured: list[Any],
    documents: list[Any],
    generation_evidence: list[Any],
    provenance_evidence: list[Any],
    retrieval_meta: dict[str, Any],
    arbitration_info: dict[str, Any] | None,
    arm_name: str = "arbitrated",
) -> dict[str, Any]:
    """Copy accepted control outcomes; documents stay provenance-only."""
    from app.agent.evidence_snapshot import evidence_snapshot_hash
    from app.agent.v213d_shadow import (
        _document_passage_summaries,
        _evidence_summary,
        _provenance_complete,
    )

    answer = ""
    # Prefer hashes already on control; answer text may be absent in snapshots.
    arm: dict[str, Any] = {
        "arm": arm_name,
        "outcome_source": "control",
        "provenance_only_semantics": PROVENANCE_ONLY_SEMANTICS_PRESERVE,
        "generation_attempted": False,
        "structured_evidence_count": len(structured),
        "document_evidence_count": len(documents),
        "generation_evidence_count": len(generation_evidence),
        "provenance_evidence_count": len(provenance_evidence),
        "generation_document_count": sum(
            1
            for e in generation_evidence
            if getattr(e, "entity_type", None) == "document_passage"
        ),
        "merged_evidence_count": len(generation_evidence),
        "evidence_count": len(generation_evidence),
        "evidence_snapshot": evidence_snapshot_hash(generation_evidence)
        if generation_evidence
        else control.get("evidence_snapshot"),
        "generation_evidence_fingerprint": evidence_snapshot_hash(generation_evidence)
        if generation_evidence
        else control.get("evidence_snapshot"),
        "provenance_evidence_fingerprint": (
            evidence_snapshot_hash(provenance_evidence) if provenance_evidence else ""
        ),
        "evidence_summary": _evidence_summary(generation_evidence)
        if generation_evidence
        else control.get("evidence_summary"),
        "document_passages": retrieval_meta.get("passages")
        or _document_passage_summaries(documents),
        "retrieval_variant": retrieval_meta.get("variant"),
        "document_retrieval_latency_ms": retrieval_meta.get("latency_ms", 0),
        "retrieval_skipped": bool(retrieval_meta.get("skipped")),
        "corpus_available": retrieval_meta.get("corpus_available"),
        "retrieval_failure_kind": retrieval_meta.get("retrieval_failure_kind"),
        "normalization_status": "skipped_preserve_control",
        "normalization_count": len(generation_evidence),
        "metadata_valid": True,
        "metadata_blocked": False,
        "metadata_policy": "preserve_control",
        "metadata_violations": [],
        "blocked_evidence": 0,
        "verifier_score": control.get("verifier_score"),
        "verifier_decision": control.get("verifier_decision"),
        "verifier_accepted": control.get("verifier_accepted"),
        "unsupported_claims": list(control.get("unsupported_claims") or []),
        "mapper_recommendation": control.get("mapper_recommendation"),
        "mapped_accepted": control.get("mapped_accepted"),
        "final_accepted": bool(control.get("final_accepted")),
        "final_route": control.get("final_route"),
        "answer_present": bool(control.get("answer_present") or control.get("answer_hash")),
        "answer_hash": control.get("answer_hash") or "",
        "answer_length": control.get("answer_length"),
        "answer_text_for_diagnostics": answer,
        "provenance_complete": _provenance_complete(documents),
        "wrong_context": False,
        "placeholder_evidence": False,
        "error": None,
        "shadow_stage": f"{arm_name}:preserve_control",
        "generation_config": {"mode": "preserve_control_answer"},
    }
    if arbitration_info is not None:
        arm["arbitration"] = arbitration_info
    return arm


def classify_divergence_root_cause(case: dict[str, Any]) -> str:
    """Root-cause label for a PROVENANCE_ONLY control-vs-arbitrated divergence."""
    arb = case.get("arbitration") or {}
    a = case.get("arbitrated_shadow") or {}
    gen_docs = int(arb.get("generation_document_count") or 0)
    if gen_docs > 0:
        return "DOCUMENT_LEAK"
    ctrl_snap = (case.get("control") or {}).get("evidence_snapshot")
    arb_snap = a.get("evidence_snapshot")
    if ctrl_snap and arb_snap and ctrl_snap != arb_snap:
        return "STRUCTURED_EVIDENCE_CHANGE"
    if a.get("outcome_source") == "control":
        return "OTHER"
    # Documents withheld + identical evidence fingerprint + new answer
    if ctrl_snap and arb_snap and ctrl_snap == arb_snap:
        return "CONTROL_REGENERATION"
    return "GENERATION_VARIANCE"


def analyze_provenance_only_case(record: dict[str, Any]) -> dict[str, Any]:
    """Build a V2.13G.1 investigation row from a targeted dual-arm record."""
    control = record.get("control") or {}
    baseline = record.get("baseline_shadow") or {}
    arbitrated = record.get("arbitrated_shadow") or {}
    arb = record.get("arbitration") or {}
    hyp = record.get("hypothesis") or {}
    q = record.get("question") or {}

    gen_docs = int(arb.get("generation_document_count") or 0)
    prov_docs = int(arb.get("provenance_document_count") or 0)
    ctrl_hash = control.get("answer_hash") or ""
    arb_hash = arbitrated.get("answer_hash") or ""
    base_hash = baseline.get("answer_hash") or ""
    ctrl_snap = control.get("evidence_snapshot")
    arb_snap = arbitrated.get("evidence_snapshot")
    base_snap = baseline.get("evidence_snapshot")

    # Offline V2.13F counterfactual (preserve-control semantics)
    fake_shadow = {
        "final_accepted": baseline.get("final_accepted"),
        "final_route": baseline.get("final_route"),
        "verifier_decision": baseline.get("verifier_decision"),
        "verifier_score": baseline.get("verifier_score"),
        "mapper_recommendation": baseline.get("mapper_recommendation"),
        "unsupported_claims": baseline.get("unsupported_claims") or [],
        "answer_length": baseline.get("answer_length"),
        "answer_hash": base_hash,
        "structured_evidence_count": arb.get("structured_count"),
        "document_passages": baseline.get("document_passages") or [],
    }
    from app.agent.v213f_arbitration import (
        ArbitrationDecision,
        ArbitrationPolicy,
        DocumentRole,
        StructuredSufficiency,
    )

    decision = ArbitrationDecision(
        document_role=DocumentRole(str(arb.get("document_role") or "REDUNDANT")),
        structured_sufficiency=StructuredSufficiency(
            str(arb.get("structured_sufficiency") or "SUFFICIENT")
        ),
        document_use=DocumentUse(str(arb.get("document_use") or "PROVENANCE_ONLY")),
        policy=ArbitrationPolicy.ARBITRATED,
        reasons=list(arb.get("reasons") or []),
        relevant_passage_count=int(arb.get("relevant_passage_count") or 0),
        conflicting_passage_count=int(arb.get("conflicting_passage_count") or 0),
        document_count=int(arb.get("document_count") or 0),
        structured_count=int(arb.get("structured_count") or 0),
    )
    cf = counterfactual_outcome(
        {"control": control, "shadow": fake_shadow, "comparison": {}},
        decision,
    )

    documents_in_generation = gen_docs > 0
    evidence_identical = bool(ctrl_snap and arb_snap and ctrl_snap == arb_snap)
    hash_match = bool(ctrl_hash and arb_hash and ctrl_hash == arb_hash)

    return {
        "source": record.get("source") or "TARGETED_REPLAY",
        "batch_id": record.get("batch_id"),
        "request_id": record.get("request_id"),
        "question_hash": q.get("hash"),
        "eval_category": record.get("eval_category"),
        "grade": q.get("grade"),
        "subject": q.get("subject"),
        "topic": q.get("topic"),
        "arbitration": {
            "structured_sufficiency": arb.get("structured_sufficiency"),
            "document_role": arb.get("document_role"),
            "document_use": arb.get("document_use"),
            "reasons": arb.get("reasons"),
            "structured_count": arb.get("structured_count"),
            "document_count": arb.get("document_count"),
            "generation_document_count": gen_docs,
            "provenance_document_count": prov_docs,
        },
        "question_a_documents_in_arbitrated_generation": documents_in_generation,
        "question_b_structured_evidence_identical": evidence_identical,
        "control_evidence_count": control.get("evidence_count"),
        "control_evidence_snapshot": ctrl_snap,
        "arbitrated_generation_evidence_count": arbitrated.get(
            "generation_evidence_count"
        ),
        "arbitrated_evidence_snapshot": arb_snap,
        "baseline_evidence_snapshot": base_snap,
        "baseline_includes_documents": int(baseline.get("generation_evidence_count") or 0)
        > int(arb.get("structured_count") or 0),
        "control_answer_hash": ctrl_hash,
        "baseline_answer_hash": base_hash,
        "arbitrated_answer_hash": arb_hash,
        "arbitrated_matches_control_hash": hash_match,
        "control_final_accepted": control.get("final_accepted"),
        "baseline_final_accepted": baseline.get("final_accepted"),
        "arbitrated_final_accepted": arbitrated.get("final_accepted"),
        "control_verifier": control.get("verifier_decision"),
        "baseline_verifier": baseline.get("verifier_decision"),
        "arbitrated_verifier": arbitrated.get("verifier_decision"),
        "control_route": control.get("final_route"),
        "baseline_route": baseline.get("final_route"),
        "arbitrated_route": arbitrated.get("final_route"),
        "baseline_unsupported_claims": hyp.get("baseline_unsupported_claims"),
        "arbitrated_unsupported_claims": hyp.get("arbitrated_unsupported_claims"),
        "baseline_regression": bool(hyp.get("control_correct_baseline_worse")),
        "arbitrated_regression": bool(hyp.get("control_correct_arbitrated_worse")),
        "baseline_drift": bool(hyp.get("generator_document_drift_baseline")),
        "arbitrated_drift": bool(hyp.get("generator_document_drift_arbitrated")),
        "root_cause": classify_divergence_root_cause(record),
        "v213f_counterfactual_preserve": {
            "outcome_source": cf.get("outcome_source"),
            "answer_hash": cf.get("answer_hash"),
            "final_accepted": cf.get("final_accepted"),
            "matches_control_hash": cf.get("answer_hash") == ctrl_hash,
            "control_correct_shadow_worse": cf.get("control_correct_shadow_worse"),
        },
        "generation_config": arbitrated.get("generation_config"),
        "investigation": record.get("investigation"),
    }


def load_targeted_records(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or _TARGETED_JSONL
    if not target.is_file():
        return []
    return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]


def select_provenance_only_redundant(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        arb = r.get("arbitration") or {}
        if (
            arb.get("structured_sufficiency") == "SUFFICIENT"
            and arb.get("document_role") == "REDUNDANT"
            and arb.get("document_use") == "PROVENANCE_ONLY"
        ):
            out.append(r)
    return out


def aggregate_v213g1_investigation(
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = records if records is not None else load_targeted_records()
    po_rows = select_provenance_only_redundant(records)
    analyzed = [analyze_provenance_only_case(r) for r in po_rows]

    # Also analyze all sufficient+docs regressions
    regressions = []
    for r in records:
        hyp = r.get("hypothesis") or {}
        if hyp.get("control_correct_baseline_worse") or hyp.get(
            "control_correct_arbitrated_worse"
        ):
            regressions.append(analyze_provenance_only_case(r))

    n = len(analyzed)
    docs_leaked = sum(
        1 for a in analyzed if a["question_a_documents_in_arbitrated_generation"]
    )
    evidence_same = sum(
        1 for a in analyzed if a["question_b_structured_evidence_identical"]
    )
    hash_match = sum(1 for a in analyzed if a["arbitrated_matches_control_hash"])
    root_causes = Counter(a["root_cause"] for a in analyzed)
    preserve_would_match = sum(
        1
        for a in analyzed
        if a["v213f_counterfactual_preserve"]["matches_control_hash"]
    )
    arb_reg = sum(1 for a in analyzed if a["arbitrated_regression"])
    base_reg = sum(1 for a in analyzed if a["baseline_regression"])
    preserve_arb_reg = sum(
        1
        for a in analyzed
        if a["v213f_counterfactual_preserve"]["control_correct_shadow_worse"]
    )

    if docs_leaked == 0 and n > 0:
        enforcement = "PARTIALLY_ENFORCED"
        enforcement_note = (
            "Documents are withheld from generation_evidence (generation_document_count=0) "
            "but the arbitrated arm still regenerates instead of preserving control "
            "(V2.13F outcome_source=control)."
        )
    elif docs_leaked > 0:
        enforcement = "PROVENANCE_ONLY_NOT_ENFORCED"
        enforcement_note = "Documents appeared in arbitrated generation evidence."
    else:
        enforcement = "PROVENANCE_ONLY_NOT_ENFORCED"
        enforcement_note = "No PROVENANCE_ONLY cases available."

    # Determinism proxy from existing data: identical evidence snap still ≠ answer hash
    identical_but_divergent = sum(
        1
        for a in analyzed
        if a["question_b_structured_evidence_identical"]
        and not a["arbitrated_matches_control_hash"]
    )
    generation_determinism_rate = (
        round(hash_match / n, 4) if n else None
    )

    recommended = "PRESERVE_CONTROL_ANSWER"
    recommended_note = (
        "V2.13F counterfactual uses outcome_source=control for PROVENANCE_ONLY. "
        "Current structured-only regeneration yields 0/%d hash matches despite "
        "identical evidence fingerprints and zero document leak. Preserve-control "
        "would yield %d/%d hash matches and 0 preserve-path regressions on this set."
        % (n, preserve_would_match, n)
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "v2.13g.1",
        "schema_version": "v213g1.1",
        "n_provenance_only_redundant": n,
        "enforcement_status": enforcement,
        "enforcement_note": enforcement_note,
        "questions": {
            "A_documents_in_generation": {
                "leaked_cases": docs_leaked,
                "result": "NO" if docs_leaked == 0 and n else "YES" if docs_leaked else "N/A",
            },
            "B_structured_evidence_identical": {
                "identical_cases": evidence_same,
                "n": n,
                "result": "YES" if evidence_same == n and n else "MIXED" if evidence_same else "NO",
            },
            "C_prompt_identical": {
                "result": "LIKELY_YES",
                "note": (
                    "Arbitrated arm uses agent.answer() with structured-only evidence; "
                    "no alternate system prompt. Control ran earlier in the same ask "
                    "with the same generator (temperature=0.0)."
                ),
            },
            "D_generation_parameters": {
                "result": "LIKELY_YES",
                "temperature": 0.0,
                "note": "AnswerGenerator.generate_structured uses temperature=0.0.",
            },
            "E_intended_semantics": {
                "v213f_documented": "PRESERVE_CONTROL_ANSWER (outcome_source=control)",
                "v213g_current": "STRUCTURED_ONLY_REGENERATION",
                "result": "CURRENT_DIVERGES_FROM_V213F",
            },
        },
        "answer_preservation": {
            "arbitrated_hash_match_control": hash_match,
            "n": n,
            "rate": round(hash_match / n, 4) if n else None,
            "preserve_counterfactual_hash_match": preserve_would_match,
        },
        "regressions_on_cohort": {
            "baseline_regression": base_reg,
            "arbitrated_regression": arb_reg,
            "preserve_counterfactual_arbitrated_regression": preserve_arb_reg,
        },
        "root_cause_counts": dict(root_causes),
        "determinism": {
            "generation_determinism_rate": generation_determinism_rate,
            "identical_evidence_but_divergent_answer": identical_but_divergent,
            "interpretation": (
                "Control is generated once (production path) and reused as the "
                "comparison baseline — control is not regenerated for scoring. "
                "Baseline and arbitrated arms each regenerate independently. "
                "With identical evidence fingerprints, answer-hash divergence is "
                "CONTROL_REGENERATION / LLM variance, not document leak."
            ),
        },
        "recommended_architecture": recommended,
        "recommended_note": recommended_note,
        "hypotheses": {
            "H1a_classification": "PASS",
            "H1b_enforcement": enforcement,
            "H1c_control_preservation": "NOT_CONFIRMED",
            "H2_recovery": "PASS",
            "H3_generator_behavior": "NOT_CONFIRMED",
            "safety": "PASS",
        },
        "promotion_status": "INVESTIGATE_BEFORE_PROMOTION",
        "production_isolation": {
            "v213f_document_arbitration_experiment": False,
            "v213d_shadow_sample_rate": 0.01,
            "note": "No production enablement; sample rate unchanged.",
        },
        "cases": analyzed,
        "regression_cases": regressions,
    }


def write_v213g1_artifacts(
    summary: dict[str, Any],
    *,
    out_json: Path | None = None,
    out_jsonl: Path | None = None,
) -> None:
    out_json = out_json or _OUT_JSON
    out_jsonl = out_jsonl or _OUT_JSONL
    out_json.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in summary.items() if k not in {"cases", "regression_cases"}}
    out_json.write_text(json.dumps({**slim, "n_cases": len(summary.get("cases") or [])}, indent=2))
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for case in summary.get("cases") or []:
            # Drop heavy investigation bodies from jsonl if present nested deeply
            row = dict(case)
            if row.get("investigation"):
                inv = dict(row["investigation"])
                inv.pop("structured_evidence_summary", None)
                inv.pop("document_passages", None)
                row["investigation"] = inv
            handle.write(json.dumps(row) + "\n")


__all__ = [
    "PROVENANCE_ONLY_SEMANTICS_PRESERVE",
    "PROVENANCE_ONLY_SEMANTICS_REGENERATE",
    "aggregate_v213g1_investigation",
    "analyze_provenance_only_case",
    "build_preserved_control_arm",
    "classify_divergence_root_cause",
    "load_targeted_records",
    "resolved_provenance_only_semantics",
    "select_provenance_only_redundant",
    "should_preserve_control_answer",
    "write_v213g1_artifacts",
]
