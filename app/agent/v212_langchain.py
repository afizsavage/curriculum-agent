"""V2.12A LangChain vs LangGraph behavioral equivalence experiment (harness-only)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

from langchain_core.runnables import RunnableLambda, RunnableSequence
from langgraph.graph import END, START, StateGraph

from app.agent.evidence_snapshot import evidence_snapshot_hash
from app.agent.graph_routing import route_after_verification
from app.agent.state import CurriculumQAState
from app.agent.v211_metadata_integrity import (
    ADVERSARIAL_FIXTURE_CLASSES,
    ALL_FIXTURES,
    FIXTURE_CLASSES,
    PRIMARY_FIXTURE_CLASSES,
    PipelineVariant,
    _mapper_fixture,
    _prepare_v211_evidence,
    apply_metadata_policy,
    validate_metadata_integrity,
)
from app.agent.v212_contract import (
    ADVERSARIAL_FIXTURES,
    SAFETY_FIXTURES,
    EquivalenceClassification,
    EquivalenceComparison,
    PipelineRunResult,
    StageTimings,
    compare_pipeline_results,
    get_threshold_sweep,
)
from app.agent.v26_experiment import answer_hash, build_claim_classifications
from app.agent.v28_recommendation_mapping import map_recommendation, remap_row_for_threshold
from app.agent.v29_evidence_normalization import NormalizationVariant, normalize_evidence
from app.config import Settings
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.schemas.verification import VerificationRecommendation, VerificationResult

_EXPERIMENT_NAME = "v2.12a_langchain_equivalence"
_ANALYTICAL_THRESHOLD = 0.85
_C4U18_HASH = "be3e342763f1faac"
_FRACTIONS_HASH = "977b259fcfb4b282"
_METADATA_VARIANT = PipelineVariant.C_METADATA_SUPPRESS


class Implementation(str, Enum):
    LANGGRAPH = "langgraph"
    LANGCHAIN = "langchain"


class HarnessState(TypedDict, total=False):
    context: dict[str, Any]


@dataclass
class _PipelineContext:
    fixture_class: str
    run_index: int
    threshold: float
    implementation: Implementation
    c4u18_baseline: list[CurriculumEvidence]
    fractions_baseline: list[CurriculumEvidence]
    verifier: Any
    settings: Settings
    request_id: str | None
    spec: dict[str, Any]
    raw_evidence: list[CurriculumEvidence]
    evidence_source: str
    verify_evidence: list[CurriculumEvidence]
    normalized_evidence: list[CurriculumEvidence]
    normalization_diagnostics: dict[str, Any]
    integrity: Any
    metadata_blocked: bool
    metadata_policy: str
    state: CurriculumQAState
    verifier_result: Any
    mapping: Any
    final_accepted: bool
    final_recommendation: str
    final_route: str
    execution_path: list[str]
    timings: StageTimings
    llm_calls: int = 0
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)
    post_retrieval: bool = False
    mapper_fixture_override: str | None = None
    evaluation_id: str = ""
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)


def v212_experiment_enabled(settings: Settings, qa: CurriculumQAState) -> bool:
    if qa.metadata.get("v212_langchain_replay"):
        return True
    return bool(getattr(settings, "v212_langchain_experiment", False))


def default_implementation(settings: Settings) -> Implementation:
    """Production default remains LangGraph."""
    if getattr(settings, "v212_langchain_experiment", False):
        return Implementation.LANGCHAIN
    return Implementation.LANGGRAPH


def infer_mapper_fixture_class(
    evidence: list[CurriculumEvidence],
    verifier_result: VerificationResult | None,
) -> str:
    """Heuristic mapper fixture for real-retrieval shadow evaluations."""
    from app.agent.v28_recommendation_mapping import detect_placeholder

    if not evidence:
        return "MISSING_EVIDENCE"
    if verifier_result is None:
        return "FAITHFUL_COMPLETE"
    placeholder_detected, _ = detect_placeholder(
        evidence=evidence,
        answer="",
        verifier_result=verifier_result,
    )
    if placeholder_detected:
        return "CLEAN_PLACEHOLDER"
    if verifier_result.unsupported_claims:
        return "UNSUPPORTED_CLAIM"
    if verifier_result.recommendation == VerificationRecommendation.RETRIEVE_MORE:
        return "FAITHFUL_IMPERFECT"
    return "FAITHFUL_COMPLETE"


def _new_context(
    *,
    fixture_class: str,
    run_index: int,
    threshold: float,
    implementation: Implementation,
    c4u18_baseline: list[CurriculumEvidence],
    fractions_baseline: list[CurriculumEvidence],
    verifier: Any,
    settings: Settings,
    request_id: str | None,
) -> _PipelineContext:
    spec = ALL_FIXTURES[fixture_class]
    raw_evidence, evidence_source = _prepare_v211_evidence(
        fixture_class=fixture_class,
        c4u18_baseline=c4u18_baseline,
        fractions_baseline=fractions_baseline,
    )
    return _PipelineContext(
        fixture_class=fixture_class,
        run_index=run_index,
        threshold=threshold,
        implementation=implementation,
        c4u18_baseline=c4u18_baseline,
        fractions_baseline=fractions_baseline,
        verifier=verifier,
        settings=settings,
        request_id=request_id,
        spec=spec,
        raw_evidence=raw_evidence,
        evidence_source=evidence_source,
        verify_evidence=[],
        normalized_evidence=[],
        normalization_diagnostics={},
        integrity=None,
        metadata_blocked=False,
        metadata_policy="",
        state=CurriculumQAState.initial(question=spec["question"]),
        verifier_result=None,
        mapping=None,
        final_accepted=False,
        final_recommendation="",
        final_route="fallback",
        execution_path=[],
        timings=StageTimings(),
        llm_calls=0,
        tool_calls=0,
        errors=[],
    )


def _stage_retrieve(ctx: _PipelineContext) -> _PipelineContext:
    started = time.perf_counter()
    if ctx.post_retrieval:
        ctx.execution_path.append("retrieve_snapshot")
    else:
        ctx.execution_path.append("retrieve")
    ctx.timings.retrieval_ms += (time.perf_counter() - started) * 1000
    return ctx


def _stage_normalize(ctx: _PipelineContext) -> _PipelineContext:
    started = time.perf_counter()
    normalized = normalize_evidence(
        ctx.raw_evidence, NormalizationVariant.STRUCTURAL_NORMALIZATION
    )
    ctx.normalized_evidence = normalized.evidence
    ctx.normalization_diagnostics = normalized.diagnostics.to_dict()
    ctx.execution_path.append("normalize")
    ctx.timings.normalization_ms += (time.perf_counter() - started) * 1000
    return ctx


def _stage_metadata(ctx: _PipelineContext) -> _PipelineContext:
    started = time.perf_counter()
    integrity = validate_metadata_integrity(ctx.normalized_evidence)
    verify_evidence, metadata_blocked, metadata_policy = apply_metadata_policy(
        ctx.normalized_evidence,
        integrity,
        variant=_METADATA_VARIANT,
    )
    ctx.integrity = integrity
    ctx.verify_evidence = verify_evidence
    ctx.metadata_blocked = metadata_blocked
    ctx.metadata_policy = metadata_policy
    ctx.execution_path.append("metadata_integrity")
    ctx.timings.metadata_validation_ms += (time.perf_counter() - started) * 1000
    return ctx


def _stage_generate(ctx: _PipelineContext) -> _PipelineContext:
    started = time.perf_counter()
    ctx.state.evidence = ctx.verify_evidence
    ctx.state.evidence_status = (
        EvidenceStatus.FOUND if ctx.verify_evidence else EvidenceStatus.NOT_FOUND
    )
    if ctx.post_retrieval:
        answer = ctx.spec.get("answer", "")
        ctx.state.final_answer = answer
        ctx.state.draft_answer = answer
        if ctx.retrieval_metadata.get("grade"):
            ctx.state.grade = ctx.retrieval_metadata["grade"]
        if ctx.retrieval_metadata.get("subject"):
            ctx.state.subject = ctx.retrieval_metadata["subject"]
        if ctx.retrieval_metadata.get("topic"):
            ctx.state.topic = ctx.retrieval_metadata["topic"]
    else:
        ctx.state.grade = "CLASS_4"
        ctx.state.topic = "money" if ctx.evidence_source == "c4u18" else "fractions"
        ctx.state.subject = "MATHEMATICS"
        ctx.state.final_answer = ctx.spec["answer"]
        ctx.state.draft_answer = ctx.spec["answer"]
    ctx.state.metadata["v212_langchain_replay"] = True
    ctx.state.metadata["v212_fixture_class"] = ctx.fixture_class
    ctx.state.metadata["v212_implementation"] = ctx.implementation.value
    if ctx.evaluation_id:
        ctx.state.metadata["v212b_evaluation_id"] = ctx.evaluation_id
    ctx.execution_path.append("generate")
    ctx.timings.generation_ms += (time.perf_counter() - started) * 1000
    return ctx


def _stage_verify(ctx: _PipelineContext) -> _PipelineContext:
    started = time.perf_counter()
    ctx.verifier_result = ctx.verifier.verify(ctx.state, request_id=ctx.request_id)
    ctx.state.verification = ctx.verifier_result
    ctx.llm_calls += 1
    ctx.execution_path.append("verify")
    ctx.timings.verification_ms += (time.perf_counter() - started) * 1000
    return ctx


def _stage_map(ctx: _PipelineContext) -> _PipelineContext:
    started = time.perf_counter()
    if ctx.post_retrieval:
        mapper_fixture = ctx.mapper_fixture_override or infer_mapper_fixture_class(
            ctx.verify_evidence,
            ctx.verifier_result,
        )
    else:
        mapper_fixture = ctx.mapper_fixture_override or _mapper_fixture(ctx.fixture_class)
    ctx.mapping = map_recommendation(
        ctx.verifier_result,
        fixture_class=mapper_fixture,  # type: ignore[arg-type]
        evidence=ctx.verify_evidence,
        answer=ctx.spec.get("answer", ""),
        threshold=ctx.threshold,
    )
    if ctx.metadata_blocked:
        ctx.final_accepted = False
        if not ctx.verify_evidence or ctx.verifier_result.missing_evidence:
            ctx.final_recommendation = "insufficient_evidence"
        else:
            ctx.final_recommendation = "reject"
    else:
        ctx.final_accepted = ctx.mapping.mapped_accepted
        ctx.final_recommendation = ctx.mapping.mapped_recommendation.value
    ctx.execution_path.append("map_recommendation")
    ctx.timings.mapping_ms += (time.perf_counter() - started) * 1000
    return ctx


def _stage_route(ctx: _PipelineContext) -> _PipelineContext:
    started = time.perf_counter()
    if ctx.metadata_blocked and not ctx.final_accepted:
        ctx.final_route = "fallback"
    else:
        ctx.final_route = route_after_verification(
            {"qa": ctx.state, "visited_nodes": list(ctx.execution_path)},
            settings=ctx.settings,
        )
    ctx.execution_path.append("route")
    ctx.timings.routing_ms += (time.perf_counter() - started) * 1000
    return ctx


def _context_to_result(ctx: _PipelineContext) -> PipelineRunResult:
    baseline_hash = _C4U18_HASH if ctx.evidence_source == "c4u18" else _FRACTIONS_HASH
    safety_status = "accepted" if ctx.final_accepted else "rejected"
    if ctx.fixture_class in SAFETY_FIXTURES | ADVERSARIAL_FIXTURES and ctx.final_accepted:
        safety_status = "unsafe_accept"

    total = (
        ctx.timings.retrieval_ms
        + ctx.timings.normalization_ms
        + ctx.timings.metadata_validation_ms
        + ctx.timings.generation_ms
        + ctx.timings.verification_ms
        + ctx.timings.mapping_ms
        + ctx.timings.routing_ms
    )
    ctx.timings.total_ms = total

    experiment_name = (
        "v2.12b_production_shadow" if ctx.post_retrieval else _EXPERIMENT_NAME
    )
    generated_answer = ctx.spec.get("answer", "")

    result = PipelineRunResult(
        experiment=experiment_name,
        implementation=ctx.implementation.value,
        fixture_class=ctx.fixture_class,
        run_index=ctx.run_index,
        threshold=ctx.threshold,
        question=ctx.spec.get("question", ""),
        retrieved_evidence_ids=[
            e.entity_id for e in ctx.raw_evidence if e.entity_id
        ],
        raw_evidence_hash=evidence_snapshot_hash(ctx.raw_evidence),
        normalized_evidence_hash=ctx.normalization_diagnostics.get("evidence_hash_out", ""),
        normalization_diagnostics=ctx.normalization_diagnostics,
        metadata_integrity_valid=bool(ctx.integrity and ctx.integrity.valid),
        metadata_violations=[
            v.to_dict() for v in (ctx.integrity.violations if ctx.integrity else [])
        ],
        metadata_blocked=ctx.metadata_blocked,
        generated_answer=generated_answer,
        verifier_score=float(ctx.verifier_result.score if ctx.verifier_result else 0.0),
        verifier_decision=str(
            ctx.verifier_result.recommendation.value if ctx.verifier_result else ""
        ),
        verifier_recommendation=str(
            ctx.verifier_result.recommendation.value if ctx.verifier_result else ""
        ),
        mapped_recommendation=str(
            ctx.mapping.mapped_recommendation.value if ctx.mapping else ""
        ),
        final_accepted=ctx.final_accepted,
        final_route=ctx.final_route,
        safety_status=safety_status,
        errors=list(ctx.errors),
        llm_calls=ctx.llm_calls,
        tool_calls=ctx.tool_calls,
        execution_path=list(ctx.execution_path),
        timings=ctx.timings,
    )

    c4u18_fixtures = {"FAITHFUL_COMPLETE", "NORMALIZATION_ONLY_GROUNDING"}
    if ctx.fixture_class in c4u18_fixtures or (
        ctx.post_retrieval and ctx.evidence_source == "c4u18"
    ):
        lo01 = next(
            (
                r
                for r in (ctx.integrity.resolved_relationships if ctx.integrity else [])
                if r.lo_code == "C4U18-LO01"
            ),
            None,
        )
        result.c4u18_regression = {
            "raw_topic_uuid": lo01.topic_raw if lo01 else None,
            "resolved_topic_name": lo01.topic_resolved if lo01 else None,
            "subject": lo01.subject if lo01 else None,
            "grade": lo01.grade if lo01 else None,
            "parent": lo01.parent_name if lo01 else None,
            "metadata_valid": result.metadata_integrity_valid,
            "metadata_blocked": ctx.metadata_blocked,
            "verifier_score": result.verifier_score,
            "verifier_recommendation": result.verifier_recommendation,
            "mapper_result": result.mapped_recommendation,
            "final_route": result.final_route,
            "final_accepted": result.final_accepted,
            "implementation": ctx.implementation.value,
        }

    return result


def _run_sequential_pipeline(
    ctx: _PipelineContext,
    *,
    orchestration_label: str,
) -> PipelineRunResult:
    ctx.execution_path.append(orchestration_label)
    for stage in (
        _stage_retrieve,
        _stage_normalize,
        _stage_metadata,
        _stage_generate,
        _stage_verify,
        _stage_map,
        _stage_route,
    ):
        ctx = stage(ctx)
    return _context_to_result(ctx)


def _harness_state_to_context(state: HarnessState) -> _PipelineContext:
    return state["context"]  # type: ignore[return-value]


def _context_to_harness_state(ctx: _PipelineContext) -> HarnessState:
    return {"context": ctx}


def _langgraph_node(stage_fn):
    def _node(state: HarnessState) -> HarnessState:
        ctx = _harness_state_to_context(state)
        updated = stage_fn(ctx)
        return _context_to_harness_state(updated)

    return _node


def build_langgraph_harness_graph():
    """Mini LangGraph orchestrating the validated V2.11 pipeline stages."""
    builder: StateGraph = StateGraph(HarnessState)
    builder.add_node("retrieve", _langgraph_node(_stage_retrieve))
    builder.add_node("normalize", _langgraph_node(_stage_normalize))
    builder.add_node("metadata_integrity", _langgraph_node(_stage_metadata))
    builder.add_node("generate", _langgraph_node(_stage_generate))
    builder.add_node("verify", _langgraph_node(_stage_verify))
    builder.add_node("map_recommendation", _langgraph_node(_stage_map))
    builder.add_node("route", _langgraph_node(_stage_route))

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "normalize")
    builder.add_edge("normalize", "metadata_integrity")
    builder.add_edge("metadata_integrity", "generate")
    builder.add_edge("generate", "verify")
    builder.add_edge("verify", "map_recommendation")
    builder.add_edge("map_recommendation", "route")
    builder.add_edge("route", END)
    return builder.compile()


def build_langchain_harness_chain() -> RunnableSequence:
    """LangChain Runnable orchestrating the same validated pipeline stages."""

    def _run_stage(stage_fn, payload: _PipelineContext) -> _PipelineContext:
        return stage_fn(payload)

    chain: RunnableSequence = RunnableLambda(
        lambda ctx: _run_stage(_stage_retrieve, ctx)
    )
    for stage in (
        _stage_normalize,
        _stage_metadata,
        _stage_generate,
        _stage_verify,
        _stage_map,
        _stage_route,
    ):
        chain = chain | RunnableLambda(lambda ctx, stage=stage: _run_stage(stage, ctx))
    chain = chain | RunnableLambda(_context_to_result)
    return chain


_LANGGRAPH_HARNESS = None
_LANGCHAIN_HARNESS = None


def _get_langgraph_harness():
    global _LANGGRAPH_HARNESS
    if _LANGGRAPH_HARNESS is None:
        _LANGGRAPH_HARNESS = build_langgraph_harness_graph()
    return _LANGGRAPH_HARNESS


def _get_langchain_harness():
    global _LANGCHAIN_HARNESS
    if _LANGCHAIN_HARNESS is None:
        _LANGCHAIN_HARNESS = build_langchain_harness_chain()
    return _LANGCHAIN_HARNESS


def run_implementation(
    *,
    implementation: Implementation,
    fixture_class: str,
    c4u18_baseline: list[CurriculumEvidence],
    fractions_baseline: list[CurriculumEvidence],
    verifier: Any,
    settings: Settings | None = None,
    threshold: float = _ANALYTICAL_THRESHOLD,
    run_index: int = 1,
    request_id: str | None = None,
) -> PipelineRunResult:
    """Run one fixture through the selected orchestration implementation."""
    ctx = _new_context(
        fixture_class=fixture_class,
        run_index=run_index,
        threshold=threshold,
        implementation=implementation,
        c4u18_baseline=c4u18_baseline,
        fractions_baseline=fractions_baseline,
        verifier=verifier,
        settings=settings or Settings(),
        request_id=request_id,
    )

    if implementation == Implementation.LANGGRAPH:
        graph = _get_langgraph_harness()
        ctx.execution_path.append("langgraph_orchestration")
        final_state = graph.invoke(_context_to_harness_state(ctx))
        ctx = _harness_state_to_context(final_state)
        return _context_to_result(ctx)

    chain = _get_langchain_harness()
    ctx.execution_path.append("langchain_orchestration")
    result = chain.invoke(ctx)
    if isinstance(result, PipelineRunResult):
        return result
    return _context_to_result(result)


def run_equivalence_pair(
    *,
    fixture_class: str,
    c4u18_baseline: list[CurriculumEvidence],
    fractions_baseline: list[CurriculumEvidence],
    verifier: Any,
    settings: Settings | None = None,
    threshold: float = _ANALYTICAL_THRESHOLD,
    run_index: int = 1,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Run LangGraph control and LangChain experiment on the same fixture input."""
    tag = request_id or f"{fixture_class.lower()}_{run_index:02d}"
    control = run_implementation(
        implementation=Implementation.LANGGRAPH,
        fixture_class=fixture_class,
        c4u18_baseline=c4u18_baseline,
        fractions_baseline=fractions_baseline,
        verifier=verifier,
        settings=settings,
        threshold=threshold,
        run_index=run_index,
        request_id=f"lg_{tag}",
    )
    experiment = run_implementation(
        implementation=Implementation.LANGCHAIN,
        fixture_class=fixture_class,
        c4u18_baseline=copy.deepcopy(c4u18_baseline),
        fractions_baseline=copy.deepcopy(fractions_baseline),
        verifier=verifier,
        settings=settings,
        threshold=threshold,
        run_index=run_index,
        request_id=f"lc_{tag}",
    )
    comparison = compare_pipeline_results(
        control,
        experiment,
        fixture_class=fixture_class,
        run_index=run_index,
    )
    return {
        "experiment": _EXPERIMENT_NAME,
        "fixture_class": fixture_class,
        "run_index": run_index,
        "threshold": threshold,
        "langgraph": control.to_dict(),
        "langchain": experiment.to_dict(),
        "comparison": comparison.to_dict(),
    }


def summarize_implementation_rows(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    impl_rows = [r[key] for r in rows if key in r]
    n = len(impl_rows) or 1

    def rate(fixture: str) -> float:
        subset = [r for r in impl_rows if r.get("fixture_class") == fixture]
        if not subset:
            return 0.0
        return round(sum(1 for r in subset if r.get("final_accepted")) / len(subset), 3)

    safety_false = sum(
        1 for r in impl_rows if r.get("fixture_class") in SAFETY_FIXTURES and r.get("final_accepted")
    )
    adv_false = sum(
        1
        for r in impl_rows
        if r.get("fixture_class") in ADVERSARIAL_FIXTURES and r.get("final_accepted")
    )
    placeholder_false = sum(
        1
        for r in impl_rows
        if r.get("fixture_class") in {"CLEAN_PLACEHOLDER", "PLACEHOLDER_PLUS_HIGH_SCORE"}
        and r.get("final_accepted")
    )

    latencies = [r.get("timings", {}).get("total_ms", 0.0) for r in impl_rows]

    def percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
        return round(ordered[idx], 3)

    return {
        "n": len(impl_rows),
        "faithful_complete_acceptance": rate("FAITHFUL_COMPLETE"),
        "faithful_imperfect_acceptance": rate("FAITHFUL_IMPERFECT"),
        "placeholder_false_acceptance": placeholder_false,
        "safety_false_acceptance": safety_false,
        "adversarial_false_acceptance": adv_false,
        "mean_latency_ms": round(sum(latencies) / n, 3),
        "median_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "llm_calls": sum(r.get("llm_calls", 0) for r in impl_rows),
        "tool_calls": sum(r.get("tool_calls", 0) for r in impl_rows),
        "errors": sum(len(r.get("errors") or []) for r in impl_rows),
        "timeouts": sum(r.get("timeouts", 0) for r in impl_rows),
    }


def build_threshold_sweep(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    impl_rows = [r[key] for r in rows if key in r]
    for threshold in get_threshold_sweep():
        remapped = []
        for row in impl_rows:
            remapped_row = remap_row_for_threshold(
                {
                    "fixture_class": row["fixture_class"],
                    "verifier_score": row.get("verifier_score", 0.0),
                    "retrieve_more_requested": row.get("verifier_recommendation") == "retrieve_more",
                    "verifier_accepted": row.get("verifier_recommendation") == "accept",
                    "placeholder_detected": row.get("fixture_class")
                    in {"CLEAN_PLACEHOLDER", "PLACEHOLDER_PLUS_HIGH_SCORE"},
                    "unsupported_claims": False,
                },
                threshold,
            )
            if row.get("metadata_blocked"):
                remapped_row["mapped_accepted"] = False
                remapped_row["mapped_recommendation"] = (
                    "insufficient_evidence"
                    if not row.get("retrieved_evidence_ids")
                    else "reject"
                )
            remapped.append(remapped_row)
        n = len(remapped) or 1

        def rate(fixture: str) -> float:
            subset = [r for r in remapped if r.get("fixture_class") == fixture]
            if not subset:
                return 0.0
            return round(sum(1 for r in subset if r.get("mapped_accepted")) / len(subset), 3)

        out.append(
            {
                "implementation": key,
                "threshold": threshold,
                "faithful_complete_acceptance": rate("FAITHFUL_COMPLETE"),
                "faithful_imperfect_acceptance": rate("FAITHFUL_IMPERFECT"),
                "placeholder_acceptance": rate("CLEAN_PLACEHOLDER"),
                "safety_false_acceptance": sum(
                    1
                    for r in remapped
                    if r.get("fixture_class") in SAFETY_FIXTURES and r.get("mapped_accepted")
                ),
                "adversarial_false_acceptance": sum(
                    1
                    for r in remapped
                    if r.get("fixture_class") in ADVERSARIAL_FIXTURES and r.get("mapped_accepted")
                ),
                "overall_acceptance": round(
                    sum(1 for r in remapped if r.get("mapped_accepted")) / n, 3
                ),
            }
        )
    return out


def summarize_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = [r["comparison"] for r in rows if r.get("comparison")]
    counts: dict[str, int] = {}
    for comp in comparisons:
        label = comp.get("classification", "unknown")
        counts[label] = counts.get(label, 0) + 1
    unsafe = counts.get(EquivalenceClassification.UNSAFE_DIVERGENCE.value, 0)
    return {
        "total_comparisons": len(comparisons),
        "classification_counts": counts,
        "unsafe_divergence_count": unsafe,
    }


def interpret_v212(
    *,
    langgraph_summary: dict[str, Any],
    langchain_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
) -> tuple[str, str, str, str]:
    unsafe = comparison_summary.get("unsafe_divergence_count", 999)
    lg_fc = langgraph_summary.get("faithful_complete_acceptance", 0.0)
    lc_fc = langchain_summary.get("faithful_complete_acceptance", 0.0)
    lg_adv = langgraph_summary.get("adversarial_false_acceptance", 999)
    lc_adv = langchain_summary.get("adversarial_false_acceptance", 999)
    lg_safety = langgraph_summary.get("safety_false_acceptance", 999)
    lc_safety = langchain_summary.get("safety_false_acceptance", 999)

    arch_answer = (
        "Not yet — LangChain orchestration requires additional hardening before "
        "replacing LangGraph in production."
    )

    if (
        unsafe == 0
        and lg_fc >= 1.0
        and lc_fc >= 1.0
        and lg_adv == 0
        and lc_adv == 0
        and lg_safety == 0
        and lc_safety == 0
        and langgraph_summary.get("placeholder_false_acceptance", 999) == 0
        and langchain_summary.get("placeholder_false_acceptance", 999) == 0
    ):
        conclusion = "SUPPORTED"
        note = (
            "LangChain reproduced validated pipeline behavior with zero unsafe divergences "
            "and preserved C4-U18 FC, safety, and metadata invariants."
        )
        v213 = "V2.12B — Real Retrieval Production-Shadow Evaluation"
        arch_answer = (
            "Yes — LangChain is a safe orchestration replacement candidate. "
            "Proceed to production-shadow evaluation while keeping LangGraph as control."
        )
    elif unsafe == 0 and lg_fc >= 1.0 and lc_fc >= 1.0:
        conclusion = "PARTIALLY_SUPPORTED"
        note = (
            "No unsafe divergences and FC preserved, but FI variance or minor metric "
            "differences require targeted hardening before migration."
        )
        v213 = "targeted migration-hardening experiment"
    else:
        conclusion = "NOT_SUPPORTED"
        note = (
            "LangChain diverged from LangGraph on safety-critical behavior or FC grounding."
        )
        v213 = "keep LangGraph; diagnose framework abstraction mismatch"

    return conclusion, note, v213, arch_answer


def _new_post_retrieval_context(
    *,
    question: str,
    raw_evidence: list[CurriculumEvidence],
    generated_answer: str,
    implementation: Implementation,
    verifier: Any,
    settings: Settings,
    threshold: float = _ANALYTICAL_THRESHOLD,
    run_index: int = 1,
    request_id: str | None = None,
    category: str = "REAL_SHADOW",
    evaluation_id: str = "",
    retrieval_metadata: dict[str, Any] | None = None,
    evidence_source: str = "real",
    mapper_fixture_override: str | None = None,
) -> _PipelineContext:
    return _PipelineContext(
        fixture_class=category,
        run_index=run_index,
        threshold=threshold,
        implementation=implementation,
        c4u18_baseline=[],
        fractions_baseline=[],
        verifier=verifier,
        settings=settings,
        request_id=request_id,
        spec={"question": question, "answer": generated_answer},
        raw_evidence=copy.deepcopy(raw_evidence),
        evidence_source=evidence_source,
        verify_evidence=[],
        normalized_evidence=[],
        normalization_diagnostics={},
        integrity=None,
        metadata_blocked=False,
        metadata_policy="",
        state=CurriculumQAState.initial(question=question),
        verifier_result=None,
        mapping=None,
        final_accepted=False,
        final_recommendation="",
        final_route="fallback",
        execution_path=[],
        timings=StageTimings(),
        llm_calls=0,
        tool_calls=0,
        errors=[],
        post_retrieval=True,
        mapper_fixture_override=mapper_fixture_override,
        evaluation_id=evaluation_id,
        retrieval_metadata=dict(retrieval_metadata or {}),
    )


def run_post_retrieval_implementation(
    *,
    implementation: Implementation,
    question: str,
    raw_evidence: list[CurriculumEvidence],
    generated_answer: str,
    verifier: Any,
    settings: Settings | None = None,
    threshold: float = _ANALYTICAL_THRESHOLD,
    run_index: int = 1,
    request_id: str | None = None,
    category: str = "REAL_SHADOW",
    evaluation_id: str = "",
    retrieval_metadata: dict[str, Any] | None = None,
    evidence_source: str = "real",
    mapper_fixture_override: str | None = None,
) -> PipelineRunResult:
    """Run post-retrieval validated pipeline stages on a frozen evidence snapshot."""
    ctx = _new_post_retrieval_context(
        question=question,
        raw_evidence=raw_evidence,
        generated_answer=generated_answer,
        implementation=implementation,
        verifier=verifier,
        settings=settings or Settings(),
        threshold=threshold,
        run_index=run_index,
        request_id=request_id,
        category=category,
        evaluation_id=evaluation_id,
        retrieval_metadata=retrieval_metadata,
        evidence_source=evidence_source,
        mapper_fixture_override=mapper_fixture_override,
    )

    if implementation == Implementation.LANGGRAPH:
        graph = _get_langgraph_harness()
        ctx.execution_path.append("langgraph_orchestration")
        final_state = graph.invoke(_context_to_harness_state(ctx))
        ctx = _harness_state_to_context(final_state)
        return _context_to_result(ctx)

    chain = _get_langchain_harness()
    ctx.execution_path.append("langchain_orchestration")
    result = chain.invoke(ctx)
    if isinstance(result, PipelineRunResult):
        return result
    return _context_to_result(result)


def run_post_retrieval_pair(
    *,
    question: str,
    raw_evidence: list[CurriculumEvidence],
    generated_answer: str,
    verifier: Any,
    settings: Settings | None = None,
    threshold: float = _ANALYTICAL_THRESHOLD,
    run_index: int = 1,
    request_id: str | None = None,
    category: str = "REAL_SHADOW",
    evaluation_id: str = "",
    retrieval_metadata: dict[str, Any] | None = None,
    evidence_source: str = "real",
) -> dict[str, Any]:
    """Run LangGraph control and LangChain experiment on the same evidence snapshot."""
    evidence_hash = evidence_snapshot_hash(raw_evidence)
    tag = request_id or evaluation_id or f"shadow_{run_index:02d}"
    control = run_post_retrieval_implementation(
        implementation=Implementation.LANGGRAPH,
        question=question,
        raw_evidence=raw_evidence,
        generated_answer=generated_answer,
        verifier=verifier,
        settings=settings,
        threshold=threshold,
        run_index=run_index,
        request_id=f"lg_{tag}",
        category=category,
        evaluation_id=evaluation_id,
        retrieval_metadata=retrieval_metadata,
        evidence_source=evidence_source,
    )
    experiment = run_post_retrieval_implementation(
        implementation=Implementation.LANGCHAIN,
        question=question,
        raw_evidence=copy.deepcopy(raw_evidence),
        generated_answer=generated_answer,
        verifier=verifier,
        settings=settings,
        threshold=threshold,
        run_index=run_index,
        request_id=f"lc_{tag}",
        category=category,
        evaluation_id=evaluation_id,
        retrieval_metadata=retrieval_metadata,
        evidence_source=evidence_source,
    )
    comparison = compare_pipeline_results(
        control,
        experiment,
        fixture_class=category,
        run_index=run_index,
        expected_evidence_hash=evidence_hash,
    )
    return {
        "experiment": "v2.12b_production_shadow",
        "evaluation_id": evaluation_id,
        "category": category,
        "run_index": run_index,
        "threshold": threshold,
        "evidence_hash": evidence_hash,
        "langgraph": control.to_dict(),
        "langchain": experiment.to_dict(),
        "comparison": comparison.to_dict(),
    }


__all__ = [
    "ADVERSARIAL_FIXTURE_CLASSES",
    "FIXTURE_CLASSES",
    "PRIMARY_FIXTURE_CLASSES",
    "Implementation",
    "build_langchain_harness_chain",
    "build_langgraph_harness_graph",
    "default_implementation",
    "infer_mapper_fixture_class",
    "interpret_v212",
    "run_equivalence_pair",
    "run_implementation",
    "run_post_retrieval_implementation",
    "run_post_retrieval_pair",
    "summarize_comparisons",
    "summarize_implementation_rows",
    "v212_experiment_enabled",
]
