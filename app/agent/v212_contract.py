"""V2.12A framework-independent pipeline contract and equivalence comparator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from app.agent.v28_recommendation_mapping import threshold_sweep

_EXPERIMENT_NAME = "v2.12a_langchain_equivalence"
_ANALYTICAL_THRESHOLD = 0.85

SafetyFixture = Literal[
    "UNSUPPORTED_CLAIM",
    "UNSUPPORTED_ABSENCE",
    "SPECULATIVE",
    "RECONSTRUCTION",
    "MISSING_EVIDENCE",
    "HIGH_SCORE_UNSUPPORTED",
    "CLEAN_PLACEHOLDER",
    "PLACEHOLDER_PLUS_HIGH_SCORE",
]

SAFETY_FIXTURES: frozenset[str] = frozenset(
    {
        "UNSUPPORTED_CLAIM",
        "UNSUPPORTED_ABSENCE",
        "SPECULATIVE",
        "RECONSTRUCTION",
        "MISSING_EVIDENCE",
        "HIGH_SCORE_UNSUPPORTED",
        "CLEAN_PLACEHOLDER",
        "PLACEHOLDER_PLUS_HIGH_SCORE",
        "ADV_HIGH_SCORE_SAFETY",
        "ADV_HIGH_SCORE_PLACEHOLDER",
    }
)

ADVERSARIAL_FIXTURES: frozenset[str] = frozenset(
    {
        "ADV_FAKE_PARENT",
        "ADV_CONFLICTING_PARENT",
        "ADV_PLACEHOLDER_PARENT",
        "ADV_WRONG_SUBJECT",
        "ADV_WRONG_GRADE",
        "ADV_HIGH_SCORE_SAFETY",
        "ADV_HIGH_SCORE_PLACEHOLDER",
        "ADV_MISSING_AFTER_NORM",
        "ADV_CONFLICTING_SUBJECT",
        "ADV_CONFLICTING_GRADE",
        "ADV_TOPIC_UUID_COLLISION",
        "ADV_PARENT_CHILD_MISMATCH",
        "ADV_SUBJECT_TOPIC_MISMATCH",
        "ADV_GRADE_TOPIC_MISMATCH",
        "ADV_PLACEHOLDER_TOPIC",
        "ADV_PLACEHOLDER_PARENT_SUBSTANTIVE_CHILD",
    }
)


class EquivalenceClassification(str, Enum):
    EXACT_EQUIVALENCE = "EXACT_EQUIVALENCE"
    BEHAVIORAL_EQUIVALENCE = "BEHAVIORAL_EQUIVALENCE"
    EXPECTED_LLM_VARIANCE = "EXPECTED_LLM_VARIANCE"
    CONTROLLED_DIFFERENCE = "CONTROLLED_DIFFERENCE"
    RETRIEVAL_VARIANCE = "RETRIEVAL_VARIANCE"
    REGRESSION = "REGRESSION"
    UNSAFE_DIVERGENCE = "UNSAFE_DIVERGENCE"


class FailureOrigin(str, Enum):
    RETRIEVAL = "RETRIEVAL"
    NORMALIZATION = "NORMALIZATION"
    METADATA_GUARD = "METADATA_GUARD"
    GENERATION = "GENERATION"
    VERIFIER = "VERIFIER"
    MAPPER = "MAPPER"
    ROUTING = "ROUTING"
    LANGCHAIN_ORCHESTRATION = "LANGCHAIN_ORCHESTRATION"
    LANGGRAPH_ORCHESTRATION = "LANGGRAPH_ORCHESTRATION"
    UNKNOWN = "UNKNOWN"


@dataclass
class StageTimings:
    total_ms: float = 0.0
    retrieval_ms: float = 0.0
    normalization_ms: float = 0.0
    metadata_validation_ms: float = 0.0
    generation_ms: float = 0.0
    verification_ms: float = 0.0
    mapping_ms: float = 0.0
    routing_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "total_ms": round(self.total_ms, 3),
            "retrieval_ms": round(self.retrieval_ms, 3),
            "normalization_ms": round(self.normalization_ms, 3),
            "metadata_validation_ms": round(self.metadata_validation_ms, 3),
            "generation_ms": round(self.generation_ms, 3),
            "verification_ms": round(self.verification_ms, 3),
            "mapping_ms": round(self.mapping_ms, 3),
            "routing_ms": round(self.routing_ms, 3),
        }


@dataclass
class PipelineRunResult:
    """Framework-neutral structured pipeline result for equivalence comparison."""

    experiment: str = _EXPERIMENT_NAME
    implementation: str = ""
    fixture_class: str = ""
    run_index: int = 0
    threshold: float = _ANALYTICAL_THRESHOLD
    question: str = ""
    retrieved_evidence_ids: list[str] = field(default_factory=list)
    raw_evidence_hash: str = ""
    normalized_evidence_hash: str = ""
    normalization_diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata_integrity_valid: bool = True
    metadata_violations: list[dict[str, Any]] = field(default_factory=list)
    metadata_blocked: bool = False
    generated_answer: str = ""
    verifier_score: float = 0.0
    verifier_decision: str = ""
    verifier_recommendation: str = ""
    mapped_recommendation: str = ""
    final_accepted: bool = False
    final_route: str = ""
    safety_status: str = ""
    errors: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    retries: int = 0
    timeouts: int = 0
    execution_path: list[str] = field(default_factory=list)
    timings: StageTimings = field(default_factory=StageTimings)
    c4u18_regression: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "implementation": self.implementation,
            "fixture_class": self.fixture_class,
            "run_index": self.run_index,
            "threshold": self.threshold,
            "question": self.question,
            "retrieved_evidence_ids": list(self.retrieved_evidence_ids),
            "raw_evidence_hash": self.raw_evidence_hash,
            "normalized_evidence_hash": self.normalized_evidence_hash,
            "normalization_diagnostics": self.normalization_diagnostics,
            "metadata_integrity_valid": self.metadata_integrity_valid,
            "metadata_violations": list(self.metadata_violations),
            "metadata_blocked": self.metadata_blocked,
            "generated_answer": self.generated_answer,
            "verifier_score": self.verifier_score,
            "verifier_decision": self.verifier_decision,
            "verifier_recommendation": self.verifier_recommendation,
            "mapped_recommendation": self.mapped_recommendation,
            "final_accepted": self.final_accepted,
            "final_route": self.final_route,
            "safety_status": self.safety_status,
            "errors": list(self.errors),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "retries": self.retries,
            "timeouts": self.timeouts,
            "execution_path": list(self.execution_path),
            "timings": self.timings.to_dict(),
            "c4u18_regression": self.c4u18_regression,
        }


@dataclass(frozen=True)
class EquivalenceComparison:
    fixture_class: str
    run_index: int
    classification: EquivalenceClassification
    control_implementation: str
    experiment_implementation: str
    control_accepted: bool
    experiment_accepted: bool
    divergences: tuple[str, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_class": self.fixture_class,
            "run_index": self.run_index,
            "classification": self.classification.value,
            "control_implementation": self.control_implementation,
            "experiment_implementation": self.experiment_implementation,
            "control_accepted": self.control_accepted,
            "experiment_accepted": self.experiment_accepted,
            "divergences": list(self.divergences),
            "notes": self.notes,
        }


def get_threshold_sweep() -> tuple[float, ...]:
    return threshold_sweep()


def _is_safety_critical(fixture_class: str) -> bool:
    return fixture_class in SAFETY_FIXTURES or fixture_class in ADVERSARIAL_FIXTURES


def compare_pipeline_results(
    control: PipelineRunResult,
    experiment: PipelineRunResult,
    *,
    fixture_class: str,
    run_index: int = 0,
    expected_evidence_hash: str | None = None,
) -> EquivalenceComparison:
    """Classify behavioral equivalence between control and experiment runs."""
    divergences: list[str] = []

    if (
        expected_evidence_hash
        and control.raw_evidence_hash
        and control.raw_evidence_hash != expected_evidence_hash
    ):
        return EquivalenceComparison(
            fixture_class=fixture_class,
            run_index=run_index,
            classification=EquivalenceClassification.RETRIEVAL_VARIANCE,
            control_implementation=control.implementation,
            experiment_implementation=experiment.implementation,
            control_accepted=control.final_accepted,
            experiment_accepted=experiment.final_accepted,
            divergences=(f"evidence_hash: {control.raw_evidence_hash} != {expected_evidence_hash}",),
            notes="Evidence snapshot differs from expected retrieval hash.",
        )

    if control.raw_evidence_hash != experiment.raw_evidence_hash:
        return EquivalenceComparison(
            fixture_class=fixture_class,
            run_index=run_index,
            classification=EquivalenceClassification.RETRIEVAL_VARIANCE,
            control_implementation=control.implementation,
            experiment_implementation=experiment.implementation,
            control_accepted=control.final_accepted,
            experiment_accepted=experiment.final_accepted,
            divergences=(
                f"raw_evidence_hash: {control.raw_evidence_hash} != {experiment.raw_evidence_hash}",
            ),
            notes="Implementations consumed different evidence snapshots.",
        )

    deterministic_fields = (
        ("raw_evidence_hash", control.raw_evidence_hash, experiment.raw_evidence_hash),
        (
            "normalized_evidence_hash",
            control.normalized_evidence_hash,
            experiment.normalized_evidence_hash,
        ),
        (
            "metadata_integrity_valid",
            control.metadata_integrity_valid,
            experiment.metadata_integrity_valid,
        ),
        ("metadata_blocked", control.metadata_blocked, experiment.metadata_blocked),
        ("mapped_recommendation", control.mapped_recommendation, experiment.mapped_recommendation),
    )
    for name, left, right in deterministic_fields:
        if left != right:
            divergences.append(f"{name}: {left!r} != {right!r}")

    violation_types_control = {v.get("violation_type") for v in control.metadata_violations}
    violation_types_experiment = {v.get("violation_type") for v in experiment.metadata_violations}
    if violation_types_control != violation_types_experiment:
        divergences.append(
            f"metadata_violation_types: {sorted(violation_types_control)} != "
            f"{sorted(violation_types_experiment)}"
        )

    if _is_safety_critical(fixture_class):
        if not control.final_accepted and experiment.final_accepted:
            return EquivalenceComparison(
                fixture_class=fixture_class,
                run_index=run_index,
                classification=EquivalenceClassification.UNSAFE_DIVERGENCE,
                control_implementation=control.implementation,
                experiment_implementation=experiment.implementation,
                control_accepted=control.final_accepted,
                experiment_accepted=experiment.final_accepted,
                divergences=tuple(divergences),
                notes="Experiment accepted a safety-critical fixture that control rejected.",
            )

    if control.final_accepted and not experiment.final_accepted:
        if fixture_class in {"FAITHFUL_COMPLETE", "FAITHFUL_IMPERFECT", "NORMALIZATION_ONLY_GROUNDING"}:
            return EquivalenceComparison(
                fixture_class=fixture_class,
                run_index=run_index,
                classification=EquivalenceClassification.REGRESSION,
                control_implementation=control.implementation,
                experiment_implementation=experiment.implementation,
                control_accepted=control.final_accepted,
                experiment_accepted=experiment.final_accepted,
                divergences=tuple(divergences),
                notes="Experiment rejected a positive-control fixture that control accepted.",
            )

    exact_match = (
        not divergences
        and control.final_accepted == experiment.final_accepted
        and control.final_route == experiment.final_route
        and control.verifier_recommendation == experiment.verifier_recommendation
        and control.verifier_score == experiment.verifier_score
    )
    if exact_match:
        classification = EquivalenceClassification.EXACT_EQUIVALENCE
    elif not divergences and control.final_accepted == experiment.final_accepted:
        score_delta = abs(control.verifier_score - experiment.verifier_score)
        if score_delta <= 0.05 and control.verifier_recommendation == experiment.verifier_recommendation:
            classification = EquivalenceClassification.EXPECTED_LLM_VARIANCE
        else:
            classification = EquivalenceClassification.BEHAVIORAL_EQUIVALENCE
    elif not divergences:
        classification = EquivalenceClassification.BEHAVIORAL_EQUIVALENCE
    elif control.final_accepted == experiment.final_accepted:
        classification = EquivalenceClassification.CONTROLLED_DIFFERENCE
    else:
        classification = EquivalenceClassification.EXPECTED_LLM_VARIANCE

    return EquivalenceComparison(
        fixture_class=fixture_class,
        run_index=run_index,
        classification=classification,
        control_implementation=control.implementation,
        experiment_implementation=experiment.implementation,
        control_accepted=control.final_accepted,
        experiment_accepted=experiment.final_accepted,
        divergences=tuple(divergences),
    )
