"""Replay the Class 4 Science complete-list verifier input.

Does not change prompts, model settings, or routing. Writes a gitignored
report under data/traces/.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.absence_evidence import needs_absence_verification_context
from app.agent.answer_generator import format_evidence_for_prompt
from app.agent.state import CurriculumQAState
from app.agent.trace import start_agent_run
from app.agent.verifier import AnswerVerifier
from app.agent.verifier_diagnostics import fingerprint
from app.config import get_settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.evidence import CurriculumEvidence, EvidenceStatus
from app.curriculum.normalize import evidence_from_hit, iter_content_nodes, node_to_search_hit
from app.llm.provider import build_llm_provider

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "data" / "traces"
RUN_12E = "run-d29d74ed01ac4a92"
RUN_12F = "run-de522020339f4a61"
QUESTION = "List every published Class 4 Science unit and theme, with its content code."
GRADE_CURRICULUM_ID = "8ed57ab3-e539-453c-93a9-e1782f084599"


def _load(run_id: str) -> dict:
    return json.loads((TRACE_DIR / f"{run_id}.json").read_text())


def _draft(trace: dict) -> str:
    for event in trace["events"]:
        if event.get("event") == "agent.verification.start":
            return event.get("answer") or ""
    return ""


def _generation_ids(trace: dict) -> list[str]:
    for event in trace["events"]:
        if event.get("event") == "agent.evidence.rank":
            return list(event.get("generation_evidence_ids") or [])
    return []


def _new_evidence_ids(trace: dict) -> list[str]:
    found: list[str] = []
    for event in trace["events"]:
        if event.get("event") != "agent.evidence.add" or event.get("duplicate"):
            continue
        item_id = event.get("id")
        if item_id and item_id not in found:
            found.append(item_id)
    return found


def _verification(trace: dict) -> dict:
    for event in trace["events"]:
        if event.get("event") == "agent.verification.end":
            return {
                "passed": event.get("passed"),
                "score": event.get("score"),
                "recommendation": event.get("recommendation"),
                "issues": event.get("issues"),
                "unsupported_claims": event.get("unsupported_claims"),
                "incorrect_claims": event.get("incorrect_claims"),
                "missing_evidence": event.get("missing_evidence"),
            }
    return {}


def _model(trace: dict) -> str | None:
    for event in trace["events"]:
        if event.get("event") == "agent.llm.end" and event.get("kind") == "verification":
            return event.get("model")
    return None


def _records_for(ids: list[str]) -> list[CurriculumEvidence]:
    client = CurriculumAPIClient(settings=get_settings())
    tree = client.get_grade_curriculum_content(GRADE_CURRICULUM_ID)
    by_id: dict[str, CurriculumEvidence] = {}
    for node, parent_id in iter_content_nodes(tree):
        hit = node_to_search_hit(
            node,
            parent_id=parent_id,
            grade="CLASS_4",
            subject="SCIENCE",
            level="primary",
        )
        if not hit.id:
            continue
        item = evidence_from_hit(hit, source_reference="grade_curriculum.content")
        by_id[hit.id] = item
    missing = [item_id for item_id in ids if item_id not in by_id]
    if missing:
        raise SystemExit(f"saved evidence ids missing from the grade curriculum: {missing}")
    return [by_id[item_id] for item_id in ids]


def _without_outcomes(records: list[CurriculumEvidence]) -> list[CurriculumEvidence]:
    stripped: list[CurriculumEvidence] = []
    for item in records:
        metadata = dict(item.metadata or {})
        metadata.pop("learning_outcomes", None)
        stripped.append(item.model_copy(update={"metadata": metadata}))
    return stripped


def _state(draft: str, records: list[CurriculumEvidence]) -> CurriculumQAState:
    qa = CurriculumQAState.initial(question=QUESTION)
    qa.grade = "CLASS_4"
    qa.subject = "SCIENCE"
    qa.level = "primary"
    qa.draft_answer = draft
    qa.evidence = records
    qa.evidence_status = EvidenceStatus.FOUND
    return qa


def main() -> None:
    settings = get_settings()
    older = _load(RUN_12E)
    newer = _load(RUN_12F)
    draft_12e = _draft(older)
    draft_12f = _draft(newer)
    ids_12e = _new_evidence_ids(older)
    ids_12f = _new_evidence_ids(newer)
    records = _records_for(ids_12f)
    block_with = format_evidence_for_prompt(records, question=QUESTION)
    block_without = format_evidence_for_prompt(_without_outcomes(records), question=QUESTION)
    state = _state(draft_12f, records)
    verifier = AnswerVerifier(build_llm_provider(settings), settings=settings)
    messages = verifier.build_messages(state)
    prompt = "\n\n".join(message.content for message in messages)

    report: dict = {
        "historical_raw_payload_recoverable": False,
        "historical_prompt_stored": False,
        "inputs": {
            "drafts_equal": draft_12e == draft_12f,
            "draft_sha256": fingerprint(draft_12f),
            "new_evidence_ids_equal": ids_12e == ids_12f,
            "new_evidence_count": len(ids_12f),
            "generation_ids_equal": _generation_ids(older) == _generation_ids(newer),
            "generation_count": len(_generation_ids(newer)),
            "model_12e": _model(older),
            "model_12f": _model(newer),
            "temperature_in_verifier_call": 0.0,
            "absence_context": needs_absence_verification_context(QUESTION, draft_12f),
            "evidence_block_sha256": fingerprint(block_with),
            "evidence_block_unchanged_without_outcome_metadata": block_with == block_without,
            "reconstructed_prompt_sha256": fingerprint(prompt),
            "answer_evidence_refs_not_stored": True,
        },
        "historical_verification": {
            "12e": _verification(older),
            "12f": _verification(newer),
        },
        "settings": {
            "assistant_enabled": settings.assistant_enabled,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "environment": settings.environment,
        },
        "flags": {
            name: getattr(settings, name)
            for name in (
                "curriculum_v2_context_boundary_experiment",
                "v23_generation_verifier_experiment",
                "v24_routing_verifier_experiment",
                "v25_verifier_evidence_quality_experiment",
                "v26_verifier_evidence_state_experiment",
                "v27_verifier_decision_boundary_experiment",
                "v28_recommendation_mapping_experiment",
                "v29_evidence_normalization_experiment",
                "v210_integrated_experiment",
                "v211_metadata_integrity_experiment",
                "v212_langchain_experiment",
            )
            if hasattr(settings, name)
        },
        "replays": [],
    }

    for index in range(5):
        with start_agent_run(
            question=QUESTION,
            request_id=f"replay-science-{index}",
            persist=False,
        ) as trace:
            result = verifier.verify(state, request_id=f"replay-science-{index}")
            qa = state.model_copy(deep=True)
            qa.verification = result
            from app.agent.graph_routing import route_after_verification

            route = route_after_verification({"qa": qa}, settings=settings)
        event = trace.latest_event("agent.verification.payload") or {}
        report["replays"].append(
            {
                "index": index,
                "raw_response_present": event.get("raw_response_present"),
                "raw_response_length": event.get("raw_response_length"),
                "raw_response_fingerprint": event.get("raw_response_fingerprint"),
                "json_extracted": event.get("json_extracted"),
                "schema_valid": event.get("schema_valid"),
                "schema_problems": event.get("schema_problems"),
                "fields": event.get("fields"),
                "parsed_recommendation": event.get("parsed_recommendation"),
                "parsed_passed": event.get("parsed_passed"),
                "parsed_score": event.get("parsed_score"),
                "parsed_notes": event.get("parsed_notes"),
                "parsed_issues": event.get("parsed_issues"),
                "parsed_unsupported_claims": event.get("parsed_unsupported_claims"),
                "parsed_incorrect_claims": event.get("parsed_incorrect_claims"),
                "parsed_missing_evidence": event.get("parsed_missing_evidence"),
                "recommendation_before_normalize": event.get(
                    "recommendation_before_normalize"
                ),
                "recommendation_after_normalize": event.get(
                    "recommendation_after_normalize"
                ),
                "final_route": event.get("final_route") or route,
                "fallback_reason": event.get("fallback_reason"),
                "exception": event.get("exception"),
                "raw_response": event.get("raw_response"),
            }
        )
        print(
            f"replay {index} parsed={event.get('parsed_recommendation')} "
            f"passed={event.get('parsed_passed')} "
            f"after={event.get('recommendation_after_normalize')} "
            f"schema_valid={event.get('schema_valid')} route={route}"
        )

    out = TRACE_DIR / "verifier_replay_science_12h.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
