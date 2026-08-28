"""POST /api/v1/agent/ask"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.agent.metrics import get_metrics
from app.agent.orchestrator import CurriculumQAAgent
from app.agent.response_mapper import map_graph_result_to_response
from app.deps import get_agent
from app.logging_utils import get_logger, new_request_id, timed_request
from app.schemas.agent import AskRequest, AskResponse

router = APIRouter(prefix="/agent", tags=["agent"])
logger = get_logger(__name__)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the Curriculum Q&A agent",
    description=(
        "Accept a curriculum question, run the LangGraph Curriculum Q&A workflow "
        "(understand → retrieve → generate → verify with a bounded loop), and "
        "return a grounded answer, clarification, or insufficient-evidence fallback."
    ),
    responses={
        422: {"description": "Invalid request (validation error)"},
        500: {"description": "Agent execution or configuration failure"},
        502: {"description": "LLM provider or tool failure"},
        503: {"description": "Curriculum API unavailable"},
        504: {"description": "LLM or Curriculum API timeout"},
    },
)
def ask(
    payload: AskRequest,
    request: Request,
    agent: CurriculumQAAgent = Depends(get_agent),
) -> AskResponse:
    request_id = getattr(request.state, "request_id", None) or new_request_id()
    conversation_id = (
        str(payload.conversation_id) if payload.conversation_id is not None else None
    )

    with timed_request(
        logger,
        request_id=request_id,
        conversation_id=conversation_id,
        question=payload.question,
        model=agent.llm.model,
    ) as ctx:
        state = agent.ask(
            payload.question,
            conversation_id=conversation_id,
            request_id=request_id,
            context_boundary_experiment=payload.context_boundary_experiment,
        )
        ctx["conversation_id"] = state.conversation_id
        ctx["status"] = state.status.value
        ctx["iteration"] = state.iteration
        ctx["tool_calls"] = state.tool_calls
        ctx["confidence"] = (
            state.answer_confidence.value if state.answer_confidence else None
        )

        return map_graph_result_to_response(
            qa=state,
            llm=agent.llm,
            graph_state={
                "qa": state,
                "graph_run_id": state.metadata.get("graph_run_id"),
                "visited_nodes": list(state.metadata.get("visited_nodes") or []),
                "route": state.metadata.get("route"),
                "max_iterations_hit": bool(
                    state.metadata.get("max_iterations_hit")
                ),
            },
        )


@router.get(
    "/metrics",
    summary="Agent metrics snapshot",
    description="In-process counters for verification loop observability.",
)
def metrics() -> dict:
    return get_metrics().snapshot()


@router.get(
    "/graph",
    summary="Inspect compiled LangGraph topology (development)",
    description="Returns ASCII and Mermaid representations of the Curriculum Q&A graph.",
)
def inspect_graph(agent: CurriculumQAAgent = Depends(get_agent)) -> dict:
    inspection = agent.inspect_graph()
    parts = inspection.split("\n\n", 1)
    return {
        "ascii": parts[0],
        "mermaid": parts[1] if len(parts) > 1 else "",
    }


@router.get(
    "/debug/runs/{agent_run_id}",
    summary="Fetch a diagnostic execution trace (development only)",
)
def get_debug_run(agent_run_id: str) -> dict:
    from fastapi import HTTPException

    from app.agent.trace import get_trace_store
    from app.config import get_settings

    if get_settings().environment.strip().lower() == "production":
        raise HTTPException(status_code=404, detail="Not found")
    trace = get_trace_store().get(agent_run_id)
    if trace is None:
        # Fall back to on-disk persistence.
        from pathlib import Path
        import json

        path = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "traces"
            / f"{agent_run_id}.json"
        )
        if path.is_file():
            return json.loads(path.read_text())
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.to_dict()
