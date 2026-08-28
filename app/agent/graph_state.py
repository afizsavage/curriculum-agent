"""LangGraph state schema wrapping CurriculumQAState.

CurriculumQAState remains the domain source of truth. GraphState is a thin
TypedDict envelope for orchestration metadata (visited nodes, routing, limits).
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict
from typing_extensions import NotRequired
from uuid import uuid4

from app.agent.state import CurriculumQAState


class GraphState(TypedDict):
    """LangGraph channels for the Curriculum Q&A workflow."""

    qa: CurriculumQAState
    request_id: NotRequired[Optional[str]]
    graph_run_id: NotRequired[str]
    visited_nodes: NotRequired[list[str]]
    route: NotRequired[Optional[str]]
    max_iterations_hit: NotRequired[bool]
    fallback_reason: NotRequired[Optional[str]]
    # Snapshot of prior-turn filters for understand inheritance.
    prior_filters: NotRequired[dict[str, Any]]


ALLOWED_ROUTES = frozenset(
    {"finish", "retrieve_more", "regenerate", "clarify", "fallback"}
)


def new_graph_run_id() -> str:
    return str(uuid4())


def mark_visited(graph_state: GraphState, node: str) -> list[str]:
    """Append a node name for path tracing (per-turn list, last-write-wins)."""
    visited = list(graph_state.get("visited_nodes") or [])
    if not visited or visited[-1] != node:
        visited.append(node)
    return visited


def initial_graph_input(
    *,
    qa: CurriculumQAState,
    request_id: str | None = None,
    prior_filters: dict[str, Any] | None = None,
) -> GraphState:
    return {
        "qa": qa,
        "request_id": request_id,
        "graph_run_id": new_graph_run_id(),
        "visited_nodes": [],
        "route": None,
        "max_iterations_hit": False,
        "fallback_reason": None,
        "prior_filters": prior_filters or {},
    }
