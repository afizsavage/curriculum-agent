"""Curriculum Q&A LangGraph factory and inspection helpers.

LangGraph orchestrates domain services; it does not replace them.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from app.agent.graph_nodes import GraphNodes
from app.agent.graph_routing import route_after_prepare, route_after_verification
from app.agent.graph_state import GraphState
from app.config import Settings
from app.logging_utils import get_logger

logger = get_logger(__name__)


def build_curriculum_qa_graph(
    *,
    nodes: GraphNodes,
    settings: Settings,
    checkpointer: Any | None = None,
):
    """Register nodes/edges, attach optional checkpointer, compile the graph."""

    def _route_after_verification(graph_state: GraphState) -> str:
        return route_after_verification(graph_state, settings=settings)

    builder: StateGraph = StateGraph(GraphState)

    builder.add_node("understand", nodes.understand)
    builder.add_node("prepare_cycle", nodes.prepare_cycle)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("generate_answer", nodes.generate_answer)
    builder.add_node("verify_answer", nodes.verify_answer)
    builder.add_node("clarify", nodes.clarify)
    builder.add_node("fallback", nodes.fallback)
    builder.add_node("finish", nodes.finish)

    builder.add_edge(START, "understand")
    builder.add_edge("understand", "prepare_cycle")
    builder.add_conditional_edges(
        "prepare_cycle",
        route_after_prepare,
        {"retrieve": "retrieve", "fallback": "fallback"},
    )
    builder.add_edge("retrieve", "generate_answer")
    builder.add_edge("generate_answer", "verify_answer")
    builder.add_conditional_edges(
        "verify_answer",
        _route_after_verification,
        {
            "finish": "finish",
            "retrieve_more": "prepare_cycle",
            "clarify": "clarify",
            "fallback": "fallback",
        },
    )
    builder.add_edge("finish", END)
    builder.add_edge("clarify", END)
    builder.add_edge("fallback", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info(
        "agent.graph.compiled",
        extra={
            "checkpointing": checkpointer is not None,
            "nodes": [
                "understand",
                "prepare_cycle",
                "retrieve",
                "generate_answer",
                "verify_answer",
                "clarify",
                "fallback",
                "finish",
            ],
        },
    )
    return compiled


def graph_mermaid(compiled_graph) -> str:
    """Return Mermaid source for the compiled graph (dev inspection)."""
    try:
        return compiled_graph.get_graph().draw_mermaid()
    except Exception as exc:  # pragma: no cover - rendering is best-effort
        return f"# graph inspection unavailable: {exc}"


def graph_ascii(compiled_graph) -> str:
    """Return a compact ASCII outline of known Curriculum Q&A topology."""
    return "\n".join(
        [
            "START",
            "  ↓",
            "understand",
            "  ↓",
            "prepare_cycle",
            "  ├─(limits)→ fallback → END",
            "  ↓",
            "retrieve",
            "  ↓",
            "generate_answer",
            "  ↓",
            "verify_answer",
            "  ├── finish → END",
            "  ├── retrieve_more → prepare_cycle ↺",
            "  ├── clarify → END",
            "  └── fallback → END",
        ]
    )


def print_graph_inspection(compiled_graph, *, printer: Callable[[str], None] = print) -> None:
    printer(graph_ascii(compiled_graph))
    printer("")
    printer(graph_mermaid(compiled_graph))
