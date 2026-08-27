"""LangGraph checkpoint backends for short-term thread memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import Settings
from app.logging_utils import get_logger

logger = get_logger(__name__)

# Keep SQLite connections alive for the process (SqliteSaver needs an open conn).
_open_sqlite_connections: list[sqlite3.Connection] = []


def build_checkpointer(settings: Settings) -> Any | None:
    """Return a LangGraph checkpointer, or None when checkpointing is disabled.

    - memory: InMemorySaver (lost on restart)
    - sqlite: SqliteSaver to a local file (survives restarts)
    """
    if not settings.agent_checkpointing_enabled:
        return None

    backend = (settings.agent_checkpoint_backend or "memory").strip().lower()
    if backend in ("memory", "inmemory", "in_memory"):
        from langgraph.checkpoint.memory import InMemorySaver

        logger.info("agent.checkpoint.backend", extra={"backend": "memory"})
        return InMemorySaver()

    if backend == "sqlite":
        return _build_sqlite_checkpointer(settings)

    raise ValueError(f"Unsupported AGENT_CHECKPOINT_BACKEND: {backend!r}")


def _build_sqlite_checkpointer(settings: Settings) -> Any:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "AGENT_CHECKPOINT_BACKEND=sqlite requires "
            "langgraph-checkpoint-sqlite (pip install langgraph-checkpoint-sqlite)"
        ) from exc

    path = Path(settings.agent_checkpoint_sqlite_path or "data/checkpoints.sqlite")
    if not path.is_absolute():
        # Resolve relative to curriculum-agent package root (…/curriculum-agent).
        root = Path(__file__).resolve().parents[2]
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    _open_sqlite_connections.append(conn)
    saver = SqliteSaver(conn)
    saver.setup()
    logger.info(
        "agent.checkpoint.backend",
        extra={"backend": "sqlite", "path": str(path)},
    )
    return saver


def thread_config(conversation_id: str, *, request_id: str | None = None) -> dict:
    """LangGraph invoke/config keyed by conversation thread."""
    configurable: dict[str, Any] = {"thread_id": conversation_id}
    if request_id:
        configurable["request_id"] = request_id
    return {"configurable": configurable}
