"""
agent/database.py — SQLite persistence layer (SQLAlchemy Core).

Tables:
  research_sessions  — session metadata (id, name, timestamps)
  research_results   — individual Q&A results (question, answer, sources JSON, etc.)

All functions use SQLAlchemy Core (no ORM) for simplicity.

Usage:
    from agent.database import init_db, save_result, load_session_results, list_sessions
    init_db()
    save_result(session_id, research_result)
    results = load_session_results(session_id)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlalchemy as sa
from sqlalchemy import (
    Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine,
)

from agent.config import cfg
from agent.logger import get_logger

logger = get_logger(__name__)

# ── Engine + metadata ─────────────────────────────────────────────────────────

# Ensure the data/ directory exists before creating the SQLite file
_db_path = cfg.db_url.replace("sqlite:///", "")
Path(_db_path).parent.mkdir(parents=True, exist_ok=True)

_engine   = create_engine(cfg.db_url, connect_args={"check_same_thread": False})
_metadata = MetaData()

# ── Table definitions ─────────────────────────────────────────────────────────

sessions_table = Table(
    "research_sessions",
    _metadata,
    Column("session_id",  String,   primary_key=True),
    Column("name",        String,   nullable=False, default="Research Session"),
    Column("created_at",  DateTime, nullable=False),
    Column("updated_at",  DateTime, nullable=False),
    Column("metadata",    Text,     nullable=True),   # JSON
)

results_table = Table(
    "research_results",
    _metadata,
    Column("question_id",    String,  primary_key=True),
    Column("session_id",     String,  nullable=False, index=True),
    Column("question",       Text,    nullable=False),
    Column("answer",         Text,    nullable=False),
    Column("answer_type",    String,  nullable=False, default="full_rag"),
    Column("summary_level",  String,  nullable=False, default="standard"),
    Column("sources",        Text,    nullable=True),   # JSON list
    Column("sub_queries",    Text,    nullable=True),   # JSON list
    Column("steps",          Text,    nullable=True),   # JSON list
    Column("confidence",     Float,   nullable=False, default=0.0),
    Column("latency_ms",     Float,   nullable=False, default=0.0),
    Column("model_used",     String,  nullable=True),
    Column("timestamp",      DateTime, nullable=False),
)


def init_db() -> None:
    """Create all tables if they don't exist."""
    _metadata.create_all(_engine)
    logger.info("Database initialised  url=%s", cfg.db_url)


# ── Session CRUD ──────────────────────────────────────────────────────────────

def save_session(session_id: str, name: str, metadata: dict | None = None) -> None:
    """Insert or update a session record."""
    now = datetime.utcnow()
    with _engine.begin() as conn:
        exists = conn.execute(
            sa.select(sessions_table.c.session_id).where(
                sessions_table.c.session_id == session_id
            )
        ).fetchone()

        if exists:
            conn.execute(
                sessions_table.update()
                .where(sessions_table.c.session_id == session_id)
                .values(name=name, updated_at=now, metadata=json.dumps(metadata or {}))
            )
        else:
            conn.execute(
                sessions_table.insert().values(
                    session_id=session_id,
                    name=name,
                    created_at=now,
                    updated_at=now,
                    metadata=json.dumps(metadata or {}),
                )
            )


def list_sessions() -> List[Dict[str, Any]]:
    """Return all sessions ordered by most recently updated."""
    with _engine.connect() as conn:
        rows = conn.execute(
            sa.select(sessions_table).order_by(sessions_table.c.updated_at.desc())
        ).fetchall()
    return [dict(row._mapping) for row in rows]


def delete_session(session_id: str) -> bool:
    """Delete a session and all its results."""
    with _engine.begin() as conn:
        conn.execute(
            results_table.delete().where(results_table.c.session_id == session_id)
        )
        result = conn.execute(
            sessions_table.delete().where(sessions_table.c.session_id == session_id)
        )
    return result.rowcount > 0


# ── Result CRUD ───────────────────────────────────────────────────────────────

def save_result(session_id: str, result: Any) -> None:
    """
    Persist a ResearchResult to the database.
    `result` is a ResearchResult Pydantic model instance.
    """
    sources    = [s.model_dump() for s in result.sources]
    sub_queries = [sq.model_dump() for sq in result.sub_queries]
    steps      = [
        {
            "step_type":   s.step_type.value,
            "description": s.description,
            "duration_ms": s.duration_ms,
            "success":     s.success,
            "error":       s.error,
        }
        for s in result.steps
    ]

    with _engine.begin() as conn:
        exists = conn.execute(
            sa.select(results_table.c.question_id).where(
                results_table.c.question_id == result.question_id
            )
        ).fetchone()

        row = dict(
            question_id=result.question_id,
            session_id=session_id,
            question=result.question,
            answer=result.answer,
            answer_type=result.answer_type.value,
            summary_level=result.summary_level.value,
            sources=json.dumps(sources, default=str),
            sub_queries=json.dumps(sub_queries),
            steps=json.dumps(steps),
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            model_used=result.model_used,
            timestamp=result.timestamp,
        )

        if exists:
            conn.execute(
                results_table.update()
                .where(results_table.c.question_id == result.question_id)
                .values(**{k: v for k, v in row.items() if k != "question_id"})
            )
        else:
            conn.execute(results_table.insert().values(**row))

    logger.debug("Result persisted  qid=%s  session=%s", result.question_id[:8], session_id[:8])


def load_session_results(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return all results for a session, most recent first."""
    with _engine.connect() as conn:
        rows = conn.execute(
            sa.select(results_table)
            .where(results_table.c.session_id == session_id)
            .order_by(results_table.c.timestamp.desc())
            .limit(limit)
        ).fetchall()

    results = []
    for row in rows:
        r = dict(row._mapping)
        r["sources"]     = json.loads(r.get("sources") or "[]")
        r["sub_queries"] = json.loads(r.get("sub_queries") or "[]")
        r["steps"]       = json.loads(r.get("steps") or "[]")
        results.append(r)
    return results


def get_result(question_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single result by question_id."""
    with _engine.connect() as conn:
        row = conn.execute(
            sa.select(results_table).where(
                results_table.c.question_id == question_id
            )
        ).fetchone()

    if not row:
        return None
    r = dict(row._mapping)
    r["sources"]     = json.loads(r.get("sources") or "[]")
    r["sub_queries"] = json.loads(r.get("sub_queries") or "[]")
    r["steps"]       = json.loads(r.get("steps") or "[]")
    return r


def search_results(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Full-text search over question and answer fields (SQLite LIKE)."""
    pattern = f"%{query}%"
    with _engine.connect() as conn:
        rows = conn.execute(
            sa.select(results_table)
            .where(
                results_table.c.question.like(pattern)
                | results_table.c.answer.like(pattern)
            )
            .order_by(results_table.c.timestamp.desc())
            .limit(limit)
        ).fetchall()
    return [dict(row._mapping) for row in rows]
