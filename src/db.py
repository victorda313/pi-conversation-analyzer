"""Database access layer using SQLAlchemy Core.

Defines table metadata for existing `messages` table (minimal columns),
result tables for classifications, and helper functions for common queries
and upserts. Uses PostgreSQL JSONB types and ON CONFLICT for idempotency.

Functions:
    get_engine(database_url: str) -> Engine
    create_result_tables(engine: Engine) -> None
    fetch_sessions_to_process(engine: Engine, since: str | None) -> list[dict]
    fetch_messages_for_session(engine: Engine, session_id: str, roles: list[str]) -> list[dict]
    fetch_unclassified_messages(engine: Engine, roles: list[str], since: str | None, limit: int | None) -> list[dict]
    upsert_session_classification(...)
    upsert_message_classification(...)
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Table, Column, MetaData, Integer, String, Text, DateTime, select, func,
)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

metadata = MetaData()

# Existing message table (minimal definition for reads)
message_table = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", String, nullable=False),
    Column("user_id", String),
    Column("role", String, nullable=False),  # assistant|user|tool
    Column("content", Text),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("tool_call_id", String),
)

# Result tables
session_classification_table = Table(
    "session_classification",
    metadata,
    Column("session_id", String, primary_key=True),
    Column("primary_category", String, nullable=False),
    Column("all_categories", JSONB, nullable=False),
    Column("processed_upto", DateTime(timezone=True), nullable=False),
    Column("run_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("model", String, nullable=False),
    Column("instructions_version", String),
    Column("notes", Text),
)

message_classification_table = Table(
    "message_classification",
    metadata,
    Column("message_id", Integer, primary_key=True),
    Column("session_id", String, nullable=False),
    Column("role", String, nullable=False),
    Column("primary_category", String, nullable=False),
    Column("all_categories", JSONB, nullable=False),
    Column("run_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("model", String, nullable=False),
    Column("instructions_version", String),
)


def get_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine.

    Args:
        database_url: PostgreSQL connection string.

    Returns:
        Engine: SQLAlchemy engine with pool_pre_ping.
    """
    return create_engine(database_url, pool_pre_ping=True, future=True)


def create_result_tables(engine: Engine) -> None:
    """Create result tables if they do not exist.

    Args:
        engine: Database engine.
    """
    try:
        metadata.create_all(engine, tables=[session_classification_table, message_classification_table])
        logger.info("Ensured classification tables exist.")
    except SQLAlchemyError as exc:
        logger.exception("Failed creating tables: %s", exc)
        raise


def fetch_sessions_to_process(engine: Engine, since: Optional[str] = None) -> list[dict]:
    """Return sessions needing (re)classification.

    A session needs processing if it has no row in session_classification OR
    if its current max(messages.timestamp) > session_classification.processed_upto.

    Args:
        engine: Database engine.
        since: Optional ISO datetime string to restrict to newer messages.

    Returns:
        List of dicts: {session_id, max_ts, message_count}
    """
    with engine.connect() as conn:
        m = message_table.alias("m")
        sc = session_classification_table.alias("sc")

        base = (
            select(
                m.c.session_id.label("session_id"),
                func.max(m.c.timestamp).label("max_ts"),
                func.count(m.c.id).label("message_count"),
            )
            .group_by(m.c.session_id)
        )
        if since:
            base = base.where(m.c.timestamp >= since)

        subq = base.subquery()

        # Left join against existing classifications
        join_stmt = (
            select(subq.c.session_id, subq.c.max_ts, subq.c.message_count)
            .select_from(subq.outerjoin(sc, sc.c.session_id == subq.c.session_id))
            .where((sc.c.processed_upto.is_(None)) | (subq.c.max_ts > sc.c.processed_upto) | (sc.c.session_id.is_(None)))
            .order_by(subq.c.max_ts.asc())
        )
        rows = conn.execute(join_stmt).mappings().all()
        return [dict(r) for r in rows]


def fetch_messages_for_session(engine: Engine, session_id: str, roles: list[str]) -> list[dict]:
    """Fetch messages for a session filtered by roles.

    Args:
        engine: Database engine.
        session_id: Session ID to fetch.
        roles: Roles to include (e.g., ["user"]).

    Returns:
        List of dicts for messages ordered by timestamp.
    """
    with engine.connect() as conn:
        stmt = (
            select(
                message_table.c.id,
                message_table.c.session_id,
                message_table.c.role,
                message_table.c.content,
                message_table.c.timestamp,
            )
            .where(message_table.c.session_id == session_id)
            .where(message_table.c.role.in_(roles))
            .order_by(message_table.c.timestamp.asc())
        )
        rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]


def fetch_unclassified_messages(
    engine: Engine,
    roles: list[str],
    since: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Fetch messages that do not yet have a classification.

    Args:
        engine: Database engine.
        roles: Roles to include.
        since: Optional lower bound on timestamp as ISO string.
        limit: Optional max number of messages to return.

    Returns:
        List of dicts with message fields.
    """
    with engine.connect() as conn:
        m = message_table.alias("m")
        mc = message_classification_table.alias("mc")

        stmt = (
            select(m.c.id, m.c.session_id, m.c.role, m.c.content, m.c.timestamp)
            .select_from(m.outerjoin(mc, mc.c.message_id == m.c.id))
            .where(mc.c.message_id.is_(None))
            .where(m.c.role.in_(roles))
        )
        if since:
            stmt = stmt.where(m.c.timestamp >= since)
        stmt = stmt.order_by(m.c.timestamp.asc())
        if limit:
            stmt = stmt.limit(limit)

        rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]


def upsert_session_classification(
    engine: Engine,
    *,
    session_id: str,
    primary_category: str,
    all_categories: dict,
    processed_upto: datetime,
    model: str,
    instructions_version: str | None,
    notes: str | None = None,
) -> None:
    """Upsert a session-level classification row.

    Args:
        engine: DB engine.
        session_id: Session identifier.
        primary_category: Winning class label.
        all_categories: Score/probability per label.
        processed_upto: Max message timestamp included in this run.
        model: Model name used.
        instructions_version: ETag or similar to trace instructions.
        notes: Optional notes.
    """
    payload = {
        "session_id": session_id,
        "primary_category": primary_category,
        "all_categories": all_categories,
        "processed_upto": processed_upto,
        "model": model,
        "instructions_version": instructions_version,
        "notes": notes,
    }
    ins = pg_insert(session_classification_table).values(**payload)
    on_conflict = ins.on_conflict_do_update(
        index_elements=[session_classification_table.c.session_id],
        set_={
            "primary_category": ins.excluded.primary_category,
            "all_categories": ins.excluded.all_categories,
            "processed_upto": ins.excluded.processed_upto,
            "run_at": func.now(),
            "model": ins.excluded.model,
            "instructions_version": ins.excluded.instructions_version,
            "notes": ins.excluded.notes,
        },
    )
    with engine.begin() as conn:
        conn.execute(on_conflict)


def upsert_message_classification(
    engine: Engine,
    *,
    message_id: int,
    session_id: str,
    role: str,
    primary_category: str,
    all_categories: dict,
    model: str,
    instructions_version: str | None,
) -> None:
    """Upsert a message-level classification row.

    Args:
        engine: DB engine.
        message_id: Message ID (primary key in message table).
        session_id: Session ID.
        role: Message role.
        primary_category: Winning class label.
        all_categories: Score/probability per label.
        model: Model used.
        instructions_version: ETag or similar to trace instructions.
    """
    payload = {
        "message_id": message_id,
        "session_id": session_id,
        "role": role,
        "primary_category": primary_category,
        "all_categories": all_categories,
        "model": model,
        "instructions_version": instructions_version,
    }
    ins = pg_insert(message_classification_table).values(**payload)
    on_conflict = ins.on_conflict_do_update(
        index_elements=[message_classification_table.c.message_id],
        set_={
            "primary_category": ins.excluded.primary_category,
            "all_categories": ins.excluded.all_categories,
            "run_at": func.now(),
            "model": ins.excluded.model,
            "instructions_version": ins.excluded.instructions_version,
        },
    )
    with engine.begin() as conn:
        conn.execute(on_conflict)
