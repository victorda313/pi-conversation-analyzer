"""End-to-end pipeline orchestration.

Provides a single `run_pipeline` that can perform **session-level** and/or
**message-level** classification in one pass over sessions to avoid
re-fetching.

Functions:
    run_pipeline(...):
        - Walks sessions that need (re)processing (incremental by timestamp).
        - Optionally classifies the session as a whole.
        - Optionally classifies messages within that session (batch per-session).

Design notes:
    - Incremental sessions: we compute `processed_upto` as max timestamp per
      session and reclassify if newer messages exist.
    - Message classification: by default we **skip** messages that already have
      a row in `message_classification` to save tokens; set
      `reclassify_existing_messages=True` to force refresh.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Iterable

from sqlalchemy import select

from .config import get_config
from .db import (
    get_engine,
    create_result_tables,
    fetch_sessions_to_process,
    fetch_messages_for_session,
    upsert_session_classification,
    upsert_message_classification,
    message_classification_table,
)
from .taxonomy import load_categories
from .storage import get_blob_text
from .openai_client import build_client
from .classifiers.session_classifier import classify_session
from .classifiers.message_classifier import classify_messages

logger = logging.getLogger(__name__)


def _load_instructions(cfg, *, is_session: bool):
    """Load instruction text + version (etag) from Azure Blob."""
    blob = cfg.blob_session_instructions if is_session else cfg.blob_message_instructions
    return get_blob_text(cfg.storage_connection_string, cfg.storage_container, blob)


def _existing_classified_message_ids(engine, session_id: str) -> set[int]:
    """Return the set of message_ids already classified for a given session.

    Args:
        engine: SQLAlchemy engine.
        session_id: Session identifier.

    Returns:
        Set of message_id integers that already exist in message_classification.
    """
    with engine.connect() as conn:
        stmt = select(message_classification_table.c.message_id).where(
            message_classification_table.c.session_id == session_id
        )
        return {row[0] for row in conn.execute(stmt).all()}


def run_pipeline(
    *,
    roles: List[str] = ["user","assistant"],
    since: Optional[str] = None,
    limit: Optional[int] = None,
    run_session_classification: bool = True,
    run_message_classification: bool = True,
    reclassify_existing_messages: bool = False,
    per_session_message_batch_size: Optional[int] = None,
) -> dict:
    """Run the unified classification pipeline in a single pass.

    This function walks candidate sessions once and (optionally) performs both
    session-level and message-level classification, avoiding duplicate fetches.

    Args:
        roles: Which message roles to include when building the session
            transcript and selecting messages to classify (default: ["user"]).
        since: Optional ISO datetime lower bound for incremental processing.
        limit: Optional cap on number of sessions to process this run.
        run_session_classification: If True, classify whole sessions.
        run_message_classification: If True, classify messages within sessions.
        reclassify_existing_messages: If True, classify messages even if
            they already have rows in `message_classification`.
        per_session_message_batch_size: If provided, chunk the per-session
            message list into batches of this size for model calls. If None,
            classify all target messages for a session in one call.

    Returns:
        dict: {"sessions_processed": int, "messages_processed": int}
    """
    cfg = get_config()
    engine = get_engine(cfg.database_url)
    create_result_tables(engine)

    client = build_client(cfg)
    categories = load_categories()

    # Load instruction blobs only if needed
    sess_instr_text, sess_instr_ver = ("", None)
    msg_instr_text, msg_instr_ver = ("", None)
    if run_session_classification:
        sess_instr_text, sess_instr_ver = _load_instructions(cfg, is_session=True)
    if run_message_classification:
        msg_instr_text, msg_instr_ver = _load_instructions(cfg, is_session=False)

    sessions = fetch_sessions_to_process(engine, since=since)
    if limit:
        sessions = sessions[:limit]

    sessions_processed = 0
    messages_processed = 0

    for s in sessions:
        session_id = s["session_id"]
        max_ts = s["max_ts"]
        msgs_raw = fetch_messages_for_session(engine, session_id, roles=roles)

        # Prepare transcript for session classification
        messages_for_session = [
            {
                "role": m["role"],
                "content": (m["content"] or "")[:4000],
                "timestamp": m["timestamp"].isoformat(),
            }
            for m in msgs_raw
        ]

        # 1) Session-level classification
        if run_session_classification and messages_for_session:
            result = classify_session(
                client=client,
                model=cfg.assistant_model,
                system_instructions=sess_instr_text,
                categories=categories,
                session_id=session_id,
                messages=messages_for_session,
                max_tokens=cfg.classifier_max_tokens,
                temperature=cfg.classifier_temperature,
            )
            upsert_session_classification(
                engine,
                session_id=session_id,
                primary_category=result.get("primary_category", "other"),
                all_categories=result.get("scores", {}),
                processed_upto=max_ts,
                model=cfg.assistant_model,
                instructions_version=sess_instr_ver,
                notes=result.get("rationale"),
            )
            sessions_processed += 1

        # 2) Message-level classification (within the same session)
        if run_message_classification and msgs_raw:
            # Optionally skip messages we already have
            already = set()
            if not reclassify_existing_messages:
                already = _existing_classified_message_ids(engine, session_id)

            targets = [
                {"message_id": m["id"], "text": (m["content"] or "")[:4000]}
                for m in msgs_raw
                if m["id"] not in already
            ]
            if not targets:
                continue

            def _chunks(seq: List[dict], n: int) -> Iterable[List[dict]]:
                for i in range(0, len(seq), n):
                    yield seq[i : i + n]

            batches = [targets]
            if per_session_message_batch_size and per_session_message_batch_size > 0:
                batches = list(_chunks(targets, per_session_message_batch_size))

            by_id = {m["id"]: m for m in msgs_raw}
            for batch in batches:
                results = classify_messages(
                    client=client,
                    model=cfg.assistant_model,
                    system_instructions=msg_instr_text,
                    categories=categories,
                    batch=batch,
                    max_tokens=cfg.classifier_max_tokens,
                    temperature=cfg.classifier_temperature,
                )
                for r in results:
                    try:
                        mid = int(r["message_id"])  # model returns numeric id per schema
                    except Exception:
                        continue
                    m = by_id.get(mid)
                    if not m:
                        continue
                    primary = r.get("primary_category", "other")
                    scores = r.get("scores", {})
                    upsert_message_classification(
                        engine,
                        message_id=mid,
                        session_id=m["session_id"],
                        role=m["role"],
                        primary_category=primary,
                        all_categories=scores,
                        model=cfg.assistant_model,
                        instructions_version=msg_instr_ver,
                    )
                    messages_processed += 1

    logger.info(
        "Unified pipeline complete. Sessions: %d, Messages: %d",
        sessions_processed,
        messages_processed,
    )
    return {"sessions_processed": sessions_processed, "messages_processed": messages_processed}
