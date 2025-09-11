"""CLI entrypoint to run the unified classification pipeline.

This CLI wraps `run_pipeline(...)`, allowing you to choose whether to run
session-level and/or message-level classification in a single pass.

Examples:
    # Run both session and message classification with defaults
    python -m src.main

    # Only session classification
    python -m src.main --no-messages

    # Only message classification (user+assistant) with per-session batches of 20
    python -m src.main --no-session --roles user assistant --per-session-message-batch-size 20

    # Incremental since a timestamp, cap sessions to process at 100
    python -m src.main --since 2025-01-01T00:00:00Z --limit 100

    # Reclassify messages even if they already have a row
    python -m src.main --reclassify-existing-messages

Notes:
    - By default, both session- and message-level classification run.
    - By default, only 'user' messages are considered/classified.
"""
from __future__ import annotations
import argparse
import logging

from .logging_utils import configure_logging
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace with parsed arguments.
    """
    p = argparse.ArgumentParser(description="Unified chatbot classification pipeline")

    # Role selection
    p.add_argument(
        "--roles",
        nargs="*",
        default=["user"],
        help="Roles to include when building transcripts and selecting messages (e.g., user assistant tool).",
    )

    # Incremental controls
    p.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO timestamp lower bound for incremental processing (e.g., 2025-01-01T00:00:00Z).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of sessions to process this run.",
    )

    # Toggle which classifiers to run
    p.add_argument(
        "--no-session",
        action="store_true",
        help="Disable session-level classification (default: enabled).",
    )
    p.add_argument(
        "--no-messages",
        action="store_true",
        help="Disable message-level classification (default: enabled).",
    )

    # Message classification behavior
    p.add_argument(
        "--reclassify-existing-messages",
        action="store_true",
        help="Force reclassification of messages even if they already have results.",
    )
    p.add_argument(
        "--per-session-message-batch-size",
        type=int,
        default=None,
        help="If set, split each session's messages into batches of this size per model call.",
    )

    # Logging
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    return p.parse_args()


def main() -> None:
    """Entrypoint for CLI execution."""
    args = parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level)

    run_flags = {
        "run_session_classification": not args.no_session,
        "run_message_classification": not args.no_messages,
    }

    result = run_pipeline(
        roles=args.roles,
        since=args.since,
        limit=args.limit,
        reclassify_existing_messages=args.reclassify_existing_messages,
        per_session_message_batch_size=args.per_session_message_batch_size,
        **run_flags,
    )

    # Friendly summary
    logging.getLogger(__name__).info(
        "Completed. Sessions processed: %d | Messages processed: %d",
        result.get("sessions_processed", 0),
        result.get("messages_processed", 0),
    )


if __name__ == "__main__":
    main()
