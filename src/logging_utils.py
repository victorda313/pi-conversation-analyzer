"""Logging configuration utilities.

Provides a single function to configure structured logging for the project.

Functions:
    configure_logging(level: int = logging.INFO) -> None
"""
from __future__ import annotations
import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a concise format.

    Args:
        level: Logging level, defaults to logging.INFO.
    """
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
