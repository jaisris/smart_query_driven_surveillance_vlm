"""Centralised logging: file (DEBUG) + console (INFO) with rotation.

Usage
-----
# Once at application startup (ui/app.py, video_pipeline.py CLI, etc.)
from utils.logger import setup_logging
setup_logging()          # writes to logs/surveillance.log

# In every module
from utils.logger import get_logger
logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> None:
    """Configure root logger with rotating file handler + console handler.

    Call exactly once at application startup. Idempotent on repeated calls.
    """
    global _configured
    if _configured:
        return

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "surveillance.log")

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # File: DEBUG and above — full trace for debugging
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console: INFO and above — concise progress output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers added by get_logger's fallback before setup_logging was called
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _configured = True
    logging.getLogger(__name__).info(
        "Logging initialised — file: %s", os.path.abspath(log_file)
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Triggers basic console fallback if setup_logging() was never called."""
    global _configured
    if not _configured:
        _basic_console_setup()
    return logging.getLogger(name)


def _basic_console_setup() -> None:
    """Minimal fallback: console-only INFO logging (used in tests / notebooks)."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    _configured = True
