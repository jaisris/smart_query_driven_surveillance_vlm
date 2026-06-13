"""Centralised logging configuration."""

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_root_configured = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _root_configured
    if not _root_configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
        _root_configured = True
    return logging.getLogger(name)
