"""Logging setup helpers."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: str = "INFO", logger_name: Optional[str] = None) -> logging.Logger:
    """Configure and return a project logger."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level}")

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for project modules."""

    return logging.getLogger(name)
