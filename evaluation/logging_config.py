"""Logging configuration for the evaluation package."""

from __future__ import annotations

import logging
import sys

# Module-level logger; consumers use get_logger()
_LOGGER: logging.Logger | None = None


def setup_logging(
    level: str | int = "INFO",
    format_string: str = "[%(levelname)s] %(message)s",
    stream: bool = True,
) -> logging.Logger:
    """Configure and return the evaluation package logger.

    Parameters
    ----------
    level : str or int, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL or numeric).
    format_string : str, default="[%(levelname)s] %(message)s"
        Log message format.
    stream : bool, default=True
        If True, add a StreamHandler to stderr.

    Returns:
    -------
    logging.Logger
        The configured evaluation logger.
    """
    global _LOGGER
    logger = logging.getLogger("evaluation")
    if _LOGGER is not None and logger.handlers:
        # Already configured; optionally update level
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
        return logger

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(format_string)

    if stream:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    _LOGGER = logger
    return logger


def get_logger() -> logging.Logger:
    """Return the evaluation package logger, configuring it if needed."""
    logger = logging.getLogger("evaluation")
    if not logger.handlers:
        setup_logging()
    return logger
