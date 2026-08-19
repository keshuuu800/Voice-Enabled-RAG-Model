"""
Structured logging configuration.
Uses Python's standard logging with a JSON-friendly format for production.
"""
import logging
import sys
from app.core.config import get_settings


def setup_logging() -> None:
    """Configure root logger. Call once at application startup."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called multiple times (e.g., during tests)
    if not root.handlers:
        root.addHandler(handler)

    # Quiet down noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Use __name__ as the name."""
    return logging.getLogger(name)
