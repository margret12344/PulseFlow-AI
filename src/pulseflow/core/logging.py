
import logging
import logging.config
from pathlib import Path

import structlog


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/pulseflow.log",
) -> None:
    """Configure structured logging for PulseFlow AI."""

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    logging.basicConfig(
        format="%(message)s",
        level=level.upper(),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )

    structlog.configure(
        processors=shared_processors
        + [
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "pulseflow"):
    """Return a structured logger."""
    return structlog.get_logger(name)
