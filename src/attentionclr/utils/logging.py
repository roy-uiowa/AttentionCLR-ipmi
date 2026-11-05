"""Rich logging. get_logger(__name__) everywhere."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "attentionclr", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger