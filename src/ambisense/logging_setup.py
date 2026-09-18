"""Central logging configuration.

Every module obtains its logger through :func:`get_logger` so that log output
carries the originating layer (``ambisense.ambiguity.lexical``, and so on).
That makes it obvious during a demo which layer produced which decision.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from ambisense.config import LoggingConfig

_CONFIGURED = False


def configure_logging(config: Optional[LoggingConfig] = None) -> None:
    """Install a single stream handler on the ``ambisense`` logger.

    Idempotent: calling it more than once in a process (as the test suite
    and the live-LLM-test loop both do) will not duplicate handlers or
    duplicate log lines.
    """
    global _CONFIGURED
    config = config or LoggingConfig()

    root = logging.getLogger("ambisense")
    level = getattr(logging, config.level.upper(), logging.INFO)
    root.setLevel(level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(config.format, config.date_format))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    else:
        for handler in root.handlers:
            handler.setFormatter(
                logging.Formatter(config.format, config.date_format)
            )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. ``get_logger(__name__)``."""
    if not name.startswith("ambisense"):
        name = f"ambisense.{name}"
    return logging.getLogger(name)
