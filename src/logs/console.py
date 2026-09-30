"""Writing what a run does to the terminal."""

from __future__ import annotations

import logging


def console_logger(name: str) -> logging.Logger:
    """Return a logger writing to the console.

    Args:
        name: Whose logger, which is the module's own name.

    Returns:
        log: The logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(name)
