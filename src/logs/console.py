"""Writing what a run does to the terminal."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def logger(name: str) -> logging.Logger:
    """Return a logger writing through rich, the root set up once for every one.

    Args:
        name: Whose logger, which is the module's own name.

    Returns:
        log: The logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)
