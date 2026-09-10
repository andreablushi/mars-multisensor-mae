"""The credentials a platform run holds, which lapse long before the run ends."""

from __future__ import annotations

import os

from digitalhub.stores.client.base.factory import get_client
from dotenv import load_dotenv

from config.paths import REPO_ROOT

TOKEN = "DHCORE_PERSONAL_ACCESS_TOKEN"

WANDB_KEY = "WANDB_API_KEY"

FORWARDED = ("DHCORE_ISSUER", "DHCORE_CLIENT_ID", "WANDB_ENTITY", "WANDB_PROJECT")


def forwarded_envs() -> list[dict[str, str]]:
    """Return what a job is told from the `.env` beside this file's repository.

    Returns:
        told: The authority to ask for credentials and the client to ask as,
            which the token names neither of, and the entity and project the
            run is tracked under, each as the platform spells a variable.

    Raises:
        RuntimeError: When any is unset, which a job cannot run without.
    """
    load_dotenv(REPO_ROOT / ".env")
    told = []
    for name in FORWARDED:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is unset; see .env.example")
        told.append({"name": name, "value": value})
    return told


def refresh() -> None:
    """Mint the run's credentials again, the platform's and the store's alike."""
    get_client().eval_retry()
