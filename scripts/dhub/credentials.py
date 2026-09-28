"""The credentials a platform run holds, which lapse long before the run ends."""

from __future__ import annotations

from digitalhub.stores.client.base.factory import get_client

SECRETS = [
    "DHCORE_PERSONAL_ACCESS_TOKEN",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
]


def refresh() -> None:
    """Mint the run's credentials again, the platform's and the store's alike."""
    get_client().eval_retry()
