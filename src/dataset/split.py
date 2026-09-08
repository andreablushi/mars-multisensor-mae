"""Placing whole features in a split, so no two patches of one straddle two."""

from __future__ import annotations

from collections.abc import Iterable

from dataset.models.settings import Settings
from dataset.seed import seeded_number

PLACES = 10_000


def split_features(
    features: Iterable[tuple[str, str]], settings: Settings
) -> dict[str, list[tuple[str, str]]]:
    """Return which features each split holds, whole features at a time.

    Args:
        features: What tells each feature apart, its class and its name.
        settings: The settled choices, which carry the share each split holds
            and the number that fixes where a feature falls.

    Returns:
        splits: The features of each split, keyed as the config names it. A
            feature falls in one split by its name alone, so no two patches of
            it straddle two splits and a later build that adds features leaves
            the ones already placed where they were.
    """
    names = list(settings.shares)
    total = sum(settings.shares.values())
    edges, running = [], 0.0
    for name in names:
        running += settings.shares[name] / total
        edges.append(running * PLACES)
    splits: dict[str, list[tuple[str, str]]] = {name: [] for name in names}
    for identity in features:
        drawn = seeded_number("/".join(identity), settings.seed) % PLACES
        placed = next(
            (name for name, edge in zip(names, edges, strict=True) if drawn < edge),
            names[-1],
        )
        splits[placed].append(identity)
    return splits
