"""Placing whole features in a split, so no two patches of one straddle two."""

from __future__ import annotations

import random
from collections.abc import Iterable


def split_features(
    features: Iterable[tuple[str, str]], config: dict
) -> dict[str, list[tuple[str, str]]]:
    """Return which features each split holds, whole features at a time.

    Args:
        features: What tells each feature apart, its class and its name.
        config: The choices a read is made with, which carry the share each
            split holds and the number that fixes where a feature falls.

    Returns:
        splits: The features of each split, keyed as the config names it. A
            feature falls in one split by its name alone, so no two patches of
            it straddle two splits and a later build that adds features leaves
            the ones already placed where they were.
    """
    shares = config["split"]
    total = sum(shares.values())
    splits: dict[str, list[tuple[str, str]]] = {name: [] for name in shares}
    for identity in features:
        drawn = random.Random(f"{config['seed']}/{'/'.join(identity)}").random() * total
        running = 0.0
        for name in shares:
            running += shares[name]
            if drawn < running:
                break
        splits[name].append(identity)
    return splits
