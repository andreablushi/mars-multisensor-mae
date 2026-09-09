#!/usr/bin/env python
"""What one build of the dataset holds, and what one small sample of it cuts into."""

from __future__ import annotations

import argparse
import math

from building.metadata.observation import ObservationMetadata
from rich.console import Console

from dataset import config as settings
from dataset import split, stats
from dataset.models.patch import Patch
from dataset.patches import grid
from dataset.store import index, load
from dataset.store.dh import build as opened


def describe_build(build_name: str | None, per_instrument: int) -> int:
    """Print what one published build holds, and what one sample of it cuts into.

    Args:
        build_name: The build to read, or None for the one the config names.
        per_instrument: How many observations of each instrument the sample draws.

    Returns:
        code: A process exit code, non zero when the build holds no crop.
    """
    console = Console()
    config = settings.load_config()
    config["build"] = build_name or config["build"]
    config["per_instrument"] = per_instrument
    build = opened.opened_build(config)
    console.print(f"reading [bold]dataset-{config['build']}[/bold] from {build.bucket}")

    manifest = index.read_manifest(build)
    observations = index.read_observations(build)
    console.print(
        f"built {manifest['built_at']}, {len(observations):,} crops "
        f"over {', '.join(manifest['instruments'])}"
    )
    for instrument, held in stats.pooled_statistics(observations).items():
        tiles = settings.tiles_of(config, instrument)
        counted = sum(
            grid.patch_count(one.shape, one.axes, tiles)
            for one in observations
            if one.instrument == instrument
        )
        console.print(
            f"  {instrument:<7} {counted:>10,} patches   "
            f"mean {held['mean']:>12.4g}  sd {held['deviation']:>10.4g}"
        )

    standing = index.observations_by_feature(observations)
    if not standing:
        console.print("[red]the build holds no crop[/red]")
        return 1
    placed = split.split_features(standing, config)
    counted = {name: len(held) for name, held in placed.items()}
    console.print(f"{len(standing):,} features with crops, split {counted}")

    identity, rows = _smallest_multisensor(standing)
    drawn = index.drawn_observations(rows, config)
    weight = sum(math.prod(one.shape) for one in drawn)
    console.print(
        f"loading [bold]{' '.join(identity)}[/bold], {len(drawn)} crops of "
        f"{', '.join(sorted({one.instrument for one in drawn}))}, "
        f"{weight / 1e6:.1f}M values"
    )
    for line in _by_instrument(load.load_patches(drawn, build, config)):
        console.print(line)
    return 0


def _smallest_multisensor(
    standing: dict[tuple[str, str], list[ObservationMetadata]],
) -> tuple[tuple[str, str], list[ObservationMetadata]]:
    """Return the lightest feature that still reaches more than one instrument.

    Args:
        standing: The crops of each feature, keyed by what tells it apart.

    Returns:
        identity: What tells the feature apart, its class and its name.
        observations: Its own index rows, the fewest values of those covering
            more than one instrument, so a check stays short rather than taking
            the largest feature of the build.
    """
    covering = {
        identity: held
        for identity, held in standing.items()
        if len({one.instrument for one in held}) > 1
    }
    return min(
        (covering or standing).items(),
        key=lambda one: sum(math.prod(held.shape) for held in one[1]),
    )


def _by_instrument(patches: list[Patch]) -> list[str]:
    """Return one line per instrument, saying what its patches came out as.

    Args:
        patches: Every patch one sample cut into.

    Returns:
        lines: One line an instrument, in the order they were drawn.
    """
    standing: dict[str, list[Patch]] = {}
    for one in patches:
        standing.setdefault(one.instrument, []).append(one)
    lines = []
    for instrument, held in standing.items():
        first = held[0]
        spectral = first.wavelengths_nm
        bands = "" if spectral is None else f"  {spectral.size} bands"
        ground = tuple(round(one) for one in first.ground_sample_m)
        measured = sum(one.measured_share for one in held) / len(held)
        lines.append(
            f"  {instrument:<7} {len(held):>6,} patches  {first.values.shape}  "
            f"{first.values.dtype}  gsd {ground} m  measured {measured:.0%}  "
            f"at {first.lon:.3f}, {first.lat:.3f}{bands}"
        )
    return lines


def main() -> int:
    """Read one build and one small sample of it, and say what both hold.

    Returns:
        code: A process exit code, non zero when the build holds no crop.
    """
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument("--build", default=None, help="the build to read")
    parsed.add_argument(
        "--per-instrument",
        type=int,
        default=1,
        help="how many observations of each instrument the sample draws",
    )
    arguments = parsed.parse_args()
    return describe_build(arguments.build, arguments.per_instrument)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
