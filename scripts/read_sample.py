#!/usr/bin/env python
"""What one build of the dataset holds, and what one small sample of it cuts into."""

from __future__ import annotations

import argparse
import math

from rich.console import Console

from dataset import configs, split, stats
from dataset.models.sample import Sample
from dataset.patches import grid
from dataset.store import artifact, index, load


def describe_build(build_name: str | None, per_instrument: int) -> int:
    """Print what one published build holds, and what one sample of it cuts into.

    Args:
        build_name: The build to read, or None for the one the config names.
        per_instrument: How many observations of each instrument the sample draws.

    Returns:
        code: A process exit code, non zero when the build holds no sample.
    """
    console = Console()
    choices = configs.load(build=build_name, per_instrument=per_instrument)
    build = artifact.opened_build(choices)
    console.print(f"reading [bold]{choices.artifact}[/bold] from {build.bucket}")

    manifest = index.read_manifest(build)
    observations = index.read_observations(build)
    version = manifest.get("version", "unwritten")
    console.print(
        f"built {manifest['built_at']}, version {version}, "
        f"{len(observations):,} crops over {', '.join(manifest['instruments'])}"
    )
    for instrument, held in stats.pooled_statistics(observations).items():
        tiles = choices.tiles_of(instrument)
        counted = sum(
            grid.patch_count(one.shape, one.axes, tiles)
            for one in observations
            if one.instrument == instrument
        )
        banded = "" if held.band_mean is None else f", {len(held.band_mean)} bands"
        console.print(
            f"  {instrument:<7} {counted:>10,} patches   "
            f"mean {held.mean:>12.4g}  sd {held.deviation:>10.4g}{banded}"
        )

    features = index.read_features(build)
    standing = index.observations_by_feature(observations)
    if not standing:
        console.print("[red]the build holds no crop[/red]")
        return 1
    placed = split.split_features(standing, choices)
    counted = {name: len(held) for name, held in placed.items()}
    console.print(f"{len(standing):,} features with crops, split {counted}")

    samples = [
        index.drawn_sample(features[identity].frame, held, choices)
        for identity, held in standing.items()
        if identity in features
    ]
    sample = _smallest_multisensor(samples)
    weight = sum(math.prod(one.shape) for one in sample.observations)
    console.print(
        f"loading [bold]{' '.join(sample.identity)}[/bold], "
        f"{len(sample.observations)} crops of {', '.join(sample.instruments)}, "
        f"{weight / 1e6:.1f}M values"
    )
    for line in _by_instrument(load.load_patches(sample, build, choices)):
        console.print(line)
    return 0


def _smallest_multisensor(samples: list[Sample]) -> Sample:
    """Return the lightest sample that still reaches more than one instrument.

    Args:
        samples: Every sample the index was read into.

    Returns:
        sample: The one holding the fewest values of those covering the most
            instruments a light sample can, so a check stays short rather than
            taking the largest feature of the build.

    Raises:
        ValueError: When there is no sample to pick from.
    """
    if not samples:
        raise ValueError("no sample to pick from")
    reached = max(min(len(one.instruments) for one in samples), 2)
    covering = [one for one in samples if len(one.instruments) >= reached]
    return min(
        covering or samples,
        key=lambda one: sum(math.prod(held.shape) for held in one.observations),
    )


def _by_instrument(patches) -> list[str]:
    """Return one line per instrument, saying what its patches came out as.

    Args:
        patches: Every patch one sample cut into.

    Returns:
        lines: One line an instrument, in the order they were drawn.
    """
    standing: dict[str, list] = {}
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
        code: A process exit code, non zero when the build holds no sample.
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
