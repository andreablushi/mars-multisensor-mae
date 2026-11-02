"""What an evaluation writes: one vector per tile and view, and the drawn tiles."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common.disk import parquet

from configs.paths import (
    ATTENTION_FILE,
    RANGING_FILE,
    RECONSTRUCTION_FILE,
    TOKENS_FILE,
    results_path,
)


@dataclass(frozen=True, slots=True)
class TileVector:
    """One labelled tile, its class and its vector over one pooling and view.

    Attributes:
        tile: The tile.
        label: The class it earned.
        pooling: How the tokens were pooled, CLS or GAP.
        instrument: The instrument the vector stands for, or all of them.
        vector: Its tokens pooled, of unit length.
    """

    tile: str
    label: str
    pooling: str
    instrument: str
    vector: tuple[float, ...]


RESULTS_SCHEMA = parquet.schema_of(TileVector)


def write_results(
    run_name: str,
    classes: Mapping[str, str],
    vectors: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    reconstructions: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    attentions: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
    rangings: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
    tokens: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> Path:
    """Return where one run's evaluation was written, its vectors and drawn tiles.

    Args:
        run_name: What the run is called, which names its results.
        classes: The class each tile earned, keyed by the tile.
        vectors: Each tile's vector, keyed by pooling, then view, then tile.
        reconstructions: Each rebuilt tile's arrays, keyed by what it shows.
        attentions: Each tile's instrument rebuilt from the others whole, keyed by
            what the tile shows and the instrument.
        rangings: Each tile's ranging instrument rebuilt from the other alone, keyed
            alike.
        tokens: Each tile's observations drawn whole with their shared tokens, keyed
            by what the tile shows.

    Returns:
        directory: The run's results, emptied of any earlier evaluation first.
    """
    directory = results_path(run_name).parent
    shutil.rmtree(directory, ignore_errors=True)
    parquet.write_rows(
        [
            TileVector(tile, classes[tile], pooling, view, tuple(vector.tolist()))
            for pooling, by_view in vectors.items()
            for view, by_tile in by_view.items()
            for tile, vector in sorted(by_tile.items())
        ],
        RESULTS_SCHEMA,
        results_path(run_name),
    )
    drawn = {
        **{
            RECONSTRUCTION_FILE.format(place): one
            for place, one in reconstructions.items()
        },
        **{TOKENS_FILE.format(place): one for place, one in tokens.items()},
        **{ATTENTION_FILE.format(*key): one for key, one in attentions.items()},
        **{RANGING_FILE.format(*key): one for key, one in rangings.items()},
    }
    for name, by_instrument in drawn.items():
        np.savez(
            results_path(run_name, name),
            **{
                f"{instrument}/{key}": array
                for instrument, arrays in by_instrument.items()
                for key, array in arrays.items()
            },
        )
    return directory
