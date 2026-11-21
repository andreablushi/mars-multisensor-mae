"""What an evaluation writes: one vector per tile and view, and the drawn tiles."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mars_multisensor_dataset.common.disk import parquet

from mars_multisensor_mae.configs.paths import (
    RANGING_FILE,
    RECONSTRUCTION_FILE,
    RESULTS_FILE,
    results_directory,
)
from mars_multisensor_mae.evaluation.models.vectors import TileVectors
from mars_multisensor_mae.logs.evaluation import EVALUATION, RECONSTRUCTION


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


def write_vectors(
    run_name: str,
    classes: Mapping[str, str],
    vectors: Sequence[TileVectors],
) -> Path:
    """Return where one run's tile vectors were written, any earlier ones removed.

    Args:
        run_name: What the run is called, which names its results.
        classes: The class each tile earned, keyed by the tile.
        vectors: Each pooling's unit tile vectors, rows sorted by tile.

    Returns:
        directory: The folder holding the vectors alone.
    """
    directory = results_directory(run_name, EVALUATION)
    shutil.rmtree(directory, ignore_errors=True)
    parquet.write_rows(
        [
            TileVector(
                tile, classes[tile], pooled.pooling, view, tuple(vector.tolist())
            )
            for pooled in vectors
            for view, held in pooled.views().items()
            for tile, vector in zip(pooled.identities, held, strict=True)
        ],
        RESULTS_SCHEMA,
        directory / RESULTS_FILE,
    )
    return directory


def write_reconstructions(
    run_name: str,
    reconstructions: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
    rangings: Mapping[tuple[str, str], Mapping[str, Mapping[str, np.ndarray]]],
) -> Path:
    """Return where one run's drawn tiles were written, any earlier ones removed.

    Args:
        run_name: What the run is called, which names its results.
        reconstructions: Each tile's instrument masked and rebuilt from the others
            whole, keyed by what the tile shows and the instrument.
        rangings: Each tile's ranging instrument rebuilt from the other alone, keyed
            alike.

    Returns:
        directory: The folder holding the drawn tiles alone.
    """
    directory = results_directory(run_name, RECONSTRUCTION)
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    drawn = {
        **{
            RECONSTRUCTION_FILE.format(*key): one
            for key, one in reconstructions.items()
        },
        **{RANGING_FILE.format(*key): one for key, one in rangings.items()},
    }
    for name, by_instrument in drawn.items():
        np.savez(
            directory / name,
            **{
                f"{instrument}/{key}": array
                for instrument, arrays in by_instrument.items()
                for key, array in arrays.items()
            },
        )
    return directory
