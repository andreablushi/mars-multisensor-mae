"""The published build of the dataset, and what its objects hold."""

from __future__ import annotations

import io
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from analysis import paths as labelled
from analysis.ground_truth.artifacts import LABELS
from analysis.ground_truth.models.label import Label
from building import paths as built
from building.common.layout import Axis
from building.metadata.dataset import read_normalization
from building.metadata.index import read_observation_metadata
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common.store import EAST, MEASURED, META, NORTH
from common.disk import parquet
from common.disk.files import atomic_path

from dataset.models.observation import Observation


@dataclass(slots=True)
class DatasetBuild:
    """One published build of the dataset, read one object at a time.

    Attributes:
        root: Where the build sits on this machine.
        fetch: How one object is brought down when the root holds none of it.
        records: The rows of each sensor of each tile, once the index is read, else
            None.
        stats: The mean and std of each instrument, once the manifest is read, else
            None.
    """

    root: Path
    fetch: Callable[[str], bytes]
    records: dict[str, dict[str, list[ObservationMetadata]]] | None = None
    stats: dict[str, dict[str, Any]] | None = None

    def fetch_object(self, path: str) -> bytes:
        """Return what one object of the build holds, off disk or fetched.

        Args:
            path: Where it sits, relative to the build root, as the index names it.

        Returns:
            data: The bytes of that object.
        """
        held = self.root / path
        return held.read_bytes() if held.is_file() else self.fetch(path)

    def local_root(self, *names: str) -> Path:
        """Return the build root, the named objects fetched onto it where missing.

        Args:
            names: The objects the dataset repository's own readers open there.

        Returns:
            root: The build root, holding every named object.
        """
        for name in names:
            held = self.root / name
            if not held.is_file():
                with atomic_path(held) as written:
                    written.write_bytes(self.fetch(name))
        return self.root

    def read_observation_metadata(
        self,
    ) -> dict[str, dict[str, list[ObservationMetadata]]]:
        """Return the observations of each tile, by the instrument that took them.

        Returns:
            records: The rows of each sensor of each tile, keyed by tile.
        """
        if self.records is None:
            root = self.local_root(built.OBSERVATION_METADATA_NAME)
            standing = defaultdict(lambda: defaultdict(list))
            for one in read_observation_metadata(root):
                standing[one.tile][one.instrument].append(one)
            self.records = {tile: dict(rows) for tile, rows in standing.items()}
        return self.records

    def read_row_by_instrument(self) -> dict[str, ObservationMetadata]:
        """Return one index row of each instrument the build reached.

        Returns:
            rows: One row per sensor, saying what each axis holds and how far it runs.
        """
        return {
            name: held[-1]
            for rows in self.read_observation_metadata().values()
            for name, held in rows.items()
        }

    def read_axes_by_instrument(self) -> dict[str, tuple[str, ...]]:
        """Return what each axis of each instrument's values holds.

        Returns:
            axes: The axes of one sensor's values, in the array's own order, per sensor.
        """
        return {name: one.axes for name, one in self.read_row_by_instrument().items()}

    def read_stats(self) -> dict[str, dict[str, Any]]:
        """Return the mean and std each instrument is standardised by, read once.

        Returns:
            stats: The constants of each instrument, per band where it has bands, an
                instrument left unscaled absent.

        Raises:
            ValueError: When the build records no normalization.
        """
        if self.stats is None:
            stats = read_normalization(self.local_root(built.DATASET_MANIFEST_NAME))
            if stats is None:
                raise ValueError(f"{self.root.name} records no normalization.")
            self.stats = stats
        return self.stats

    def read_label_by_tile(self) -> dict[str, str]:
        """Return the class the feature catalogue gave every tile the draw took.

        Returns:
            classes: The class each tile earned, keyed by the tile it was drawn for.
        """
        held = self.local_root(labelled.LABELS_NAME) / labelled.LABELS_NAME
        return {one.tile: one.label for one in parquet.read_rows(Label, LABELS, held)}

    def read_observation(self, path: str, beside: Sequence[str] = ()) -> Observation:
        """Return one stored observation, read out of the object it was written as.

        Args:
            path: Where that object sits, as the index names it.
            beside: What else the instrument stores to read, none of it by default.

        Returns:
            observation: The observation, its values standardised by the build's own
                constants and the rest as the build wrote them.
        """
        with np.load(io.BytesIO(self.fetch_object(path))) as held:
            # What the observation is, is stored beside its arrays as one json string.
            described = json.loads(str(held[META]))
            # Only these are read, so what is not asked for stays packed.
            arrays = {
                name: held[name]
                for name in (described["measurement"], MEASURED, NORTH, EAST, *beside)
            }
        axes = tuple(described["axes"])
        return Observation(
            values=standardised_values(
                arrays[described["measurement"]],
                axes,
                self.read_stats().get(described["instrument"]),
            ),
            axes=axes,
            measured=arrays[MEASURED],
            north=arrays[NORTH],
            east=arrays[EAST],
            beside={name: arrays[name] for name in beside},
            described=described,
        )


def standardised_values(
    values: np.ndarray, axes: Sequence[str], constants: Mapping[str, Any] | None
) -> np.ndarray:
    """Return one observation's values standardised by its instrument's constants.

    Args:
        values: The values, as the build stored them.
        axes: What each axis of the values holds.
        constants: The instrument's mean and std, per band where it has bands, or
            None for an instrument left unscaled.

    Returns:
        values: The standardised values, or the stored ones where there are no
            constants.
    """
    # Leave the values as they are when the build has no constants for this instrument.
    if constants is None:
        return values
    # Per band constants run along the wavelength axis, a scalar along none.
    shape = [-1 if holds == Axis.WAVELENGTH else 1 for holds in axes]
    mean, std = (
        np.asarray(constants[name], np.float32).reshape(shape)
        for name in ("mean", "std")
    )
    return (values - mean) / std
