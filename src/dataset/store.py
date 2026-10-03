"""The published build of the dataset, and what its objects hold."""

from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from analysis import paths as labelled
from analysis.ground_truth.artifacts import LABELS
from analysis.ground_truth.models.label import Label
from building import paths as built
from building.common.layout import Axis
from building.metadata.dataset import DatasetManifest
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common.store import EAST, MEASURED, META, NORTH
from common.disk import parquet

from dataset.models.observation import Observation


@dataclass(slots=True)
class DatasetBuild:
    """One published build of the dataset, read one object at a time.

    Attributes:
        root: Where the build sits on this machine.
        fetch: How one object is brought down when the root holds none of it.
        records: What every observation is, once the index is read, else None.
        normalization: The mean and std of each instrument, once the manifest is read,
            else None.
    """

    root: Path
    fetch: Callable[[str], bytes]
    records: list[ObservationMetadata] | None = None
    normalization: dict[str, dict[str, Any]] | None = None

    def read_object(self, path: str) -> bytes:
        """Return what one object of the build holds, off disk or fetched.

        Args:
            path: Where it sits, relative to the build root, as the index names it.

        Returns:
            data: The bytes of that object.
        """
        held = self.root / path
        if held.is_file():
            return held.read_bytes()
        return self.fetch(path)

    def keep(self, path: str, data: bytes) -> None:
        """Keep one file under the root.

        Args:
            path: Where it goes, relative to the build root.
            data: What it holds.
        """
        held = self.root / path
        held.parent.mkdir(parents=True, exist_ok=True)
        # Written whole then moved, so a reader beside this one finds it finished.
        temporary = held.with_suffix(f"{held.suffix}.{os.getpid()}")
        temporary.write_bytes(data)
        temporary.replace(held)

    def read_table(self, path: str, schema: pa.Schema | None = None) -> pa.Table:
        """Return one parquet object of the build, read out of the bytes it holds.

        Args:
            path: Where it sits, relative to the build root.
            schema: What to read it under, or None to take the file's own.

        Returns:
            table: The rows it holds.
        """
        return pq.read_table(io.BytesIO(self.read_object(path)), schema=schema)

    def read_observation_metadata(self) -> list[ObservationMetadata]:
        """Return what every observation of the build is, reading the index once.

        Returns:
            records: One row per observation, in the order the index holds them.
        """
        if self.records is None:
            held = self.read_table(built.OBSERVATION_METADATA_NAME)
            self.records = [
                parquet.built_row(ObservationMetadata, row) for row in held.to_pylist()
            ]
        return self.records

    def read_normalization(self) -> dict[str, dict[str, Any]]:
        """Return the mean and std each instrument is standardised by, read once.

        Returns:
            normalization: The constants of each instrument, per band where it has
                bands, an instrument left unscaled absent.

        Raises:
            ValueError: When the build records no normalization.
        """
        if self.normalization is None:
            held = json.loads(self.read_object(built.DATASET_MANIFEST_NAME))
            normalization = DatasetManifest(**held).normalization
            if normalization is None:
                raise ValueError(f"{self.root.name} records no normalization.")
            self.normalization = normalization
        return self.normalization

    def read_observation_metadata_by_tile(
        self,
    ) -> dict[str, dict[str, list[ObservationMetadata]]]:
        """Return the observations of each tile, by the instrument that took them.

        Returns:
            standing: The rows of each sensor of each tile, keyed by identity.
        """
        standing = defaultdict(lambda: defaultdict(list))
        for one in self.read_observation_metadata():
            standing[one.tile][one.instrument].append(one)
        return {tile: dict(rows) for tile, rows in standing.items()}

    def read_row_by_instrument(self) -> dict[str, ObservationMetadata]:
        """Return one index row of each instrument the build reached.

        Returns:
            rows: One row per sensor, saying what each axis holds and how far it runs.
        """
        return {one.instrument: one for one in self.read_observation_metadata()}

    def read_axes_by_instrument(self) -> dict[str, tuple[str, ...]]:
        """Return what each axis of each instrument's values holds.

        Returns:
            axes: The axes of one sensor's values, in the array's own order, per sensor.
        """
        return {name: one.axes for name, one in self.read_row_by_instrument().items()}

    def read_resolution_by_instrument(self) -> dict[str, float]:
        """Return how much ground one sample of each instrument spans, over the build.

        Returns:
            resolution_m: One length per sensor, its median finest ground axis.
        """
        standing = defaultdict(list)
        for one in self.read_observation_metadata():
            standing[one.instrument].append(min(one.sample_spacing_m))
        return {name: float(np.median(held)) for name, held in standing.items()}

    def read_label_by_tile(self) -> dict[str, str]:
        """Return the class the feature catalogue gave every tile the draw took.

        Returns:
            classes: The class each tile earned, keyed by the tile it was drawn for.
        """
        held = self.read_table(labelled.LABELS_NAME, LABELS)
        labels = [parquet.built_row(Label, row) for row in held.to_pylist()]
        return {one.tile: one.label for one in labels}

    def read_observation(self, path: str, beside: Sequence[str] = ()) -> Observation:
        """Return one stored observation, read out of the object it was written as.

        Args:
            path: Where that object sits, as the index names it.
            beside: What else the instrument stores to read, none of it by default.

        Returns:
            observation: The observation, its values standardised by the build's own
                constants and the rest as the build wrote them.
        """
        with np.load(io.BytesIO(self.read_object(path))) as held:
            # What the observation is, is stored beside its arrays as one json string.
            described = json.loads(str(held[META]))
            # Only these are read, so what is not asked for stays packed.
            arrays = {
                name: held[name]
                for name in (described["measurement"], MEASURED, NORTH, EAST, *beside)
            }
        values = arrays[described["measurement"]]
        axes = tuple(described["axes"])
        constants = self.read_normalization().get(described["instrument"])
        if constants is not None:
            # Per band constants run along the wavelength axis, a scalar along none.
            shape = [-1 if holds == Axis.WAVELENGTH else 1 for holds in axes]
            mean, std = (
                np.asarray(constants[name], np.float32).reshape(shape)
                for name in ("mean", "std")
            )
            values = (values - mean) / std
        return Observation(
            values=values,
            axes=axes,
            measured=arrays[MEASURED],
            north=arrays[NORTH],
            east=arrays[EAST],
            beside={name: arrays[name] for name in beside},
            described=described,
        )
