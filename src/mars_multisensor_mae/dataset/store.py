"""The published build of the dataset, and what its objects hold."""

from __future__ import annotations

import io
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from lavaflow.analysis.artifacts import read_lavaflow_labels
from lavaflow.configs import LAVAFLOW_LABELS_NAME
from lavaflow.models import (
    LavaFlowLabel,
)
from mars_multisensor_dataset.analysis import paths as labelled
from mars_multisensor_dataset.analysis.ground_truth.nomenclature.catalogue import LABELS
from mars_multisensor_dataset.analysis.ground_truth.nomenclature.models.label import (
    Label,
)
from mars_multisensor_dataset.building import paths as built
from mars_multisensor_dataset.building.common.normalize import CLUTTER
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.building.metadata.dataset import read_normalization
from mars_multisensor_dataset.building.metadata.index import read_observation_metadata
from mars_multisensor_dataset.building.metadata.observation import ObservationMetadata
from mars_multisensor_dataset.building.preprocessing.common.store import (
    EAST,
    MEASURED,
    META,
    NORTH,
)
from mars_multisensor_dataset.building.preprocessing.crism.correction import (
    centre_wavelengths,
)
from mars_multisensor_dataset.building.preprocessing.sharad.window import (
    rows_below_atmosphere,
)
from mars_multisensor_dataset.common.disk import parquet
from mars_multisensor_dataset.common.disk.blocks import BLOCKS, OFFSETS, decoded_window
from mars_multisensor_dataset.common.disk.files import atomic_path

from mars_multisensor_mae.dataset.models.observation import Observation, Window
from mars_multisensor_mae.dataset.models.patch import TileSample


@dataclass(slots=True)
class DatasetBuild:
    """One published build of the dataset, read one object at a time.

    Attributes:
        root: Where the build sits on this machine.
        fetch: How one object is brought down when the root holds none of it.
        clips: The low and high each instrument's normalised values are clipped to,
            None leaving a side open.
        sharad_atmosphere_cropping: Whether SHARAD is cropped to the rows below its
            atmosphere before it is patched.
        acquisition: The acquisition angles each instrument's patches carry, as the
            index names them, none when the run reads no angles.
        records: The rows of each sensor of each tile, once the index is read, else
            None.
        stats: The constants of each instrument, once the manifest is read, else
            None.
        windowed: The build some instruments are read from instead, a window
            around each centre at a time, keyed by instrument.
        keep_fetched: Whether an object fetched is written onto the root, so the
            next read finds it there.
    """

    root: Path
    fetch: Callable[[str], bytes]
    clips: Mapping[str, Sequence[float | None]]
    sharad_atmosphere_cropping: bool
    acquisition: Mapping[str, Sequence[str]]
    records: dict[str, dict[str, list[ObservationMetadata]]] | None = None
    stats: dict[str, dict[str, Any]] | None = None
    windowed: dict[str, DatasetBuild] = field(default_factory=dict)
    keep_fetched: bool = False

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
            records: The rows of each sensor of each tile, keyed by tile, a windowed
                instrument's from its own build and a tile it lacks left out.
        """
        if self.records is None:
            root = self.local_root(built.OBSERVATION_METADATA_NAME)
            standing = defaultdict(lambda: defaultdict(list))
            for one in read_observation_metadata(root):
                standing[one.tile][one.instrument].append(one)
            records = {tile: dict(rows) for tile, rows in standing.items()}
            for name, other in self.windowed.items():
                held = other.read_observation_metadata()
                records = {
                    tile: rows | {name: held[tile][name]}
                    for tile, rows in records.items()
                    if name in held.get(tile, {})
                }
            self.records = records
        return self.records

    def read_stats(self) -> dict[str, dict[str, Any]]:
        """Return the constants each instrument is normalised by, read once.

        Returns:
            stats: The constants of each instrument, per band where it has bands.
        """
        if self.stats is None:
            self.stats = read_normalization(
                self.local_root(built.DATASET_MANIFEST_NAME)
            )
        return self.stats | {
            name: other.read_stats()[name] for name, other in self.windowed.items()
        }

    def read_label_by_tile(self) -> dict[str, str]:
        """Return the class the feature catalogue gave every tile the draw took.

        Returns:
            classes: The class each tile earned, keyed by the tile it was drawn for.
        """
        held = self.local_root(labelled.LABELS_NAME) / labelled.LABELS_NAME
        return {one.tile: one.label for one in parquet.read_rows(Label, LABELS, held)}

    def read_lavaflow_crops(self) -> list[LavaFlowLabel]:
        """Return every SHARAD crop of a lava flow build, with the review's verdict.

        Returns:
            crops: Every crop the build holds, a track over a tile, as written.
        """
        return read_lavaflow_labels(self.local_root(LAVAFLOW_LABELS_NAME))

    def read_observation(self, path: str, window: Window | None = None) -> Observation:
        """Return one stored observation, read out of the object it was written as.

        Args:
            path: Where that object sits, as the index names it.
            window: The part of an observation stored as WebP blocks to decode, or
                None for all of it.

        Returns:
            observation: Values and cluttergram as stored, CRISM gaps unmeasured,
                SHARAD's atmosphere cropped if asked, an encoded image decoded.
        """
        stored = self.root / path
        if stored.is_file():
            data = stored.read_bytes()
        else:
            data = self.fetch(path)
            if self.keep_fetched:
                with atomic_path(stored) as written:
                    written.write_bytes(data)
        with np.load(io.BytesIO(data)) as held:
            # What the observation is, is stored beside its arrays as one json string.
            described = json.loads(str(held[META]))
            instrument = described["instrument"]
            sounder = instrument == sharad.LAYOUT.instrument
            encoding = described.get("encoding")
            # Only these are read, so what is not asked for stays packed.
            measured, north, east = (held[name] for name in (MEASURED, NORTH, EAST))
            if encoding is None:
                values = held[described["measurement"]]
            else:
                lines, samples = (
                    (slice(0, size) for size in measured.shape)
                    if window is None
                    else window.slices(described, north, east, measured.shape)
                )
                values = decoded_window(
                    {name: held[name] for name in (BLOCKS, OFFSETS)},
                    encoding,
                    (lines, samples),
                )
                measured, north, east = (
                    measured[lines, samples],
                    north[lines],
                    east[samples],
                )
            clutter = held[sharad.Kind.CLUTTER] if sounder else None
        axes = tuple(described["axes"])
        first_row = 0
        if sounder and self.sharad_atmosphere_cropping:
            rows = rows_below_atmosphere(values, measured)
            first_row = rows.start
            values, clutter = values[rows], clutter[rows]
        if instrument == centre_wavelengths.LAYOUT.instrument:
            # A pixel missing any band is unmeasured, so no band is read as a zero
            measured = measured & np.isfinite(values).all(axis=-1)
        return Observation(
            values=values,
            axes=axes,
            measured=measured,
            north=north,
            east=east,
            clutter=clutter,
            described=described,
            first_row=first_row,
        )

    def normalised_sample(self, sample: TileSample) -> TileSample:
        """Return stored patches as a model reads them, z-scored on their constants.

        Args:
            sample: Every instrument's patches as stored, a tile or a cut of it.

        Returns:
            sample: Values and condition clipped in single precision, per band
                constants along the last axis, CRISM's unmeasured pixels and the
                unmeasured condition zeroed.
        """
        stats, normalised = self.read_stats(), {}
        for name, held in sample.instruments.items():
            clip, constants = self.clips[name], stats[name]
            mean, std = (
                np.asarray(constants[key], np.float32) for key in ("mean", "std")
            )
            values = (held.values - mean) / std
            if name == centre_wavelengths.LAYOUT.instrument:
                values = np.where(held.measured, values, 0.0)
            condition = held.condition
            if condition.shape[1]:
                # Its constants are plain floats, so the half precision is lifted first
                simulated = stats[CLUTTER]
                clutter = (
                    condition.astype(np.float32) - simulated["mean"]
                ) / simulated["std"]
                # The cluttergram is cut as the values, so it is zeroed where they are
                clipped = np.clip(clutter, *clip).reshape(held.values.shape)
                condition = (clipped * held.measured).reshape(len(condition), -1)
            normalised[name] = held._replace(
                values=np.clip(values, *clip), condition=condition
            )
        return TileSample(normalised)
