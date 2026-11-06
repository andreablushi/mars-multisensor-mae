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
from analysis.ground_truth.nomenclature.catalogue import LABELS
from analysis.ground_truth.nomenclature.models.label import Label
from building import paths as built
from building.common.layout import Axis
from building.configs import sharad
from building.metadata.dataset import read_normalization
from building.metadata.index import read_observation_metadata
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common.store import EAST, MEASURED, META, NORTH
from building.preprocessing.crism.correction.centre_wavelengths import LAYOUT
from building.preprocessing.sharad.window import rows_below_atmosphere
from common.disk import parquet
from common.disk.files import atomic_path

from dataset.models.observation import Observation


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
    """

    root: Path
    fetch: Callable[[str], bytes]
    clips: Mapping[str, Sequence[float | None]]
    sharad_atmosphere_cropping: bool
    acquisition: Mapping[str, Sequence[str]]
    records: dict[str, dict[str, list[ObservationMetadata]]] | None = None
    stats: dict[str, dict[str, Any]] | None = None

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

    def read_stats(self) -> dict[str, dict[str, Any]]:
        """Return the constants each instrument is normalised by, read once.

        Returns:
            stats: The constants of each instrument, per band where it has bands.
        """
        if self.stats is None:
            self.stats = read_normalization(
                self.local_root(built.DATASET_MANIFEST_NAME)
            )
        return self.stats

    def read_label_by_tile(self) -> dict[str, str]:
        """Return the class the feature catalogue gave every tile the draw took.

        Returns:
            classes: The class each tile earned, keyed by the tile it was drawn for.
        """
        held = self.local_root(labelled.LABELS_NAME) / labelled.LABELS_NAME
        return {one.tile: one.label for one in parquet.read_rows(Label, LABELS, held)}

    def read_observation(self, path: str) -> Observation:
        """Return one stored observation, read out of the object it was written as.

        Args:
            path: Where that object sits, as the index names it.

        Returns:
            observation: The observation, its values and SHARAD's cluttergram
                normalised by the instrument's own constants and clipped, a CRISM
                pixel missing any band left unmeasured and zeroed, SHARAD cropped
                below its atmosphere when asked, the rest as the build wrote them.
        """
        stored = self.root / path
        data = stored.read_bytes() if stored.is_file() else self.fetch(path)
        with np.load(io.BytesIO(data)) as held:
            # What the observation is, is stored beside its arrays as one json string.
            described = json.loads(str(held[META]))
            instrument = described["instrument"]
            sounder = instrument == sharad.LAYOUT.instrument
            # Only these are read, so what is not asked for stays packed.
            values, measured, north, east = (
                held[name] for name in (described["measurement"], MEASURED, NORTH, EAST)
            )
            clutter = held[sharad.Kind.CLUTTER] if sounder else None
        axes = tuple(described["axes"])
        first_row = 0
        if sounder and self.sharad_atmosphere_cropping:
            rows = rows_below_atmosphere(values, measured)
            first_row = rows.start
            values, clutter = values[rows], clutter[rows]
        # Per band constants run along the wavelength axis, a scalar along none.
        shape = [-1 if holds == Axis.WAVELENGTH else 1 for holds in axes]
        constants = self.read_stats()[instrument]
        mean, std = (
            np.asarray(constants[name], np.float32).reshape(shape)
            for name in ("mean", "std")
        )
        values = (values - mean) / std
        if instrument == LAYOUT.instrument:
            # A pixel missing any band is unmeasured, so no band is read as a zero
            measured = measured & np.isfinite(values).all(axis=-1)
            values = np.where(measured[..., None], values, 0.0)
        clip = self.clips[instrument]
        return Observation(
            values=np.clip(values, *clip),
            axes=axes,
            measured=measured,
            north=north,
            east=east,
            clutter=None if clutter is None else np.clip((clutter - mean) / std, *clip),
            described=described,
            first_row=first_row,
        )
