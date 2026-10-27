"""The published build of the dataset, and what its objects hold."""

from __future__ import annotations

import io
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from analysis import paths as labelled
from analysis.ground_truth.catalogue import LABELS
from analysis.ground_truth.models.label import Label
from building import paths as built
from building.common.layout import Axis
from building.configs import sharad
from building.metadata.dataset import read_normalization
from building.metadata.index import read_observation_metadata
from building.metadata.observation import ObservationMetadata
from building.preprocessing.common.store import EAST, MEASURED, META, NORTH
from building.preprocessing.crism.correction.centre_wavelengths import LAYOUT
from common.disk import parquet
from common.disk.files import atomic_path

from architecture.tokens import Tokens, padded_batch
from configs.paths import ready_array_path
from dataset.models.observation import Observation
from dataset.patches import (
    cropped_patch_arrays,
    cut_tile_patches,
    patch_arrays,
    patch_lengths,
)

CONDITIONS = {sharad.LAYOUT.instrument: ("clutter",)}

SHARED_SCALE = {"clutter": "SHARAD"}


@dataclass(slots=True)
class DatasetBuild:
    """One published build of the dataset, read one object at a time.

    Attributes:
        root: Where the build sits on this machine.
        fetch: How one object is brought down when the root holds none of it.
        clips: The low and high each instrument's normalised values are clipped to,
            None leaving a side open.
        conditioned: Whether patches are read with what `CONDITIONS` conditions
            them on.
        acquiring: Whether patches carry the acquisition angles `acquisition` names.
        acquisition: The acquisition angles each instrument's patches may carry, as
            the index names them.
        records: The rows of each sensor of each tile, once the index is read, else
            None.
        stats: The constants of each instrument, once the manifest is read, else
            None.
    """

    root: Path
    fetch: Callable[[str], bytes]
    clips: Mapping[str, Sequence[float | None]]
    conditioned: bool
    acquiring: bool
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

    def conditions(self, name: str) -> tuple[str, ...]:
        """Return what one instrument's patches are conditioned on.

        Args:
            name: The instrument, keyed as ODE names it.

        Returns:
            conditions: The arrays read beside its values, none when unconditioned.
        """
        return CONDITIONS.get(name, ()) if self.conditioned else ()

    def acquired(self, name: str) -> tuple[str, ...]:
        """Return the acquisition angles one instrument's patches carry.

        Args:
            name: The instrument, keyed as ODE names it.

        Returns:
            fields: The angles, as the index names them, none when not acquiring.
        """
        return tuple(self.acquisition[name]) if self.acquiring else ()

    def condition_width(self, name: str, shape: Sequence[int]) -> int:
        """Return how many values each patch of one instrument is conditioned on.

        Args:
            name: The instrument, keyed as ODE names it.
            shape: The shape of one of its patches as the model reads it.

        Returns:
            width: One value per sample of every array it is conditioned on.
        """
        return len(self.conditions(name)) * math.prod(shape)

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

    def read_patch_shapes(
        self, sizes: Mapping[str, Mapping[str, int]]
    ) -> dict[str, tuple[int, ...]]:
        """Return the shape of one patch of each instrument as the model reads it.

        Args:
            sizes: How far a patch of each sensor runs along each axis it is cut on.

        Returns:
            shapes: The shape of one patch of each instrument, keyed as ODE names it.
        """
        rows = {
            name: held[-1]
            for rows in self.read_observation_metadata().values()
            for name, held in rows.items()
        }
        return {
            name: patch_lengths(rows[name].shape, rows[name].axes, size)
            for name, size in sizes.items()
        }

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

    def normalised_values(
        self, values: np.ndarray, axes: Sequence[str], instrument: str
    ) -> np.ndarray:
        """Return one observation's values z-scored by its instrument's constants.

        Args:
            values: The values, as the build stored them.
            axes: What each axis of the values holds.
            instrument: The instrument that took them.

        Returns:
            values: The normalised values.
        """
        # Per band constants run along the wavelength axis, a scalar along none.
        shape = [-1 if holds == Axis.WAVELENGTH else 1 for holds in axes]
        constants = self.read_stats()[instrument]
        mean, std = (
            np.asarray(constants[name], np.float32).reshape(shape)
            for name in ("mean", "std")
        )
        return (values - mean) / std

    def read_label_by_tile(self) -> dict[str, str]:
        """Return the class the feature catalogue gave every tile the draw took.

        Returns:
            classes: The class each tile earned, keyed by the tile it was drawn for.
        """
        held = self.local_root(labelled.LABELS_NAME) / labelled.LABELS_NAME
        return {one.tile: one.label for one in parquet.read_rows(Label, LABELS, held)}

    def ready_array(
        self, identity: str, name: str, size: Mapping[str, int], key: str
    ) -> Path:
        """Return where one cut array of one tile is cached, by what changes its values.

        Args:
            identity: The tile.
            name: The instrument.
            size: How far its patch runs along each axis it is cut on.
            key: Which of its patch arrays.

        Returns:
            path: The array's file, apart for every patch size, condition, clip and
                acquisition.
        """
        setup = (
            sorted(size.items()),
            self.conditions(name),
            self.clips[name],
            self.acquired(name),
        )
        return self.root / ready_array_path(identity, name, setup, key)

    def cache_tile(
        self,
        identity: str,
        rows: Mapping[str, list[ObservationMetadata]],
        sizes: Mapping[str, Mapping[str, int]],
        shapes: Mapping[str, tuple[int, ...]],
    ) -> None:
        """Cut one tile into the run's cache, unless it is there already.

        Args:
            identity: The tile.
            rows: The tile's index rows of each sensor.
            sizes: How far a patch of each sensor runs along each axis it is cut on.
            shapes: The shape of one patch of each instrument as the model reads it.
        """
        if all(
            self.ready_array(identity, name, sizes[name], key).is_file()
            for name in sizes
            for key in Tokens._fields
        ):
            return
        for name, patches in cut_tile_patches(rows, self, sizes).items():
            arrays = patch_arrays(
                patches,
                shapes[name],
                rows[name][0].axes,
                self.condition_width(name, shapes[name]),
                len(self.acquired(name)),
            )
            for key, array in arrays.items():
                with (
                    atomic_path(
                        self.ready_array(identity, name, sizes[name], key)
                    ) as written,
                    written.open("wb") as file,
                ):
                    np.save(file, array)

    def read_batch(
        self,
        identities: Sequence[str],
        sizes: Mapping[str, Mapping[str, int]],
        budget: Mapping[str, int] | None,
        scale: float,
        seed: int | None,
    ) -> tuple[dict[str, Tokens], list[str]]:
        """Return one batch of cached tiles, each cut to the budget around a centre.

        Args:
            identities: The tiles of the batch.
            sizes: How far a patch of each sensor runs along each axis it is cut on.
            budget: How many patches of each instrument a tile keeps, or None to
                read it whole.
            scale: The most the budget is multiplied by, drawn once per batch from 1.
            seed: What fixes each tile's centre, or None to draw a new one each read.

        Returns:
            batch: Each instrument's patches over the batch, keyed as ODE names it.
            identities: The tile each read belongs to, in the batch's own order.
        """
        if budget is not None:
            factor = random.uniform(1.0, scale)
            budget = {name: round(count * factor) for name, count in budget.items()}
        samples = []
        for identity in identities:
            sample = {
                name: {
                    key: np.load(
                        self.ready_array(identity, name, sizes[name], key),
                        mmap_mode="c",
                    )
                    for key in Tokens._fields
                }
                for name in sizes
            }
            if budget is not None:
                draw = random.Random(
                    random.random() if seed is None else f"{seed}/{identity}"
                )
                sample = cropped_patch_arrays(sample, budget, draw)
            samples.append((sample, identity))
        return padded_batch(samples)

    def read_observation(self, path: str, beside: Sequence[str]) -> Observation:
        """Return one stored observation, read out of the object it was written as.

        Args:
            path: Where that object sits, as the index names it.
            beside: What else the instrument stores to read, each z-scored and clipped
                as the instrument `SHARED_SCALE` says it shares a scale with.

        Returns:
            observation: The observation, its values normalised by the build's own
                constants and clipped, a CRISM pixel missing any band left unmeasured
                and zeroed, the rest as the build wrote them.
        """
        stored = self.root / path
        data = stored.read_bytes() if stored.is_file() else self.fetch(path)
        with np.load(io.BytesIO(data)) as held:
            # What the observation is, is stored beside its arrays as one json string.
            described = json.loads(str(held[META]))
            # Only these are read, so what is not asked for stays packed.
            arrays = {
                name: held[name]
                for name in (described["measurement"], MEASURED, NORTH, EAST, *beside)
            }
        axes = tuple(described["axes"])
        instrument = described["instrument"]
        values = self.normalised_values(
            arrays[described["measurement"]], axes, instrument
        )
        measured = arrays[MEASURED]
        if instrument == LAYOUT.instrument:
            # A pixel missing any band is unmeasured, so no band is read as a zero
            measured = measured & np.isfinite(values).all(axis=-1)
            values = np.where(measured[..., None], values, 0.0)
        values = np.clip(values, *self.clips[instrument])
        return Observation(
            values=values,
            axes=axes,
            measured=measured,
            north=arrays[NORTH],
            east=arrays[EAST],
            beside={
                name: np.clip(
                    self.normalised_values(arrays[name], axes, SHARED_SCALE[name]),
                    *self.clips[SHARED_SCALE[name]],
                )
                for name in beside
            },
            described=described,
        )
