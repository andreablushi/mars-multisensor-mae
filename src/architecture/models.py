"""What one stage hands the next: a sensor's patches, and the cells they reach."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

CELL_KEY = 1 << 20


@dataclass(frozen=True, slots=True)
class Tokens:
    """One sensor's patches over a batch of features, padded to one count.

    Attributes:
        values: The normalised patches, zero where padded. (B, K, *P)
        valid: Whether each sample is a measurement. (B, K, *P')
        position: The patch centre and its span, in metres. (B, K, 6)
        channels: What each channel of each patch measures, in its own unit. (B, K, C)
        visible: Whether a slot holds a patch its encoder may read. (B, K)
        present: Whether each slot holds a patch rather than padding. (B, K)
    """

    values: Tensor
    valid: Tensor
    position: Tensor
    channels: Tensor
    visible: Tensor
    present: Tensor

    def to(self, device: torch.device) -> Tokens:
        """Return the same tokens held on one device."""
        # Transfer all underlying data and mask tensors to target execution device
        return Tokens(
            self.values.to(device),
            self.valid.to(device),
            self.position.to(device),
            self.channels.to(device),
            self.visible.to(device),
            self.present.to(device),
        )


@dataclass(frozen=True, slots=True)
class Cells:
    """The cells a batch of features is cut into, and which instrument reaches each.

    A cell is a square of ground the same size for every feature, so a cell
    offset stands for the same place whichever feature holds it and two
    features are compared over the offsets they share. How many cells a
    feature holds is its own, since a patch reaches every cell its span covers.

    Attributes:
        offset: Which cell each slot stands for, east then north. (B, Q, 2)
        covered: Whether each instrument reaches the cell, keyed as ODE names it. (B, Q)
        present: Whether a slot holds a cell rather than padding. (B, Q)
    """

    offset: Tensor
    covered: dict[str, Tensor]
    present: Tensor

    def to(self, device: torch.device) -> Cells:
        """Return the same cells held on one device."""
        return Cells(
            self.offset.to(device),
            {name: one.to(device) for name, one in self.covered.items()},
            self.present.to(device),
        )


@dataclass(frozen=True, slots=True)
class FeatureGrid:
    """A batch of features as a grid of cells, each standing for the ground it covers.

    Attributes:
        values: The cell vectors, of unit length where occupied. (B, Q, D)
        occupied: Whether an instrument it was built from reaches the cell. (B, Q)
        offset: Which cell each slot stands for, east then north. (B, Q, 2)
    """

    values: Tensor
    occupied: Tensor
    offset: Tensor


def reached_cells(position: np.ndarray, cell_m: float) -> np.ndarray:
    """Return every cell one instrument's patches reach, without repeats.

    Args:
        position: Where each patch sits and how far it reaches, in metres. (K, 6)
        cell_m: How far a cell runs along the ground, in metres.

    Returns:
        cells: The east and north index of each cell reached, sorted. (C, 2)
    """
    if not len(position):
        return np.zeros((0, 2), np.int64)
    centre, span = position[:, :2], position[:, 3:5]  # (K, 2)
    low = np.floor((centre - span / 2) / cell_m).astype(np.int64)  # (K, 2)
    high = np.floor((centre + span / 2) / cell_m).astype(np.int64)  # (K, 2)
    spread = [
        np.stack(
            np.meshgrid(
                np.arange(east, east_end + 1),
                np.arange(north, north_end + 1),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2)
        for (east, north), (east_end, north_end) in zip(low, high, strict=True)
    ]
    return np.unique(np.concatenate(spread), axis=0)  # (C, 2)


def cell_keys(cells: np.ndarray) -> np.ndarray:
    """Return one number standing for each cell, so two sets of them can be matched.

    Args:
        cells: The east and north index of each cell. (C, 2)

    Returns:
        keys: One number per cell, telling it from every other. (C,)
    """
    return cells[:, 0] * CELL_KEY + cells[:, 1]  # (C,)


def collate(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], tuple[str, str]]],
    cell_m: float,
) -> tuple[dict[str, Tokens], Cells, list[tuple[str, str]]]:
    """Return one batch of every instrument's tokens, the cells they reach, and whose.

    Args:
        samples: What one read of each feature holds, and the feature it belongs to.
        cell_m: How far a cell runs along the ground, in metres.

    Returns:
        batch: Each instrument's patches over the batch, keyed as ODE names it.
        cells: The cells those patches reach, and which instrument reaches each.
        identities: The feature each read belongs to, in the batch's own order.
    """
    batch = {}
    # Process each instrument/sensor present in the first dataset sample
    for name in samples[0][0]:
        held = [sample[name] for sample, _ in samples]
        # Count number of active patches per sample to find max sequence length K
        counts = torch.tensor([len(one["values"]) for one in held])  # (B,)
        slots = torch.arange(int(counts.max()))  # (K,)
        # Pad variable-length patch arrays across batch samples to uniform length K
        padded = {
            key: pad_sequence(
                [torch.as_tensor(one[key]) for one in held], batch_first=True
            )
            for key in held[0]
        }
        # Construct boolean mask identifying real patches versus zero-padded slots
        present = slots.unsqueeze(0) < counts.unsqueeze(1)  # (B, K)
        # Instantiate Tokens container (defaulting visible patches to present patches)
        batch[name] = Tokens(**padded, visible=present, present=present)
    return batch, feature_cells(samples, cell_m), [one for _, one in samples]


def feature_cells(
    samples: list[tuple[dict[str, dict[str, np.ndarray]], tuple[str, str]]],
    cell_m: float,
) -> Cells:
    """Return the cells each read of the batch reaches, padded to one count.

    Args:
        samples: What one read of each feature holds, and the feature it belongs to.
        cell_m: How far a cell runs along the ground, in metres.

    Returns:
        cells: One slot per cell reached, saying where it sits and who reaches it.
    """
    every: list[Tensor] = []
    covered: list[dict[str, Tensor]] = []
    for sample, _ in samples:
        reached = {
            name: reached_cells(held["position"], cell_m)
            for name, held in sample.items()
        }
        held = np.unique(np.concatenate(list(reached.values())), axis=0)  # (Q, 2)
        keys = cell_keys(held)  # (Q,)
        every.append(torch.as_tensor(held))
        covered.append(
            {
                name: torch.as_tensor(np.isin(keys, cell_keys(one)))
                for name, one in reached.items()
            }
        )
    counts = torch.tensor([len(one) for one in every])  # (B,)
    slots = torch.arange(int(counts.max()))  # (Q,)
    return Cells(
        offset=pad_sequence(every, batch_first=True),  # (B, Q, 2)
        covered={
            name: pad_sequence([one[name] for one in covered], batch_first=True)
            for name in samples[0][0]
        },  # (B, Q)
        present=slots.unsqueeze(0) < counts.unsqueeze(1),  # (B, Q)
    )
