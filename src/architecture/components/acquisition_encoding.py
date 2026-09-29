"""Saying how a patch was taken: where the Sun and the spacecraft stood, and when."""

from __future__ import annotations

import math
from dataclasses import fields

import torch
from building.metadata.acquisition_info import AcquisitionInfo
from torch import Tensor, nn

# A = acquisition quantities, D = token channels.

PERIODS = {
    "incidence_deg": 360.0,
    "emission_deg": 360.0,
    "phase_deg": 360.0,
    "solar_longitude_deg": 360.0,
    "local_solar_time_h": 24.0,
}

NAMES = [one.name for one in fields(AcquisitionInfo)]

ENCODED = [NAMES.index(name) for name in PERIODS]


class AcquisitionEncoding(nn.Module):
    """Each quantity read over its own period, projected, and summed into one vector.

    A quantity an observation does not carry is stood for by a learned vector.

    Attributes:
        periods: What each encoded quantity repeats over, in its own unit. (A)
        project: From the cosine and sine of each quantity to a token. (A, 2, D)
        missing: What stands for each quantity when it is unknown. (A, D)
    """

    def __init__(self, dim: int) -> None:
        """Build the encoding at one token width.

        Args:
            dim: The token width.
        """
        super().__init__()
        self.register_buffer("periods", torch.tensor(list(PERIODS.values())))  # (A)
        self.project = nn.Parameter(torch.zeros(len(PERIODS), 2, dim))  # (A, 2, D)
        nn.init.normal_(self.project, std=0.02)
        self.missing = nn.Parameter(torch.zeros(len(PERIODS), dim))  # (A, D)
        nn.init.normal_(self.missing, std=0.02)

    def forward(self, acquisition: Tensor) -> Tensor:
        """Return the encoding of how each patch was taken.

        Args:
            acquisition: Every AcquisitionInfo quantity in its order, nan if unknown.
                (..., len(AcquisitionInfo))

        Returns:
            encoded: One vector per patch. (..., D)
        """
        held = acquisition[..., ENCODED]  # (..., A)
        unknown = held.isnan()  # (..., A)
        phase = 2 * math.pi * held.nan_to_num() / self.periods  # (..., A)
        waves = torch.stack([phase.cos(), phase.sin()], dim=-1)  # (..., A, 2)
        known = torch.einsum("...ac,acd->...ad", waves, self.project)  # (..., A, D)
        return torch.where(unknown.unsqueeze(-1), self.missing, known).sum(dim=-2)
