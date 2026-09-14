"""What one channel of one instrument is, beside the numbers that channel holds."""

from __future__ import annotations

import math

import torch
from building.common.layout import ELEVATION, WAVELENGTH
from torch import Tensor, nn

from dataset.patches import channel_axis

# What a channel of each kind of axis measures, and the periods it is read at: the
# nanometres a band is centred on, and the metres above the areoid a sounder reads
# one delay at. The shortest period is twice the finest step between two
# neighbouring channels, since a shorter one folds the two onto the same phase:
# CRISM bands come as close as 3.2 nm where its two detectors overlap, and a SHARAD
# delay stands 5.6 m from the one above it. The longest is the whole range the
# measurement runs over. An instrument whose patch is ground alone holds one
# channel and nothing to tell it apart by.
CHANNEL_RUN = {WAVELENGTH: (6.0, 2700.0), ELEVATION: (11.0, 30_000.0)}


def by_channel(held: Tensor, at: int | None) -> Tensor:
    """Return one batch of patches as the ground samples of each of their channels.

    Args:
        held: The patches, in the instrument's own axis order. (B, K, *P)
        at: Which axis of a patch its channels run along, or None for one that
            holds a single channel.

    Returns:
        held: The same, its channels along one axis and their ground flattened
            behind it. (B, K, C, G)
    """
    if at is None:
        return held.flatten(2).unsqueeze(2)  # (B, K, 1, G)
    return held.movedim(2 + at, 2).flatten(3)  # (B, K, C, G)


class ModalityEncoder(nn.Module):
    """What each channel of one instrument is, as the vector its token is enriched by.

    A token carries the numbers a channel holds and nothing saying what they
    are. This says it: which instrument read them, and what that channel of it
    measures, as one vector. A band of a survey is 2 microns and the one beside
    it 2.01; a delay of a radargram stands 400 metres lower than the one above
    it; a panchromatic scan has one channel and nothing to tell apart. One set
    of weights can then read every channel of an instrument without any two of
    them arriving the same.

    Attributes:
        mark: What the instrument is, the same under every channel of it. (D)
        periods: What each sinusoid repeats over, in the unit the channels are
            measured in, spaced evenly in the log over the range they run, and
            None for an instrument whose channels measure nothing. (D / 2)
    """

    def __init__(self, axes: tuple[str, ...], dim: int) -> None:
        """Build the encoding for one instrument at one token width.

        Args:
            axes: What each axis of the instrument's values holds, which says
                what its channels measure and so what they are read against.
            dim: The token width, even so each period gets a cosine and a sine.
        """
        super().__init__()
        self.mark = nn.Parameter(torch.zeros(dim))  # (D)
        nn.init.normal_(self.mark, std=0.02)
        at = channel_axis(axes)
        run = CHANNEL_RUN.get(axes[at]) if at is not None else None
        if run is None:
            self.periods = None
            return
        shortest, longest = run
        self.register_buffer(
            "periods",
            torch.logspace(math.log10(shortest), math.log10(longest), dim // 2),
        )  # (D / 2)

    def forward(self, channels: Tensor) -> Tensor:
        """Return what each channel of each patch is.

        Args:
            channels: What each channel of each patch measures, in its own unit.
                (B, K, C)

        Returns:
            encoded: The instrument's own vector, and under an instrument whose
                channels measure something the cosines then the sines of each
                over the periods added to it. (B, K, C, D)
        """
        if self.periods is None:
            return self.mark.expand(*channels.shape, -1)  # (B, K, C, D)
        phase = 2 * math.pi * channels.unsqueeze(-1) / self.periods  # (B, K, C, D / 2)
        return self.mark + torch.cat([phase.cos(), phase.sin()], dim=-1)  # (B, K, C, D)
