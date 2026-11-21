"""One step on the model device: patches hidden, then their target tokens predicted."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import torch
from torch import Tensor
from torch.nn import functional

from mars_multisensor_mae.architecture.mae import CrossSensorMAE
from mars_multisensor_mae.architecture.models.context import Pooling
from mars_multisensor_mae.architecture.models.tokens import Tokens
from mars_multisensor_mae.training.loss import (
    latent_error,
    mim,
    reconstruction_error,
    total,
)


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Return the half precision autocast runs in on a device.

    Args:
        device: Where the model runs.

    Returns:
        dtype: bfloat16 on CPU and GPUs of capability 8.0 or above, float16 below.
    """
    native = device.type != "cuda" or torch.cuda.get_device_capability(device) >= (8, 0)
    return torch.bfloat16 if native else torch.float16


def target_tokens(
    target: CrossSensorMAE, batch: dict[str, Tokens]
) -> dict[str, Tensor]:
    """Return I-JEPA's targets: each patch's token from every patch, layer normed.

    Args:
        target: The target encoder, the model's encoders followed by EMA.
        batch: Each instrument's patches over the batch, on the device.

    Returns:
        targets: Each patch's token as read with its whole crop, without a
            gradient, in float. (B, K, D)
    """
    with torch.no_grad():
        encoded = target.embed_tokens(
            batch, {name: one.present for name, one in batch.items()}
        )
    return {
        name: functional.layer_norm(held.tokens[:, 1:].float(), held.tokens.shape[-1:])
        for name, held in encoded.instruments.items()
    }


def masking_random_correspondence(
    batch: dict[str, Tokens],
    mask_ratio: float,
    generators: Mapping[str, torch.Generator],
    dynamic_range: Sequence[float] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return which patches are read and which hidden, each instrument drawn alone.

    Args:
        batch: Each instrument's patches over the batch, on the device.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        generators: What fixes each instrument's mask, on the device.
        dynamic_range: The range each tile draws each instrument's share from
            instead, uniform, or none for mask_ratio.

    Returns:
        visible: Which patches each encoder reads, drawn for each on its own. (B, K)
        hidden: Which patches are predicted, none of them padding. (B, K)
    """
    visible, hidden = {}, {}
    for name, tokens in batch.items():
        present = tokens.present  # (B, K)
        noise = torch.rand(
            present.shape, generator=generators[name], device=present.device
        )
        # Padding is given the highest noise, so only present patches are hidden.
        order = noise.masked_fill(~present, 2.0).argsort(dim=1)  # (B, K)
        share = mask_ratio
        if dynamic_range is not None:
            low, high = dynamic_range
            drawn = torch.rand(
                (present.shape[0], 1), generator=generators[name], device=present.device
            )
            share = low + (high - low) * drawn  # (B, 1)
        visible[name], hidden[name] = visible_and_hidden_from_order(
            order, present, share
        )
    return visible, hidden


def masking_identical_correspondence(
    batch: dict[str, Tokens],
    mask_ratio: float,
    generators: Mapping[str, torch.Generator],
    cell_m: float,
    independent: Sequence[str],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return which patches are read and which hidden, ground cells hidden whole.

    Args:
        batch: Each instrument's patches over the batch, on the device.
        mask_ratio: The share of each tile's ground cells hidden, and of each
            independent instrument's patches.
        generators: What fixes the ground's and each instrument's draws, on the device.
        cell_m: The side of one ground cell, in metres.
        independent: The instruments drawn patch by patch, as random correspondence
            draws them.

    Returns:
        visible: Which patches each encoder reads. (B, K)
        hidden: Which patches are predicted, none of them padding. (B, K)
    """
    visible, hidden = masking_random_correspondence(
        {name: batch[name] for name in independent if name in batch},
        mask_ratio,
        generators,
    )
    shared = {name: one for name, one in batch.items() if name not in independent}
    cells = {
        name: (one.position[..., :2] / cell_m).floor().long()
        for name, one in shared.items()
    }
    every = torch.cat([one.flatten(0, 1) for one in cells.values()])
    low = every.amin(0)
    width = every.amax(0) - low + 1
    index = {
        name: (one[..., 0] - low[0]) * width[1] + one[..., 1] - low[1]
        for name, one in cells.items()
    }
    covered = torch.zeros(
        (next(iter(cells.values())).shape[0], int(width.prod())),
        dtype=torch.int,
        device=low.device,
    )
    for name, at in index.items():
        covered.scatter_add_(1, at, shared[name].present.int())
    field = torch.rand(covered.shape, generator=generators["ground"], device=low.device)
    reached = covered > 0
    order = field.masked_fill(~reached, 2.0).argsort(dim=1)
    _, dropped = visible_and_hidden_from_order(order, reached, mask_ratio)
    for name, tokens in shared.items():
        drawn = dropped.gather(1, index[name])
        visible[name] = tokens.present & ~drawn
        hidden[name] = tokens.present & drawn
    return visible, hidden


def masking_multi_block(
    batch: dict[str, Tokens],
    mask_ratio: float,
    generators: Mapping[str, torch.Generator],
    blocks: int,
    block_scale: Sequence[float],
    aspect_range: Sequence[float],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Return which patches are read and which hidden, I-JEPA's blocks on the ground.

    Args:
        batch: Each instrument's patches over the batch, on the device.
        mask_ratio: Unread, the blocks settle the share hidden.
        generators: What fixes each instrument's blocks, on the device.
        blocks: How many rectangles each instrument of each tile hides.
        block_scale: The range of each rectangle's share of the crop's extent.
        aspect_range: The range of each rectangle's north over east side, log-uniform.

    Returns:
        visible: Which patches each encoder reads, outside every rectangle. (B, K)
        hidden: Which patches are predicted, centred in a rectangle. (B, K)
    """
    visible, hidden = {}, {}
    for name, tokens in batch.items():
        present = tokens.present  # (B, K)
        ground = tokens.position[..., :2].float()  # (B, K, 2)
        far = torch.finfo(ground.dtype).max
        low = ground.masked_fill(~present.unsqueeze(-1), far).amin(1, keepdim=True)
        high = ground.masked_fill(~present.unsqueeze(-1), -far).amax(1, keepdim=True)
        placed = (ground - low) / (high - low).clamp(min=1.0)  # (B, K, 2)
        drawn = torch.rand(
            (len(present), blocks, 4), generator=generators[name], device=present.device
        )  # (B, M, 4)
        low_scale, high_scale = block_scale
        scale = low_scale + (high_scale - low_scale) * drawn[..., 0]  # (B, M)
        low_aspect, high_aspect = (math.log(bound) for bound in aspect_range)
        aspect = torch.exp(low_aspect + (high_aspect - low_aspect) * drawn[..., 1])
        side = torch.stack([(scale / aspect).sqrt(), (scale * aspect).sqrt()], -1)
        side = side.clamp(max=1.0)  # (B, M, 2)
        corner = (1 - side) * drawn[..., 2:]  # (B, M, 2)
        offset = placed.unsqueeze(2) - corner.unsqueeze(1)  # (B, K, M, 2)
        inside = ((offset >= 0) & (offset <= side.unsqueeze(1))).all(-1).any(-1)
        visible[name], hidden[name] = present & ~inside, present & inside
    return visible, hidden


def visible_and_hidden_from_order(
    order: Tensor, present: Tensor, mask_ratio: float | Tensor
) -> tuple[Tensor, Tensor]:
    """Return which patches are read and which hidden, the first of an order hidden.

    Args:
        order: Each tile's slots in the order they are hidden, padding last. (B, K)
        present: Which slots hold a patch rather than padding. (B, K)
        mask_ratio: The share of each tile's patches hidden, or one per tile. (B, 1)

    Returns:
        visible: Which patches are read. (B, K)
        hidden: Which patches are hidden. (B, K)
    """
    rank = order.argsort(dim=1)  # (B, K)
    drawn = rank < (mask_ratio * present.sum(1, keepdim=True)).floor()  # (B, K)
    return present & ~drawn, present & drawn


def selected_masking(
    strategy: str,
    cell_m: float,
    independent: Sequence[str],
    blocks: int,
    block_scale: Sequence[float],
    aspect_range: Sequence[float],
    dynamic_range: Sequence[float] | None = None,
) -> Callable[..., tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Return the masking_* strategy a run hides patches by, its settings bound.

    Args:
        strategy: The strategy, as CSMAE names its correspondence, or multi_block.
        cell_m: The side of one ground cell identical correspondence hides whole.
        independent: The instruments identical correspondence draws patch by patch.
        blocks: How many rectangles multi block hides per instrument and tile.
        block_scale: The range of each rectangle's share of the crop's extent.
        aspect_range: The range of each rectangle's north over east side.
        dynamic_range: The range random correspondence draws each tile's shares
            from, or none for the fixed ratio.

    Returns:
        masking: The strategy, called with the batch, the ratio and the generators.
    """
    if dynamic_range is not None and strategy != "random_correspondence":
        raise ValueError(f"dynamic masking takes random correspondence, not {strategy}")
    return {
        "random_correspondence": partial(
            masking_random_correspondence, dynamic_range=dynamic_range
        ),
        "identical_correspondence": partial(
            masking_identical_correspondence, cell_m=cell_m, independent=independent
        ),
        "multi_block": partial(
            masking_multi_block,
            blocks=blocks,
            block_scale=block_scale,
            aspect_range=aspect_range,
        ),
    }[strategy]


def batch_step(
    model: CrossSensorMAE,
    target: CrossSensorMAE,
    batch: dict[str, Tokens],
    masking: Callable[..., tuple[dict[str, Tensor], dict[str, Tensor]]],
    mask_ratio: float,
    aligned: Sequence[str],
    pooling: Pooling,
    temperature: float,
    generators: Mapping[str, torch.Generator],
    device: torch.device,
) -> dict[str, Tensor]:
    """Return one batch's loss terms, its hidden patches' target tokens predicted.

    Args:
        model: The model, on the device.
        target: The target encoder, the model's encoders followed by EMA.
        batch: Each instrument's patches over the batch, as the loader hands it.
        masking: The masking_* strategy drawing which patches are read and hidden.
        mask_ratio: The share of each instrument's patches hidden from its encoder.
        aligned: The instruments MIM aligns, none for no MIM.
        pooling: What MIM aligns of each instrument, its GAP or its CLS.
        temperature: What MIM divides every cosine by.
        generators: What fixes each instrument's masks, on the device.
        device: Where the model runs.

    Returns:
        terms: Every loss term over the batch, and the loss.
    """
    batch = {name: tokens.to(device) for name, tokens in batch.items()}
    visible, hidden = masking(batch, mask_ratio, generators)
    with torch.autocast(device.type, dtype=autocast_dtype(device)):
        predictions, pooled = model(batch, visible, hidden, pooling)
        targets = target_tokens(target, batch)
    terms = {
        f"{kind}/{name}": latent_error(
            predictions[kind][name], targets[name], hidden[name]
        )
        for kind in ("latent_umr", "latent_cmr")
        for name in targets
    } | {
        f"{kind}/{name}": reconstruction_error(
            predictions[kind][name], batch[name], hidden[name]
        )
        for kind in ("umr", "cmr")
        for name in batch
    }
    return total(terms, mim(pooled, aligned, temperature))
