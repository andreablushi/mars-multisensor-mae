"""Finding, for each query box, the keys of its own tile that overlap it."""

from __future__ import annotations

from collections.abc import Iterator

from torch import Tensor

from architecture.grid import covering_indices, overlapping_boxes

# N = queries, n = queries of one chunk, B = tiles, S = keys of a tile,
# M = keys one query reads.


def neighbourhoods(
    boxes: Tensor, tiles: Tensor, keys: Tensor, usable: Tensor, size: int
) -> Iterator[tuple[slice, Tensor, Tensor, Tensor]]:
    """Yield, a chunk of queries at a time, the keys each query's box overlaps.

    Args:
        boxes: Where each query sits and how far it reaches. (N, 6)
        tiles: Which tile each query belongs to. (N,)
        keys: Where each key of each tile sits and how far it reaches. (B, S, 6)
        usable: Which keys may be read. (B, S)
        size: How many queries one chunk holds, so a chunk fits in memory.

    Yields:
        rows: Which queries the chunk holds.
        at: The index of each overlapping key in its tile, then of padding. (n, M)
        chosen: Whether each index is a key rather than padding. (n, M)
        ignored: The padding attention skips, none in a row nothing reaches. (n, M)
    """
    # Walk the queries in chunks of the given size
    for start in range(0, len(boxes), size):
        # The queries this chunk holds
        rows = slice(start, start + size)
        # The tile each of them belongs to
        tile = tiles[rows]  # (n,)
        # Which keys of its own tile each query's box overlaps, among the usable ones
        reaching = overlapping_boxes(boxes[rows, None], keys[tile], 3) & usable[tile]
        # Those keys as padded indices, with which are real and which to skip
        yield rows, *covering_indices(reaching, keys.dtype)
