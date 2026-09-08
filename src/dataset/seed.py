"""The number a feature is drawn and split by, which it is always given again."""

from __future__ import annotations

import hashlib


def seeded_number(text: str, seed: int = 0) -> int:
    """Return the number one name is always given, whichever process asks for it.

    Args:
        text: What is being numbered, such as a feature's class and its name.
        seed: A number mixed into it, so one name can be numbered differently
            for one purpose than for another.

    Returns:
        number: A number read from the name alone, the same in every process and
            every run, unlike the built in hash of a string.
    """
    digest = hashlib.blake2b(
        text.encode("utf-8"), key=str(seed).encode("utf-8"), digest_size=8
    )
    return int.from_bytes(digest.digest())
