"""The model: what turns a tile's patches into tokens it shares, and back.

Every tensor is annotated with its shape, in these letters: B the tiles of a
batch, K the patch slots of one instrument, S the tokens of the instruments read
together, P the shape of one patch, D the token width.
"""
