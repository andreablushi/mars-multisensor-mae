"""The model: what turns a feature's patches into points on a sphere, and back.

Every tensor is annotated with its shape, in these letters: B the features of a
batch, K the patch slots of one instrument, C the slots of the instrument being
read from, P the shape of one patch, D the token width, L the sphere's dimension.
"""
