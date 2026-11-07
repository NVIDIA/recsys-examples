"""STUB - not ported yet: ``dsa_replay``.

What it does: Sparse-attention selection replay: record the rollout engine's top-k key selection per token (compact codec + store) and replay it in the training forward so training attends to exactly the rollout's keys.

Model family: DSA-style sparse attention.  Frameworks: slime.
Status, source location and what is left: docs/PORTING.md (section ``dsa_replay``).
"""
from . import not_ported

NAME = "dsa_replay"


def install(*args, **kwargs):
    not_ported(NAME)
