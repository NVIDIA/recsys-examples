"""STUB - not ported yet: ``fla_nb_bucket``.

What it does: Pin / bucket the FLA causal_conv1d and l2norm autotune key so varlen RL batches stop recompiling for every new packed length.

Model family: FLA / GDN linear attention.  Frameworks: slime, verl.
Status, source location and what is left: docs/PORTING.md (section ``fla_nb_bucket``).
"""
from . import not_ported

NAME = "fla_nb_bucket"


def install(*args, **kwargs):
    not_ported(NAME)
