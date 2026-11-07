"""STUB - not ported yet: ``gdn_cudnn``.

What it does: cuDNN gated-delta-net (GDN) forward/backward kernels in training in place of the FLA Triton kernels (slime only; the verl use was deprecated).

Model family: GDN linear attention.  Frameworks: slime.
Status, source location and what is left: docs/PORTING.md (section ``gdn_cudnn``).
"""
from . import not_ported

NAME = "gdn_cudnn"


def install(*args, **kwargs):
    not_ported(NAME)
