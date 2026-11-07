"""STUB - not ported yet: ``gdn_convfuse_ep``.

What it does: GDN short-conv fusion plus four EP communication items from the training-speed line.

Model family: GDN / MoE.  Frameworks: slime.
Status, source location and what is left: docs/PORTING.md (section ``gdn_convfuse_ep``).
"""
from . import not_ported

NAME = "gdn_convfuse_ep"


def install(*args, **kwargs):
    not_ported(NAME)
