"""STUB - not ported yet: ``mtp_draft_loader``.

What it does: Speculative decoding (SGLang NEXTN) in the rollout engines for models with an MTP head, plus the draft-head weight-loading fix needed so the draft head follows weight syncs.

Model family: models with an MTP head.  Frameworks: slime.
Status, source location and what is left: docs/PORTING.md (section ``mtp_draft_loader``).
"""
from . import not_ported

NAME = "mtp_draft_loader"


def install(*args, **kwargs):
    not_ported(NAME)
