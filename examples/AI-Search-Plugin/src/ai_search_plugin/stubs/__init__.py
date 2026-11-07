"""Registry stubs for components that are NOT ported yet (GPU / Megatron / kernel heavy).

Each stub module documents what the component does and where its source lives; ``install()`` raises
:class:`NotPortedError`.  See docs/PORTING.md for what is left to port.
"""


class NotPortedError(NotImplementedError):
    pass


def not_ported(name: str):
    raise NotPortedError(f"ai_search_plugin component {name!r} is a stub (not ported yet); see docs/PORTING.md")
