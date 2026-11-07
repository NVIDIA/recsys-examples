"""AI-Search-Plugin: opt-in speed / correctness components for agentic-RL post-training (slime, verl).

    import ai_search_plugin as asp
    plan = asp.install(framework="slime", components="default")       # or a list of names
    print(plan.shell())                                                 # env + CLI arguments to use

    asp.list_components()                                               # the registry (name, category, status, ...)

Nothing heavy is imported here: torch / slime / verl / SGLang / Megatron are imported lazily inside the component
modules, in the processes where they patch something.
"""
from __future__ import annotations

from typing import Optional

from .registry import COMPONENTS, Component, get, list_components, resolve  # noqa: F401

__version__ = "0.1.0"
__all__ = ["install", "plan", "list_components", "get", "resolve", "COMPONENTS", "Component", "__version__"]


def plan(framework: str, components="default", options: Optional[dict] = None):
    """Return the :class:`ai_search_plugin.frameworks.Plan` (env, framework arguments, launcher) without side effects."""
    if framework == "slime":
        from .frameworks import slime as F
    elif framework == "verl":
        from .frameworks import verl as F
    else:
        raise ValueError(f"framework must be 'slime' or 'verl', got {framework!r}")
    return F.plan(components, options)


def install(framework: str, components="default", options: Optional[dict] = None, environ=None, **kw):
    """Enable ``components`` for ``framework`` and return the plan.

    framework: ``"slime"`` or ``"verl"``.
    components: ``"default"`` | ``"all"`` | ``"none"`` | list / comma string of names; ``"-name"`` removes one
        (``"default,-tool_hedge"``).  Stubs (not yet ported) are refused when named.
    options: framework options (see ``ai_search_plugin.frameworks.<framework>.plan``).

    Writes the plan's env into ``environ`` (default ``os.environ``).  The patches themselves are installed inside the
    framework's own processes through the hooks in ``plan.args`` (slime) or ``VERL_USE_EXTERNAL_MODULES`` (verl);
    pass ``patch_now=True`` for verl to also patch the current process.
    """
    if framework == "slime":
        from .frameworks import slime as F
        return F.install(components, options, environ)
    if framework == "verl":
        from .frameworks import verl as F
        return F.install(components, options, environ, **kw)
    raise ValueError(f"framework must be 'slime' or 'verl', got {framework!r}")
