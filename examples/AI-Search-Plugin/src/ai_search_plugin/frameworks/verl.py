"""verl entry point.

verl imports every module listed in ``VERL_USE_EXTERNAL_MODULES`` from ``verl/__init__.py`` in every process
(driver / TaskRunner, agent-loop workers, actor workers, rollout servers).  The plugin uses that hook:

    VERL_USE_EXTERNAL_MODULES=ai_search_plugin.verl_plugin
    ASRT_VERL=<feature list>        (written by plan(); read by ai_search_plugin.adapters.verl.install)

Each feature is a post-import patch (``adapters._hooks.when_imported``), so verl modules imported later are patched
when they load; nothing in verl or the launch command changes.  These variables must reach every Ray worker: export
them before ``ray start`` / in the job environment, or pass ``plan.full_env()`` as Ray ``runtime_env`` env_vars.
"""
from __future__ import annotations

import os
from typing import Optional

from .. import registry
from . import Plan, apply_env

# component -> ASRT_VERL feature(s)
_FEATURES = {
    "sampler_nosync": ["sglhooks"],
    "weight_sync": ["sglhooks"],
    "verl_inplace": ["inplace"],
    "verl_kvfresh": ["kvfresh"],
    "verl_ceiling": ["ceiling"],
    "toolcache": ["toolcache"],
    "response_logits": ["response_logits"],
    "jitcache": ["jitcache"],
    "expert_placement": ["place"],
    "prefix_tree": [],
}


def plan(components="default", options: Optional[dict] = None) -> Plan:
    """Build the verl recipe.

    options (all optional):
      max_staleness (int, default 2)   staleness ceiling S for verl_ceiling
      tool_patterns (str)              fnmatch patterns of verl tool names to cache (default '*search*,*retriev*')
      jitcache_dir (str)               persistent JIT cache directory (jitcache)
      log_dir (str)                    ASRT_LOG_DIR for per-component jsonl logs
    """
    o = dict(options or {})
    comps = registry.resolve("verl", components)
    names = [c.name for c in comps]
    p = Plan("verl", names)
    env = p.env
    jdir = o.get("jitcache_dir") or os.environ.get("ASRT_JITCACHE_DIR", "")
    if "jitcache" in names and not jdir:
        names.remove("jitcache")
        p.skipped["jitcache"] = "needs option jitcache_dir=<directory>"
    feats = []
    for n in names:
        for f in _FEATURES[n]:
            if f not in feats:
                feats.append(f)
    hooks = []
    if "sampler_nosync" in names:
        hooks.append("nosync")
    if "weight_sync" in names:
        hooks.append("fastload")
    if hooks:
        env["ASRT_SGL_HOOKS"] = ",".join(hooks)
    elif "sglhooks" in feats:
        feats.remove("sglhooks")
    if "verl_ceiling" in names:
        env["ASRT_MAX_STALENESS"] = str(int(o.get("max_staleness", 2)))
    if "verl_kvfresh" in names:
        env["ASRT_VERL_KVFRESH"] = "1"
    if "toolcache" in names and o.get("tool_patterns"):
        env["ASRT_VERL_TOOLCACHE_TOOLS"] = str(o["tool_patterns"])
    if "jitcache" in names:
        env["ASRT_JITCACHE_DIR"] = str(jdir)
    if o.get("log_dir"):
        env["ASRT_LOG_DIR"] = str(o["log_dir"])
    env["ASRT_VERL"] = ",".join(feats) if feats else "off"
    mods = [m for m in os.environ.get("VERL_USE_EXTERNAL_MODULES", "").split(",") if m.strip()]
    if "ai_search_plugin.verl_plugin" not in mods:
        mods.append("ai_search_plugin.verl_plugin")
    if "prefix_tree" in names:
        env["ASRT_PREFIX_TREE"] = "1"
        if "ai_search_plugin.prefix_tree.verl_plugin" not in mods:
            mods.append("ai_search_plugin.prefix_tree.verl_plugin")
    env["VERL_USE_EXTERNAL_MODULES"] = ",".join(mods)
    if "verl_inplace" in names:
        p.notes.append("verl_inplace: requests that span a weight sync see old/mixed/new weights (behaviour logprobs "
                       "stay exact); kvepoch is installed with it")
    p.components = names
    return p


def install(components="default", options: Optional[dict] = None, environ=None, patch_now: bool = False) -> Plan:
    """``plan()`` + write the env into ``environ`` (default ``os.environ``).

    patch_now=True additionally installs the verl patches in THIS process (what ``ai_search_plugin.verl_plugin``
    does in every verl process); use it when you call ``install`` from inside a verl process yourself."""
    p = plan(components, options)
    apply_env(p, environ)
    if patch_now:
        from ..adapters import verl as V
        V.install(p.env["ASRT_VERL"])
    return p
