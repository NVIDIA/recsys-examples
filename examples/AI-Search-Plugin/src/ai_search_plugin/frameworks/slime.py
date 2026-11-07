"""slime entry point.

slime accepts user code through command-line hooks; the plugin uses exactly those, never edits slime:

    --custom-megatron-init-path      ai_search_plugin.frameworks.slime.megatron_init   (training actors)
    --rollout-function-path          the outermost rollout wrapper of the selected chain (rollout manager)
    --custom-generate-function-path  ai_search_plugin.adapters.slime_toolcache.generate (only with toolcache)
    PYTHONPATH first entry           ai_search_plugin/adapters/sgl_boot (sitecustomize for the SGLang engines)
    launcher                         python -m ai_search_plugin.adapters.slime_train <slime>/train.py (only with prefetch)

Rollout wrapper chain (inner -> outer), each wrapper reads its inner function from an env variable:
    slime stock fully-async  <-  slime_prefetch (prefetch / pool_fill / r3_transfer / r3pack)
                             <-  slime_admit (ticket_admission, ASRT_ADMIT_INNER)
                             <-  slime_rshard (rshard, ASRT_RS_INNER)

Use ``plan()`` to get the recipe (env + args) for a launcher script, or ``install()`` to also write the env into the
current process (e.g. a Python launcher that then submits the Ray job with ``plan.full_env()`` as runtime_env).
"""
from __future__ import annotations

import os
from typing import Optional

from .. import registry
from . import SGL_BOOT_DIR, SRC_ROOT, Plan, apply_env

STOCK_ROLLOUT = "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
A = "ai_search_plugin.adapters."


def plan(components="default", options: Optional[dict] = None) -> Plan:
    """Build the slime recipe.

    options (all optional):
      max_staleness (int, default 2)          staleness ceiling S for ticket_admission
      toolcache_generate (str)                module.attr of your real slime generate function (toolcache)
      tool_targets (str or list)              module.attr of the read-only tool functions to cache / hedge
      num_experts (int)                       MoE expert count (r3_transfer: int16 vs uint8 transfer dtype)
      jitcache_dir (str)                      persistent JIT cache directory (jitcache)
      log_dir (str)                           ASRT_LOG_DIR for per-component jsonl logs
      ws_inplace (bool, default False)        weight_sync: engines keep generating during the sync (needs
                                              --flush-cache-interval 0); semantic change, see slime_weightsync
      chain_megatron_init (str)               your own --custom-megatron-init-path, called after ours
    """
    o = dict(options or {})
    comps = registry.resolve("slime", components)
    names = [c.name for c in comps]

    def on(n):
        return n in names

    p = Plan("slime", names)
    env = p.env

    def skip(name, why):
        if name in names:
            names.remove(name)
        p.skipped[name] = why

    targets = o.get("tool_targets") or os.environ.get("ASRT_TOOLCACHE_TARGETS", "")
    if not isinstance(targets, str):
        targets = ",".join(targets)
    gen = o.get("toolcache_generate") or os.environ.get("ASRT_TOOLCACHE_GENERATE", "")
    if on("toolcache") and not (gen and targets):
        skip("toolcache", "needs options toolcache_generate=<module.attr> and tool_targets=<module.attr,...>")
    if on("tool_hedge") and not (on("toolcache") and targets):
        skip("tool_hedge", "runs under toolcache and needs tool_targets (read-only tools only)")
    jdir = o.get("jitcache_dir") or os.environ.get("ASRT_JITCACHE_DIR", "")
    if on("jitcache") and not jdir:
        skip("jitcache", "needs option jitcache_dir=<directory>")

    if o.get("log_dir"):
        env["ASRT_LOG_DIR"] = str(o["log_dir"])

    # ---- engine-side hooks (SGLang processes, loaded by the sgl_boot sitecustomize)
    hooks = []
    if on("sampler_nosync"):
        hooks.append("nosync")
    if on("weight_sync"):
        hooks += ["fastload", "fastload_check"]
    if on("kvepoch"):
        hooks.append("kvepoch")
    if on("kvfresh"):
        hooks.append("kvfresh")
    ws_inplace = bool(o.get("ws_inplace")) and on("weight_sync")
    if ws_inplace:
        hooks.append("inplace")
    if hooks:
        env["ASRT_SGL_HOOKS"] = ",".join(dict.fromkeys(hooks))
        p.pythonpath_prepend = [SGL_BOOT_DIR, SRC_ROOT]

    # ---- training actor (megatron_init)
    ws = ["pipeline", "flatten", "parsync", "profile"] if on("weight_sync") else ["off"]
    if ws_inplace:
        ws.append("inplace")
        p.notes.append("ws_inplace: run slime with --flush-cache-interval 0; tokens of a request that spans a sync "
                       "come from old/mixed/new weights (behaviour logprobs stay exact)")
    env["ASRT_WEIGHTSYNC"] = ",".join(ws)
    if on("kvfresh"):
        env["ASRT_WS_KV"] = "fresh"
    if on("jitfix"):
        env["ASRT_JITFIX"] = "eager"
    chain = []
    if on("response_logits"):
        chain.append("ai_search_plugin.response_logits.adapters.slime.megatron_init")
    if on("jitcache"):
        env["ASRT_JITCACHE_DIR"] = str(jdir)
        chain.append(A + "jitcache.megatron_init")
    if on("expert_placement"):
        chain.append(A + "slime_placement.megatron_init")
    if on("r3_ep_plan"):
        chain.append(A + "slime_r3plan.megatron_init")
    if on("prefix_tree"):
        env["ASRT_PREFIX_TREE"] = "1"
        chain.append("ai_search_plugin.prefix_tree.adapters.slime.megatron_init")
    if on("hybridep_overlap"):
        from ..adapters import slime_hybridep
        frag = slime_hybridep.plan_fragment(o)
        env.update(frag["env"])
        p.args += frag["args"]
        chain += frag["chain"]
        p.notes += frag["notes"]
    if o.get("chain_megatron_init"):
        chain.append(str(o["chain_megatron_init"]))
    if chain:
        env["ASRT_CHAIN_MEGATRON_INIT"] = ",".join(chain)
    p.args += ["--custom-megatron-init-path", "ai_search_plugin.frameworks.slime.megatron_init"]

    # ---- rollout manager: wrapper chain
    pf = []
    if on("prefetch"):
        pf += ["overlap", "profile"]
        p.launcher = ["python3", "-m", A + "slime_train"]
    if on("r3_transfer"):
        pf.append("r3c")
        env["ASRT_R3CHECK"] = "fast"
        if o.get("num_experts"):
            env["ASRT_NUM_EXPERTS"] = str(int(o["num_experts"]))
    if pf:
        env["ASRT_PREFETCH"] = ",".join(pf)
    if on("r3pack"):
        env["ASRT_R3PACK"] = "1"
    env["ASRT_POOL"] = "auto" if on("pool_fill") else "0"
    if on("pool_fill"):
        env["ASRT_POOL_FILL"] = "engines"
    fn = STOCK_ROLLOUT
    if names and any(on(x) for x in ("prefetch", "pool_fill", "r3_transfer", "r3pack")):
        fn = A + "slime_prefetch.generate_rollout_fully_async"
    if on("ticket_admission"):
        env["ASRT_MAX_STALENESS"] = str(int(o.get("max_staleness", 2)))
        env["ASRT_ADMIT"] = "cap,eager,profile"
        env["ASRT_ADMIT_INNER"] = fn
        fn = A + "slime_admit.generate_rollout_fully_async"
    if on("rshard"):
        env["ASRT_RS"] = "auto,profile"
        env["ASRT_RS_INNER"] = fn
        fn = A + "slime_rshard.generate_rollout_fully_async"
    if on("prefix_tree"):
        env["PREFIX_TREE_ROLLOUT_INNER"] = fn
        fn = "ai_search_plugin.prefix_tree.adapters.slime_pack.generate_rollout_fully_async"
    if fn != STOCK_ROLLOUT:
        p.args += ["--rollout-function-path", fn]
        p.notes.append("the rollout chain ends in slime's stock fully-async rollout (needs slime's fully-async mode); "
                       "a custom --rollout-function-path of your own is replaced")

    # ---- per-sample generate: tool cache (+ hedge underneath)
    if on("toolcache"):
        env["ASRT_TOOLCACHE_GENERATE"] = gen
        env["ASRT_TOOLCACHE_TARGETS"] = targets
        p.args += ["--custom-generate-function-path", A + "slime_toolcache.generate"]
        if on("tool_hedge"):
            env["ASRT_TOOL_HEDGE"] = "auto"
            env["ASRT_TOOL_HEDGE_TARGETS"] = targets
    p.components = names
    return p


def install(components="default", options: Optional[dict] = None, environ=None) -> Plan:
    """``plan()`` + write the env into ``environ`` (default ``os.environ``).  slime itself is not imported: the
    patches install in the slime processes through the hooks listed in ``plan.args`` / ``plan.launcher``."""
    p = plan(components, options)
    apply_env(p, environ)
    return p


def megatron_init(args=None, *rest, **kwargs):
    """``--custom-megatron-init-path`` target (runs in every slime training actor).

    Runs the weight-sync / resume / jitfix / prefetch / R3-check installers and the
    chained hooks (ASRT_CHAIN_MEGATRON_INIT: response_logits, jitcache, the user's own init)."""
    from ..adapters import slime_weightsync

    return slime_weightsync.megatron_init(args)
