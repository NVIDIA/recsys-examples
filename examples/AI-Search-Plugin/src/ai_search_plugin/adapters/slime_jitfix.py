"""Stop torch.compile recompile storms in the Megatron trainer (``ASRT_JITFIX``).

Root cause of "slow bursts" in MoE/GDN training (a few micro-batches at ~10 s instead of 0.5 s, ~60 s per
burst, in the stock slime baseline too): Megatron's ``@jit_fuser`` is
``torch.compile`` (megatron/core/jit.py) on small fused helpers - GDN ``_prepare_input_for_gated_delta_rule``
/ ``_compute_g_and_beta`` / ``_apply_gated_norm``, ``bias_dropout_add_fused_train``, ``swiglu`` /
``weighted_swiglu`` (+ backward), ``_apply_output_gate``.  With dynamic batching every micro-batch has a
new token count; together with grad-mode (full recompute = no_grad forward then grad forward) and
requires_grad flips Dynamo recompiles them again and again (``TORCH_LOGS=recompiles``: "tensor 'qkv' size
mismatch", "GLOBAL_STATE changed: grad_mode", "requires_grad mismatch"; TE rmsnorm graph-breaks add
resume functions), each recompile is an Inductor/Triton compile on the critical path, and PP makes
every other stage wait.  After ``recompile_limit`` (8) per function Dynamo gives up and runs eager,
so the storm decays over steps but never at a predictable time.  Megatron's own
``--disable-jit-fuser`` comes too late: the decorators were applied when the modules were imported.

Modes (comma list in ASRT_JITFIX; empty = off):
  eager    replace every Dynamo-wrapped callable found in loaded ``megatron.*`` modules/classes by its
           original Python function (no compiles at all; fused elementwise kernels become eager ops).
  dynamic  replace them by ``torch.compile(orig, dynamic=True)``: one symbolic-shape graph per
           function (grad-mode variants still compile once each, in step 0).
  limit=N  additionally set torch._dynamo.config.recompile_limit = N (cap on the damage).

Model gate: the storms come from the MoE / GDN fused helpers (swiglu with per-expert probs, GDN
input/gate prep) under dynamic token counts. On a dense model eager only loses the fused elementwise kernels
(measured ~10-12% slower training on 7B/14B dense models), so install(args=...) skips itself when the model has no MoE layers
(args.num_experts) and no GDN / linear-attention layers (HF config); ASRT_JITFIX_FORCE=1 applies it anyway.
"""
from __future__ import annotations

import os
import sys


def _wrapped(obj):
    return callable(obj) and getattr(obj, "_torchdynamo_orig_callable", None) is not None


def model_kind(args) -> str | None:
    """'moe' / 'gdn' / 'moe+gdn' when the model has layers whose fused helpers recompile, 'dense' when it has
    none, None when it cannot tell (no args)."""
    if args is None:
        return None
    moe = bool(getattr(args, "num_experts", None))
    gdn = any(getattr(args, a, None) for a in ("linear_attention_freq", "linear_attention_type",
                                               "experimental_attention_variant"))
    try:
        import json
        with open(os.path.join(getattr(args, "hf_checkpoint", None) or "", "config.json")) as f:
            c = json.load(f)
        c = c.get("text_config", c)
        moe = moe or bool(c.get("num_experts") or c.get("n_routed_experts") or c.get("num_local_experts"))
        gdn = gdn or bool(c.get("linear_num_value_heads") or "linear_attention" in (c.get("layer_types") or []))
    except Exception:
        pass
    kinds = [k for k, on in (("moe", moe), ("gdn", gdn)) if on]
    return "+".join(kinds) if kinds else "dense"


def install(modes: str | None = None, args=None) -> dict:
    modes = modes if modes is not None else os.environ.get("ASRT_JITFIX", "")
    opts = {m.strip() for m in modes.split(",") if m.strip()}
    if not opts:
        return {"active": False}
    kind = model_kind(args)
    if kind == "dense" and os.environ.get("ASRT_JITFIX_FORCE", "0") != "1":
        return {"active": False, "skipped": "dense model (no MoE / GDN layers): eager would only lose fused kernels",
                "model": kind}
    import torch

    rep = {"active": True, "opts": sorted(opts), "replaced": []}
    for o in opts:
        if o.startswith("limit="):
            n = int(o.split("=", 1)[1])
            for k in ("recompile_limit", "cache_size_limit"):
                if hasattr(torch._dynamo.config, k):
                    setattr(torch._dynamo.config, k, n)
            rep["limit"] = n
    mode = "eager" if "eager" in opts else ("dynamic" if "dynamic" in opts else None)
    if mode is None:
        return rep

    made = set()
    cache = {}

    def repl(orig):
        if mode == "eager":
            return orig
        if id(orig) not in cache:  # one compiled wrapper per original (shared by all references)
            cache[id(orig)] = torch.compile(orig, dynamic=True)
            made.add(id(cache[id(orig)]))
        return cache[id(orig)]

    # modules imported after this point pick the new decorator up
    try:
        import functools

        import megatron.core.jit as mjit

        mjit.jit_fuser = mjit.noop_decorator if mode == "eager" else functools.partial(torch.compile, dynamic=True)
    except Exception:
        pass

    seen = set()
    for name, mod in list(sys.modules.items()):
        if not name.startswith("megatron.") or mod is None:
            continue
        for attr, val in list(vars(mod).items()):
            if _wrapped(val) and id(val) not in made and getattr(val, "__module__", name) == name:
                setattr(mod, attr, repl(val._torchdynamo_orig_callable))
                rep["replaced"].append(f"{name}.{attr}")
            elif isinstance(val, type) and val.__module__ == name and id(val) not in seen:
                seen.add(id(val))
                for a, v in list(vars(val).items()):
                    if _wrapped(v) and id(v) not in made:
                        setattr(val, a, repl(v._torchdynamo_orig_callable))
                        rep["replaced"].append(f"{name}.{val.__name__}.{a}")
    # Modules that imported a fused function by name (``from ...fused_bias_swiglu import bias_swiglu_impl``)
    # hold their own reference: patch those too.
    for name, mod in list(sys.modules.items()):
        if not name.startswith("megatron.") or mod is None:
            continue
        for attr, val in list(vars(mod).items()):
            if _wrapped(val) and id(val) not in made:
                setattr(mod, attr, repl(val._torchdynamo_orig_callable))
                rep["replaced"].append(f"{name}.{attr}(ref)")
    rep["mode"] = mode
    rep["model"] = kind
    return rep
