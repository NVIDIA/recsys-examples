# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""slime glue for ``hybridep_overlap`` (HybridEP MoE dispatch + 2-chunk compute/communication overlap).

Recipe side (:func:`plan_fragment`, called from ``frameworks.slime.plan``): Megatron arguments that select the
HybridEP flex dispatcher, the ``ASRT_HYBRIDEP*`` env, and this module's :func:`megatron_init` for the
``ASRT_CHAIN_MEGATRON_INIT`` chain of the training actors.

Actor side (:func:`megatron_init`, runs from slime's ``--custom-megatron-init-path`` after torch.distributed is up and
before the model is built, so changes to ``args`` reach the TransformerConfig):

* probes Megatron / DeepEP (:func:`megatron_hybridep.check_environment`);
* HybridEP not usable -> ``ASRT_HYBRIDEP=on``: raise; ``auto``: switch the dispatcher back to ``alltoall`` and
  continue (loudly logged);
* otherwise applies the remaining dispatcher fields to ``args`` (pad uneven inputs, unfused permute, bool routing
  map, optional SM count), pre-sizes the HybridEP buffer and installs the chunked overlap patch (only on a validated
  Megatron version, ``ASRT_HYBRIDEP_CHUNKS`` >= 2 and expert tensor parallel 1; otherwise ``on`` raises and
  ``auto`` keeps the plain HybridEP dispatcher).

The decision depends only on the software stack and the launch arguments, so every training rank takes the same one.

verl: not supported.  verl's Megatron engine builds its TransformerConfig through its own config path and has no
equivalent of slime's init hook in this package; nothing here is wired for verl.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from . import megatron_hybridep as mh

SUPPORTED_FRAMEWORKS = ("slime",)
INIT_HOOK = "ai_search_plugin.adapters.slime_hybridep.megatron_init"


def plan_fragment(options: Optional[dict] = None) -> Dict[str, object]:
    """Env / Megatron args / init-hook chain / notes for ``frameworks.slime.plan``.

    options: ``hybridep_mode`` (auto | on, default auto), ``hybridep_chunks`` (default 2),
    ``hybridep_max_tokens`` (default 32768, multiple of 64), ``hybridep_sms`` (dispatch/combine SMs, default: Megatron's).
    """
    o = dict(options or {})
    env = {"ASRT_HYBRIDEP": str(o.get("hybridep_mode", "auto")),
           "ASRT_HYBRIDEP_CHUNKS": str(int(o.get("hybridep_chunks", mh.DEFAULT_CHUNKS))),
           "ASRT_HYBRIDEP_MAX_TOKENS": str(int(o.get("hybridep_max_tokens", mh.DEFAULT_MAX_TOKENS)))}
    if o.get("hybridep_sms") is not None:
        env["ASRT_HYBRIDEP_SMS"] = str(int(o["hybridep_sms"]))
    spec = mh.parse_spec(env)  # validate now, not in the actors
    if spec.mode == "off":
        raise ValueError("hybridep_mode=off: leave the component out instead")
    notes = ["hybridep_overlap needs Megatron-Core with the HybridEP flex backend and DeepEP built from its hybrid-ep "
             "branch (HybridEPBuffer) in the training image, plus $CUDA_HOME/bin/nvcc at run time; the overlap patch "
             "is validated on Megatron-Core 0.19 only (others keep the plain HybridEP dispatcher in auto mode); "
             f"mode={spec.mode}: " + ("missing pieces raise" if spec.mode == "on" else
                                      "missing pieces switch it off (dispatcher back to alltoall)"),
             f"hybridep_overlap: micro-batches must stay <= {spec.max_tokens} tokens per rank "
             "(ASRT_HYBRIDEP_MAX_TOKENS / option hybridep_max_tokens)"]
    return {"env": env, "args": mh.megatron_args(spec.sms), "chain": [INIT_HOOK], "notes": notes}


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:  # noqa: BLE001
        return True


def _log(msg: str) -> None:
    if _is_rank0():
        print(f"[asrt] hybridep_overlap: {msg}", flush=True)


def _arg_targets(args) -> List[object]:
    """``args`` plus Megatron's global args object when that is a different object."""
    out = [args]
    try:
        import importlib

        g = importlib.import_module("megatron.training.global_vars")
        ga = getattr(g, "_GLOBAL_ARGS", None)
        if ga is not None and ga is not args:
            out.append(ga)
    except Exception:  # noqa: BLE001
        pass
    return out


def _config_fields():
    try:
        import dataclasses
        import importlib

        tc = importlib.import_module(mh._TCONFIG)
        return {f.name for f in dataclasses.fields(tc.TransformerConfig)}
    except Exception:  # noqa: BLE001
        return None


def args_problems(args) -> Dict[str, List[str]]:
    """Launch-argument checks: ``skip`` (not a MoE run -> nothing to do), ``overlap`` (chunked path not applicable)."""
    skip, ovl = [], []
    if not getattr(args, "num_experts", None):
        skip.append("not a MoE model (num_experts unset)")
    if (getattr(args, "expert_tensor_parallel_size", None) or 1) != 1:
        ovl.append("expert tensor parallel > 1")
    if getattr(args, "moe_expert_capacity_factor", None) is not None:
        ovl.append("moe_expert_capacity_factor set")
    if getattr(args, "fp8", None) or getattr(args, "fp4", None):
        ovl.append("fp8/fp4 training")
    return {"skip": skip, "overlap": ovl}


def apply_args(args, spec: mh.Spec, fields=None) -> Dict[str, object]:
    """Write the HybridEP dispatcher configuration into the parsed Megatron args (before the model is built)."""
    o = mh.config_overrides(spec.sms, fields if fields is not None else _config_fields())
    for a in _arg_targets(args):
        for k, v in o.items():
            setattr(a, k, v)
    return o


def revert_args(args) -> bool:
    """auto mode, HybridEP unusable: back to Megatron's alltoall dispatcher (only if hybridep was selected)."""
    changed = False
    for a in _arg_targets(args):
        if getattr(a, "moe_token_dispatcher_type", None) == "flex" and \
                getattr(a, "moe_flex_dispatcher_backend", None) == "hybridep":
            a.moe_token_dispatcher_type = "alltoall"
            changed = True
    return changed


def megatron_init(args=None, environ=None) -> dict:
    """``ASRT_CHAIN_MEGATRON_INIT`` target (every slime training actor).  Returns a small report (also logged)."""
    env = os.environ if environ is None else environ
    spec = mh.parse_spec(env)
    rep: Dict[str, object] = {"mode": spec.mode, "chunks": spec.chunks, "hybridep": False, "overlap": False}
    if spec.mode == "off":
        rep["why"] = "ASRT_HYBRIDEP=off"
        return rep
    ap = args_problems(args)
    if ap["skip"]:
        revert_args(args)
        rep["why"] = "; ".join(ap["skip"])
        _log(f"off ({rep['why']})")
        return rep
    mh.setup_runtime_env(env)
    chk = mh.check_environment(allow_untested=spec.allow_untested, env=env)
    rep["megatron"] = chk["version"]
    if chk["hybridep"]:
        why = "; ".join(chk["hybridep"])
        if spec.mode == "on":
            raise RuntimeError(f"ai_search_plugin hybridep_overlap (ASRT_HYBRIDEP=on): HybridEP not usable: {why}")
        reverted = revert_args(args)
        rep["why"] = why
        _log(f"off, HybridEP not usable ({why}); dispatcher {'reverted to alltoall' if reverted else 'unchanged'}")
        return rep
    rep["args"] = apply_args(args, spec)
    rep["hybridep"] = True
    rep["presized"] = mh.presize_buffer(spec.max_tokens)
    ovl = list(chk["overlap"]) + ap["overlap"]
    if spec.chunks < 2:
        ovl.append("ASRT_HYBRIDEP_CHUNKS < 2")
    if ovl:
        why = "; ".join(ovl)
        if spec.mode == "on" and spec.chunks >= 2:
            raise RuntimeError(f"ai_search_plugin hybridep_overlap (ASRT_HYBRIDEP=on): overlap patch not applicable: "
                               f"{why}")
        rep["why"] = why
        _log(f"HybridEP dispatcher on, chunked overlap off ({why})")
        return rep
    rep["install"] = mh.install(spec.chunks, spec.max_tokens)
    rep["overlap"] = True
    _log(f"HybridEP dispatcher on, {spec.chunks}-chunk overlap installed (max_tokens={spec.max_tokens}, "
         f"Megatron {chk['version']})")
    return rep
