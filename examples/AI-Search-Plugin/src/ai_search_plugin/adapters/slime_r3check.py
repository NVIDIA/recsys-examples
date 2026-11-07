"""Fast, equally strict R3 (routed-experts) validation for slime.

slime validates every sample's rollout routes before replay (slime/utils/routed_experts.py
``validate_routed_experts_tensor``): shape, no all-zero MoE layer, no partially missing token rows, bounded fully
missing rows, expert ids in range.  It does this with an int64 copy of the whole tensor (plus a fancy-indexed copy of
the MoE layers) per sample - with hundreds of long samples on a many-layer top-k MoE that is several seconds of
the rollout manager's serial handoff.

``fast`` keeps exactly the same rules but computes them without copies: min/max on the original dtype for the range
check, one ``!= 0`` reduction for the missing-row checks, and no layer gather when every layer is a MoE layer.
Any irregularity (out-of-range id, missing layer, partial rows, too many holes, or any row holes at all) is re-checked
by slime's ORIGINAL function, so error messages / warnings are exactly slime's.

Enable: ``ASRT_R3CHECK=fast`` (installed by the asrt rollout-function wrappers and by
ai_search_plugin.adapters.slime_weightsync.megatron_init; ``install()`` can also be called directly).
"""
from __future__ import annotations

import os

_state = {"installed": False}


def _moe_layers(args, num_layers):
    f = getattr(args, "moe_layer_freq", None)
    if isinstance(f, (list, tuple)):
        return [i for i, x in enumerate(f[:num_layers]) if int(x) != 0]
    if isinstance(f, int) and f > 0:
        return [i for i in range(num_layers) if i % f == 0]
    return list(range(num_layers))


def make_fast(orig):
    import torch

    def validate_routed_experts_tensor(experts, args, *, sample_index=None, expected_rows=None):
        num_layers = int(args.num_layers)
        topk = int(args.moe_router_topk)
        if (experts.ndim != 3 or tuple(experts.shape[1:]) != (num_layers, topk) or experts.shape[0] == 0
                or (expected_rows is not None and experts.shape[0] != expected_rows)
                or experts.is_floating_point()):
            return orig(experts, args, sample_index=sample_index, expected_rows=expected_rows)
        moe = _moe_layers(args, num_layers)
        if not moe:
            return None
        r = experts if len(moe) == num_layers else experts[:, moe, :]
        num_experts = int(getattr(args, "num_experts", torch.iinfo(torch.int32).max))
        lo, hi = int(r.min()), int(r.max())          # no int64 copy
        if lo < 0 or hi >= num_experts:
            return orig(experts, args, sample_index=sample_index, expected_rows=expected_rows)
        if topk > 1:
            present = (r != 0).any(dim=-1)           # [T, L] token-layer has a route
            if not bool(present.all()):              # any hole at all -> slime's own checks/messages/warnings
                return orig(experts, args, sample_index=sample_index, expected_rows=expected_rows)
        return None

    validate_routed_experts_tensor.__asrt_fast__ = True
    validate_routed_experts_tensor.__wrapped_orig__ = orig
    return validate_routed_experts_tensor


_AUD = {"samples": 0, "rows": 0, "row_mismatch": 0, "hole_rows": 0, "samples_with_holes": 0, "calls": 0}


def make_audit(inner):
    """ ASRT_R3AUDIT=1: count, per sample, whether the rollout routes cover every token row (rows ==
    expected len(tokens)-1) and how many rows are fully missing (all routed ids 0 across every MoE layer = the engine
    returned no route for that token; slime tolerates them by replaying expert 0).  Speculative decoding must not
    create such rows (NGRAM in this SGLang drops routed_experts for draft tokens).  Logs every ASRT_R3AUDIT_EVERY
    samples (default 272) to stdout and $ASRT_LOG_DIR/r3audit.jsonl; never changes the validation result."""
    import json
    import time

    every = int(os.environ.get("ASRT_R3AUDIT_EVERY", 272))

    def validate_routed_experts_tensor(experts, args, *, sample_index=None, expected_rows=None):
        try:
            if experts.ndim == 3 and experts.shape[0] > 0 and not experts.is_floating_point():
                n = int(experts.shape[0])
                holes = int((experts.reshape(n, -1) == 0).all(dim=1).sum())
                _AUD["samples"] += 1
                _AUD["rows"] += n
                _AUD["hole_rows"] += holes
                _AUD["samples_with_holes"] += holes > 0
                if expected_rows is not None and n != expected_rows:
                    _AUD["row_mismatch"] += 1
                if every and _AUD["samples"] % every == 0:
                    rec = dict(_AUD, t=time.time(), hole_frac=round(_AUD["hole_rows"] / max(1, _AUD["rows"]), 6))
                    print(f"[asrt] r3audit {json.dumps(rec)}", flush=True)
                    d = os.environ.get("ASRT_LOG_DIR")
                    if d:
                        with open(os.path.join(d, "r3audit.jsonl"), "a") as f:
                            f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        return inner(experts, args, sample_index=sample_index, expected_rows=expected_rows)

    validate_routed_experts_tensor.__asrt_fast__ = getattr(inner, "__asrt_fast__", False)
    validate_routed_experts_tensor.__wrapped_orig__ = getattr(inner, "__wrapped_orig__", inner)
    return validate_routed_experts_tensor


def install() -> bool:
    if _state["installed"]:
        return True
    audit = os.environ.get("ASRT_R3AUDIT", "0") == "1"
    if os.environ.get("ASRT_R3CHECK", "") not in ("fast", "1") and not audit:
        return False
    try:
        import slime.utils.routed_experts as RE
    except Exception:
        return False
    if getattr(RE.validate_routed_experts_tensor, "__asrt_fast__", False):
        _state["installed"] = True
        return True
    fast = make_fast(RE.validate_routed_experts_tensor) if os.environ.get("ASRT_R3CHECK", "") in ("fast", "1") \
        else RE.validate_routed_experts_tensor
    if audit:
        fast = make_audit(fast)
    RE.validate_routed_experts_tensor = fast   # module-internal callers (validate_*_for_replay / _value) use this name
    import sys
    for m in list(sys.modules.values()):       # modules that imported the function by name
        try:
            if getattr(m, "validate_routed_experts_tensor", None) is getattr(fast, "__wrapped_orig__", None):
                setattr(m, "validate_routed_experts_tensor", fast)
        except Exception:
            pass
    _state["installed"] = True
    return True
