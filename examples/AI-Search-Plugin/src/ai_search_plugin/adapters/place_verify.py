# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert re-placement consistency checks (diagnostic, ASRT_PLACE_VERIFY=1; used by the verl glue).

Fingerprints every expert by a fixed random projection of its weights (one float per expert and parameter
name) and checks, per logical expert:
  * move:   right after a re-placement the expert's fingerprint at its NEW slot equals the one it had at its
            OLD slot (bf16 model weights, the optimizer's fp32 main weights and Adam moments);
  * drift:  between two train steps a logical expert's fingerprint changes only by the optimizer update
            (lr 1e-6: ~1e-5 relative); a swapped / stale expert jumps by O(1);
  * main:   bf16 weights == bf16(main fp32 weights) after the optimizer step (DistributedOptimizer shards);
  * export: per-row fingerprints of the [E, ...] stacked expert / router tensors that leave the trainer
            (rank 0, logical order) drift only by the update between two syncs, and exported router rows
            equal the trainer's router rows in logical order.
All numbers are relative (|delta| / rms over experts) and are logged per layer as the max over experts.
Env: ASRT_PLACE_VERIFY=1 (snapshots), ASRT_PLACE_VERIFY_LOG=<jsonl>, ASRT_PLACE_VERIFY_FWD=N (forward A/B of the
first N forward-only batches while re-placed), ASRT_PLACE_VERIFY_FWD_AA=1 (adds an identity-vs-identity control).
torch is imported lazily (this module imports without it).
"""
import json
import os

_V = {"proj": {}, "prev": None, "export_prev": {}, "export_rec": {}}


def enabled():
    return os.environ.get("ASRT_PLACE_VERIFY") == "1"


def _proj(t):
    """[n, ...] -> [n] float64 projection on a fixed random vector (same vector for the same row size)."""
    import torch
    t = t.detach()
    n = t.shape[0]
    m = t[0].numel()
    key = (m, t.device)
    r = _V["proj"].get(key)
    if r is None:
        g = torch.Generator(device="cpu").manual_seed(1234 + m)
        r = _V["proj"][key] = torch.randn(m, generator=g, dtype=torch.float32).to(t.device)
    return (t.reshape(n, m).float() @ r).double()


def _rel(a, b):
    scale = float(b.pow(2).mean().sqrt()) + 1e-30
    return float((a - b).abs().max()) / scale


def _log(rec):
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    rec = dict(rec, rank=rank)
    path = os.environ.get("ASRT_PLACE_VERIFY_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if rank == 0 or float(rec.get("worst", 0)) > 1e-2:
        print(f"[asrt] place-verify {rec}", flush=True)


def snapshot(mgr):
    """{lid: {"w:<name>": {logical: fp}, "m:<name>/<key>": {...}, "router": [E] logical}} for local experts."""
    import torch
    from .megatron_placement import _expert_params
    import torch.distributed as dist
    me = dist.get_rank(mgr.ep_group) if dist.is_initialized() else 0
    out = {}
    for lid, layer in mgr.layers:
        inv = mgr.physical_to_logical(lid).tolist()
        d = {}
        for name, (params, grouped) in _expert_params(layer.experts).items():
            if grouped:
                fp = _proj(params[0])
                mains = [(params[0], None)]
            else:
                fp = torch.stack([_proj(p.reshape(1, -1))[0] for p in params])
                mains = [(p, j) for j, p in enumerate(params)]
            d["w:" + name] = {inv[me * mgr.n_local + j]: float(fp[j]) for j in range(mgr.n_local)}
            # optimizer side: only where this rank holds the whole param (DP 1 for experts is the common case)
            if mgr.opt_state is not None:
                for p, j in mains:
                    o = mgr.opt_state._owner.get(id(p))
                    if o is None:
                        continue
                    st, rng = mgr.opt_state._local(o, p)
                    if not st or rng is None or rng != (0, p.numel()):
                        continue
                    for k, v in st.items():
                        if not isinstance(v, torch.Tensor) or v.numel() != p.numel():
                            continue
                        v = v.detach().to(p.device).reshape(p.shape)
                        if j is None:
                            f = _proj(v)
                            for jj in range(mgr.n_local):
                                d.setdefault(f"m:{name}/{k}", {})[inv[me * mgr.n_local + jj]] = float(f[jj])
                        else:
                            d.setdefault(f"m:{name}/{k}", {})[inv[me * mgr.n_local + j]] = float(_proj(v.reshape(1, -1))[0])
                            if k == "param":
                                d.setdefault(f"mb:{name}", {})[inv[me * mgr.n_local + j]] = \
                                    float(_proj(v.to(p.dtype).reshape(1, -1))[0])
                    if j is None and "param" in st:
                        f = _proj(st["param"].detach().to(p.device).reshape(p.shape).to(p.dtype))
                        for jj in range(mgr.n_local):
                            d.setdefault(f"mb:{name}", {})[inv[me * mgr.n_local + jj]] = float(f[jj])
        rw = layer.router.weight
        sl = mgr.slot_of[lid].to(rw.device)
        d["router"] = _proj(rw).index_select(0, sl).cpu()
        if mgr.opt_state is not None:
            # router main weights (dense DistributedOptimizer, sharded over DP x CP: rebuilt with a collective that
            # every rank runs in the same layer order)
            full = mgr.opt_state.full(rw)
            if "param" in full:
                d["router_main"] = _proj(full["param"]).index_select(0, sl).cpu()
                d["router_mb"] = _proj(full["param"].to(rw.dtype)).index_select(0, sl).cpu()
        out[lid] = d
    return out


def _cmp(a, b):
    """max relative change per key between two snapshots (common logical experts only)."""
    import torch
    res = {}
    for lid in a:
        if lid not in b:
            continue
        for k, va in a[lid].items():
            vb = b[lid].get(k)
            if vb is None:
                continue
            if k.startswith("router"):
                r = _rel(va, vb)
            else:
                common = sorted(set(va) & set(vb))
                if not common:
                    continue
                x = torch.tensor([va[e] for e in common], dtype=torch.float64)
                y = torch.tensor([vb[e] for e in common], dtype=torch.float64)
                r = _rel(x, y)
            kk = k.split(":")[0] if not k.startswith("m:") else "m:" + k.split("/")[-1]
            res[kk] = max(res.get(kk, 0.0), r)
            res[f"{kk}@L"] = res.get(f"{kk}@L", {})
            res[f"{kk}@L"][lid] = max(res[f"{kk}@L"].get(lid, 0.0), r)
    return res


def _main_vs_bf16(s):
    import torch
    worst, per = 0.0, {}
    for lid, d in s.items():
        if "router_mb" in d:
            r = _rel(d["router_mb"], d["router"])
            per[lid] = max(per.get(lid, 0.0), r)
            worst = max(worst, r)
        for k, v in d.items():
            if not k.startswith("mb:"):
                continue
            w = d.get("w:" + k[3:])
            if not w:
                continue
            common = sorted(set(v) & set(w))
            x = torch.tensor([v[e] for e in common], dtype=torch.float64)
            y = torch.tensor([w[e] for e in common], dtype=torch.float64)
            r = _rel(x, y)
            per[lid] = max(per.get(lid, 0.0), r)
            worst = max(worst, r)
    return worst, per


def _summ(res):
    flat = {k: round(v, 6) for k, v in res.items() if not k.endswith("@L")}
    worst_l = {}
    for k, v in res.items():
        if k.endswith("@L"):
            lw = max(v, key=v.get)
            worst_l[k[:-2]] = (lw, round(v[lw], 6))
    return flat, worst_l


def before_step(mgr, step):
    """Call at train_batch start, before any re-placement."""
    if not enabled():
        return None
    s = snapshot(mgr)
    rec = {"step": step, "phase": "pre"}
    mvb, per = _main_vs_bf16(s)
    rec["main_vs_bf16"] = round(mvb, 6)
    if mvb > 1e-2:
        rec["main_vs_bf16_layers"] = {k: round(v, 4) for k, v in per.items() if v > 1e-2}
    if _V["prev"] is not None:
        flat, wl = _summ(_cmp(s, _V["prev"]))
        rec["drift"] = flat
        rec["drift_worst_layer"] = wl
    exp = _V.pop("export_rec", None)
    _V["export_rec"] = {}
    if exp:
        rec["export"] = exp
    # Adam moments legitimately change by O(1) per step: drift is judged on weights only
    rec["worst"] = max([mvb] + [v for k, v in rec.get("drift", {}).items() if k in ("w", "router", "router_main", "m:param")])
    _log(rec)
    _V["prev"] = s
    return s


def after_move(mgr, step, pre):
    if not enabled() or pre is None:
        return
    s = snapshot(mgr)
    flat, wl = _summ(_cmp(s, pre))
    mvb, _ = _main_vs_bf16(s)
    rec = {"step": step, "phase": "post-move", "move": flat, "move_worst_layer": wl, "main_vs_bf16": round(mvb, 6)}
    rec["worst"] = max([mvb] + list(flat.values()))
    _log(rec)
    _V["prev"] = s


def on_export(mgr, name, tensor, lid, kind):
    """rank 0, after the logical remap: drift of exported rows vs the previous sync; router rows vs trainer."""
    if not enabled() or tensor is None:
        return
    fp = _proj(tensor).cpu()
    prev = _V["export_prev"].get(name)
    rec = _V.setdefault("export_rec", {})
    if prev is not None and prev.shape == fp.shape:
        r = _rel(fp, prev)
        k = "drift:" + kind
        if r > rec.get(k, (0, -1.0))[1]:
            rec[k] = (lid, round(r, 6))
    _V["export_prev"][name] = fp
    if kind in ("experts.gate_up_proj", "experts.down_proj"):
        _export_vs_local(mgr, tensor, lid, "fc1" if "gate_up" in kind else "fc2", rec)
    if kind == "gate":
        for l2, layer in mgr.layers:
            if l2 == lid:
                rw = layer.router.weight.detach()
                if rw.shape == tensor.shape:
                    tr = _proj(rw.index_select(0, mgr.slot_of[lid].to(rw.device)).to(tensor.dtype)).cpu()
                    r = _rel(fp, tr)
                    if r > rec.get("router_vs_trainer", (0, -1.0))[1]:
                        rec["router_vs_trainer"] = (lid, round(r, 6))


def _logprob_tensors(out, prefix=""):
    """{path: flat float tensor} for every tensor under a key containing 'log_prob' (nested tensors flattened)."""
    import torch
    res = {}
    if isinstance(out, dict) or hasattr(out, "items"):
        try:
            items = list(out.items())
        except Exception:
            items = []
        for k, v in items:
            p = f"{prefix}/{k}"
            if isinstance(v, torch.Tensor) and "log_prob" in str(k):
                t = v.values() if getattr(v, "is_nested", False) else v
                res[p] = t.detach().float().reshape(-1)
            elif not isinstance(v, torch.Tensor):
                res.update(_logprob_tensors(v, p))
    elif isinstance(out, (list, tuple)):
        for i, v in enumerate(out):
            res.update(_logprob_tensors(v, f"{prefix}[{i}]"))
    return res


def forward_ab(engine, mgr, orig_infer, data, loss_function):
    """ASRT_PLACE_VERIFY_FWD=N: for the first N forward-only batches run while experts are re-placed, run the same
    batch a second time with the placement undone (restore_identity) and compare the log-probs: a correct
    placement (weights + R3 id mapping) gives the same numbers up to the order of bf16 expert reductions.
    Collective (all actor ranks call infer_batch together). Returns the placed run's output."""
    import torch
    import copy
    n = int(os.environ.get("ASRT_PLACE_VERIFY_FWD", "0") or 0)
    if n <= _V.get("fwd_done", 0) or mgr.is_identity():
        return orig_infer(engine, data, loss_function=loss_function)
    _V["fwd_done"] = _V.get("fwd_done", 0) + 1
    try:
        data2 = data.clone()
    except Exception:
        data2 = copy.deepcopy(data)
    out1 = orig_infer(engine, data, loss_function=loss_function)
    saved = mgr.slot_of.clone()
    mgr.restore_identity()
    out3 = None
    try:
        data3 = None
        if os.environ.get("ASRT_PLACE_VERIFY_FWD_AA") == "1":  # control: identity twice (run-to-run noise)
            try:
                data3 = data2.clone()
            except Exception:
                data3 = copy.deepcopy(data2)
        out2 = orig_infer(engine, data2, loss_function=loss_function)
        if data3 is not None:
            out3 = orig_infer(engine, data3, loss_function=loss_function)
    finally:
        ar = torch.arange(mgr.E, device=saved.device)
        mgr.apply({lid: saved[lid].tolist() for lid in range(mgr.L) if not torch.equal(saved[lid], ar)})
    a, b = _logprob_tensors(out1), _logprob_tensors(out2)
    rec = {"phase": "fwd-ab", "n": _V["fwd_done"], "moved_layers": int(sum(not x for x in mgr._ident))}
    for k in sorted(set(a) & set(b)):
        if a[k].shape == b[k].shape and a[k].numel():
            d = (a[k] - b[k]).abs()
            rec[k] = {"mean_abs": round(float(d.mean()), 6), "max_abs": round(float(d.max()), 4),
                      "n": int(d.numel())}
    if out3 is not None:
        c = _logprob_tensors(out3)
        for k in sorted(set(b) & set(c)):
            if b[k].shape == c[k].shape and b[k].numel():
                d = (b[k] - c[k]).abs()
                rec["AA" + k] = {"mean_abs": round(float(d.mean()), 6), "max_abs": round(float(d.max()), 4)}
    rec["worst"] = max([v["mean_abs"] for v in rec.values() if isinstance(v, dict)] or [0.0])
    _log(rec)
    return out1


def _export_vs_local(mgr, tensor, lid, which, rec):
    """Exported stacked expert rows (logical order) vs this rank's own expert weights, layout-invariant (sum of
    squares per expert): row e must hold the expert living in slot slot_of[lid][e]. Checks every logical expert whose
    slot is local to this rank (rank 0 = EP rank 0); a mis-ordered export shows up as a relative error >> 1e-4."""
    import torch
    from .megatron_placement import _expert_params
    import torch.distributed as dist
    layer = dict(mgr.layers).get(lid)
    if layer is None:
        return
    me = dist.get_rank(mgr.ep_group) if dist.is_initialized() else 0
    loc = None
    for name, (params, grouped) in _expert_params(layer.experts).items():
        if which not in name or not name.endswith("weight"):
            continue
        if grouped:
            v = params[0].detach().float().reshape(mgr.n_local, -1).pow(2).sum(1).double()
        else:
            v = torch.stack([p.detach().float().pow(2).sum() for p in params]).double()
        loc = v if loc is None else loc + v
    if loc is None:
        return
    exp = tensor.detach().float().reshape(tensor.shape[0], -1).pow(2).sum(1).double()
    slots = mgr.slot_of[lid].tolist()
    worst, bad = 0.0, 0
    for e, sl in enumerate(slots):
        j = sl - me * mgr.n_local
        if 0 <= j < mgr.n_local:
            if exp[e].numel() == 0:
                continue
            r = float((exp[e].cpu() - loc[j].cpu()).abs() / (loc[j].abs().cpu() + 1e-30))
            worst = max(worst, r)
            bad += r > 1e-3
    k = "export_vs_trainer:" + which
    prev = rec.get(k, (0, -1.0, 0))
    if worst > prev[1]:
        rec[k] = (lid, round(worst, 6), int(bad))
