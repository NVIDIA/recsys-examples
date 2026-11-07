"""verl v1 replay buffer: how many groups/trajectories are submitted, trained and dropped. 

Measurement only (no behaviour change).  verl's v1 ReplayBuffer evicts finished prompt groups whose model-version span
exceeds ``max_off_policy_threshold`` (strategy 'drop'), plus DAPO-filtered and failed groups, and reports the stale
count only when it is non-zero (``training/off_policy/evicted_samples`` counts GROUPS).  This hook wraps
``_evict_terminal_groups`` (every ReplayBuffer class that defines it) and ``sample`` and logs per training step:
    submitted   prompt groups seen in TransferQueue for the first time since the previous step (and in total)
    trained     groups / trajectories in the sampled batch
    dropped     stale groups / trajectories evicted by max_off_policy_threshold (+ dapo / failed groups)
to ``[asrt] verl offpol {...}`` on stdout (driver.log) and ``$ASRT_LOG_DIR/verl-offpol-<pid>.jsonl``.
Enable: ASRT_VERL=...,offpol (the asrt verl plugin; also fine alone on the plain verl baseline).
"""
from __future__ import annotations

import json
import os
import time

from ._hooks import set_kept

_ST = {"seen": set(), "seen_prev": 0, "drop_g": 0, "drop_t": 0, "dapo_g": 0, "fail_g": 0, "spans": [],
       "tot_drop_g": 0, "tot_drop_t": 0, "tot_train_g": 0, "tot_train_t": 0}


def _log(rec):
    print(f"[asrt] verl offpol {json.dumps(rec)}", flush=True)
    d = os.environ.get("ASRT_LOG_DIR")
    if d:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"verl-offpol-{os.getpid()}.jsonl"), "a") as f:
                f.write(json.dumps({"t": time.time(), **rec}) + "\n")
        except OSError:
            pass


def _patch_class(C):
    # own __dict__ only: ReplayBufferAsync (the class fully-async verl uses) inherits the flag from ReplayBuffer, and
    # getattr() made it look patched already - its own sample() was never wrapped (seen: no offpol lines)
    if C.__dict__.get("_asrt_offpol", False):
        return False
    ev = C.__dict__.get("_evict_terminal_groups")
    smp = C.__dict__.get("sample")
    if ev is None and smp is None:
        return False

    if ev is not None:
        def _evict_terminal_groups(self, global_steps, partition_id, eviction_reasons, _o=ev):
            try:
                if partition_id == "train":
                    stale, dapo, failed = eviction_reasons[0], eviction_reasons[1], eviction_reasons[2]
                    part = self.partitions.get(partition_id, {}) if hasattr(self, "partitions") else {}
                    _ST["drop_g"] += len(stale)
                    _ST["drop_t"] += sum(1 for k in part if k.split("_")[0] in stale)
                    _ST["dapo_g"] += len(dapo)
                    _ST["fail_g"] += len(failed)
                    _ST["seen"].update(stale, dapo, failed)  # cleared from prompt_global_steps by the eviction
                    pgs = getattr(self, "prompt_global_steps", {}).get(partition_id, {})
                    _ST["spans"] += [global_steps - pgs.get(u, global_steps) + 1 for u in stale]
            except Exception:
                pass
            return _o(self, global_steps, partition_id, eviction_reasons)
        set_kept(C, "_evict_terminal_groups", _evict_terminal_groups)

    if smp is not None:
        def sample(self, global_steps, partition_id, batch_size, _o=smp):
            out = _o(self, global_steps, partition_id, batch_size)
            if partition_id != "train":
                return out
            try:
                for u in getattr(self, "prompt_global_steps", {}).get(partition_id, {}):
                    _ST["seen"].add(u)
                batch = out[0] if isinstance(out, tuple) else out
                keys = list(getattr(batch, "keys", None) or [])
                groups = {str(k).split("_")[0] for k in keys}
                _ST["seen"].update(groups)
                n_t = len(keys)
                sub = len(_ST["seen"])
                rec = {"step": int(global_steps), "submitted_groups_new": sub - _ST["seen_prev"], "submitted_groups_total": sub,
                       "trained_groups": len(groups), "trained_traj": n_t,
                       "dropped_stale_groups": _ST["drop_g"], "dropped_stale_traj": _ST["drop_t"],
                       "dropped_dapo_groups": _ST["dapo_g"], "dropped_failed_groups": _ST["fail_g"],
                       "dropped_span_max": max(_ST["spans"]) if _ST["spans"] else None,
                       "threshold": getattr(self, "max_off_policy_threshold", None),
                       "strategy": getattr(self, "max_off_policy_strategy", None)}
                _ST["tot_drop_g"] += _ST["drop_g"]; _ST["tot_drop_t"] += _ST["drop_t"]
                _ST["tot_train_g"] += len(groups); _ST["tot_train_t"] += n_t
                rec.update(total_trained_traj=_ST["tot_train_t"], total_dropped_traj=_ST["tot_drop_t"],
                           drop_share=round(_ST["tot_drop_t"] / max(1, _ST["tot_drop_t"] + _ST["tot_train_t"]), 4))
                _log(rec)
                _ST["seen_prev"] = sub
                _ST.update(drop_g=0, drop_t=0, dapo_g=0, fail_g=0, spans=[])
            except Exception as e:  # never break training for a counter
                print(f"[asrt] verl offpol log failed: {e!r}", flush=True)
            return out
        set_kept(C, "sample", sample)
    C._asrt_offpol = True
    return True


def install():
    from ._hooks import when_imported

    def on_mod(mod):
        n = 0
        for v in list(vars(mod).values()):
            if isinstance(v, type) and v.__module__ == mod.__name__:
                n += bool(_patch_class(v))
        print(f"[asrt] verl offpol: patched {n} replay-buffer classes (pid {os.getpid()})", flush=True)

    when_imported("verl.trainer.ppo.v1.replay_buffer", on_mod)
    return True
