"""Hard staleness ceiling for verl v1 async trainers (separate_async / colocate_async).

Unit = verl's own ``training/off_policy/trajectory_staleness_worst`` = (global_steps - 1) - min_global_steps of every
trained trajectory (trainer_base._compute_metrics).  verl itself only evicts finished groups whose ENQUEUE span exceeds
``max_off_policy_threshold`` (threshold 4 -> worst-case staleness up to 3 versions, observed in practice; larger
thresholds go higher).  This caps it at S
(ASRT_MAX_STALENESS, default 2 when the feature is on), monkeypatch only, verl's own drop rule stays underneath.

Version math (staleness_core): in verl step g the engines hold version g-1 (update_weights(g) runs in on_step_end),
so a prompt submitted during step g starts at version >= g-1 (warm-up: global_steps is already 1, version 0).  A group
with earliest version v may be trained up to step deadline(v) = v + S + 1.  Running groups use the lower bound
v >= tag-1 (prompt tag global_steps); finished groups the exact min over their trajectories' min_global_steps.

Three parts, all in the driver process:
  ticket admission  PPOTrainer._add_batch_to_generate / _add_prompts_to_generate (prepare_step, warm-up, refills):
                    a new group (deadline g-1+S+1) is submitted only while live groups (submitted, not trained, not
                    cut) <= (deadline - next_step + 1) x batch, i.e. <= (S+1) batches when submitted right after a
                    sync.  The rest is DEFERRED (owed) and released as soon as there is room (next prepare_step or
                    while sample() polls) - with the version of that moment.  Never more prompts than verl asks for.
  EDF + wait        ReplayBufferAsync._select_prompt_uids picks finished groups by (deadline, enqueue step) instead of
                    enqueue step only; _has_enough_samples additionally waits while the batch cannot stay feasible:
                    for every d >= s, finished groups with deadline <= d must cover n_d - (d - s) x batch (n_d = live
                    groups with deadline <= d) - i.e. a straggler that must be trained now (or that would push too many
                    groups past their deadline later) is waited for, like slime's ticket mode at the ceiling.
  cut               a straggler blocking the batch is cut after ASRT_TICKET_DROP_S (auto as slime) while the whole-run
                    budget lasts (ASRT_TICKET_DROP_BUDGET, else max(ASRT_TICKET_DROP_MIN 3, 1% x batch x steps));
                    a finished group past its deadline is always cut (mandatory, cannot be trained under S).  Cut
                    groups are evicted through verl's own stale path (refilled by verl, counted in its
                    evicted_samples) and logged with uid / age.
Logs: "[asrt] staleness ceiling ACTIVE: S=.. reader verl(ticket) .." once, "[asrt] staleness rollout <step> .." per
step (staleness-audit format; worst of the sampled batch computed from the same tags verl's metric uses), and
$ASRT_LOG_DIR/admit.jsonl: ceiling_active, verl_step (admitted / deferred / refused / dropped / wait_s / worst), verl_cut.
Enable: ASRT_VERL=...,ceiling.
"""
from __future__ import annotations

import collections
import os
import sys
import time

from ._hooks import set_kept

from . import staleness_core as SC

_FLAG = "_asrt_ceiling"


def _S():
    v = os.environ.get("ASRT_MAX_STALENESS", "").strip().lower()
    if v == "":
        return 2
    return SC.max_staleness()


class Ceiling:
    def __init__(self, S):
        self.S = S
        self.active = S is not None
        self.B = None          # groups per global step (train_batch_size)
        self.Bs = None         # groups per sample() call (train_batch_size // parameter_sync_step)
        self.total_steps = None
        self.owed = 0          # prompts verl asked for that are deferred
        self.cut = {}          # uid -> info (running cut: evicted by verl's stale path when it finishes)
        self.cut_logged = set()
        self.n_cut = self.n_cut_time = self.n_cut_mandatory = self.n_cut_over_budget = 0
        self.submit_t = {}     # uid -> (time, global_steps at submission)
        self.step_s = collections.deque(maxlen=32)
        self._t_last = None
        self.verified = False
        self.cur = self._fresh()
        self.tot = {"admitted": 0, "deferred": 0, "refused": 0, "wait_s": 0.0}
        self.wait_since = None
        self.worst_hist = []

    @staticmethod
    def _fresh():
        return {"admitted": 0, "deferred_new": 0, "refused_calls": 0, "released": 0, "dropped": 0, "wait_s": 0.0,
                "wait_polls": 0}

    # ------------------------------------------------------------------ version math
    def lb(self, tag_steps):
        return max(int(tag_steps) - 1, 0)

    def deadlines(self, prompt_steps, finished, traj_min):
        """uid -> deadline step for live groups (cut ones excluded)."""
        out = {}
        for uid, g in prompt_steps.items():
            if uid in self.cut:
                continue
            v = self.lb(g)
            if uid in finished and uid in traj_min:
                v = traj_min[uid]
            elif uid in traj_min:
                v = min(v, traj_min[uid])
            out[uid] = SC.deadline(v, self.S)
        return out

    @staticmethod
    def traj_min(partition):
        tm = {}
        for k, tag in partition.items():
            m = tag.get("min_global_steps") if isinstance(tag, dict) else None
            if m is None:
                continue
            u = k.split("_")[0]
            tm[u] = min(int(m), tm.get(u, 1 << 60))
        return tm

    def room(self, live_n, global_steps):
        """Prompts that may be submitted now (deadline (g-1)+S+1, next sampled step max(g, 1))."""
        g = int(global_steps)
        d_new = SC.deadline(self.lb(g), self.S)
        return max(0, (d_new - max(g, 1) + 1) * self.B - live_n)

    # ------------------------------------------------------------------ admission (trainer side)
    def _tq_live(self, tr):
        tq = _tq_module()
        data = tq.kv_list() if tq is not None else None
        items = (data or {}).get("train", {}) or {}
        prompts, finished, part = {}, set(), {}
        for k, tag in items.items():
            if tag.get("is_prompt", False):
                prompts[k] = tag.get("global_steps", tr.global_steps)
                if tag.get("status") in ("finished", "failure"):
                    finished.add(k)
            else:
                part[k] = tag
        return self.deadlines(prompts, finished, self.traj_min(part))

    def _verify(self, tr):
        if self.verified:
            return
        self.verified = True
        cfg = tr.config
        self.B = int(cfg.data.train_batch_size)
        pss = int(getattr(tr, "parameter_sync_step", 1) or 1)
        self.Bs = max(1, self.B // pss)
        self.total_steps = getattr(tr, "total_training_steps", None)
        rb = getattr(tr, "replay_buffer", None)
        if getattr(tr, "trainer_mode", "") == "sync":
            self.active = False
            print("[asrt] verl ceiling: trainer mode sync (on-policy, staleness 0) -> nothing to cap", flush=True)
            return
        if rb is None or not any(C.__dict__.get(_FLAG, False) for C in type(rb).__mro__):
            raise RuntimeError(f"[asrt] staleness ceiling S={self.S} configured but replay buffer "
                               f"{type(rb).__name__} is not patched (custom sampler?) - refusing to run uncapped")
        try:
            skip = bool(cfg.skip.rollout_tq.enable)
        except Exception:
            skip = False
        if skip:
            self.active = False
            print("[asrt] verl ceiling: rollout_tq skip enabled -> ceiling off", flush=True)
            return
        budget = self.budget()
        dl = os.environ.get("ASRT_TICKET_DROP_S", "auto")
        print(SC.active_line(self.S, f"reader verl(ticket): {type(rb).__name__}, batch {self.B} groups (sample "
                             f"{self.Bs}), admission <= (S+1) x batch = {(self.S + 1) * self.B} live groups after a sync, "
                             f"deadline step = start version + {self.S + 1}, EDF sampling + wait at the deadline, cut "
                             f"ASRT_TICKET_DROP_S={dl} budget {budget} groups; verl threshold "
                             f"{getattr(rb, 'max_off_policy_threshold', None)} ({getattr(rb, 'max_off_policy_strategy', None)}) "
                             f"kept underneath"), flush=True)
        SC.log_admit({"event": "ceiling_active", "t": time.time(), "S": self.S, "readers": {"verl(ticket)": (self.S + 1) * self.B},
                      "batch": self.B, "budget": budget, "drop_s": dl, "framework": "verl"})

    def budget(self):
        return SC.cut_budget((self.total_steps or 0) * (self.B or 0))

    def submit(self, tr, k, orig, live=None):
        """Submit up to owed + k prompts now (orig = verl's _add_prompts_to_generate)."""
        self._verify(tr)
        if not self.active:
            return orig(tr, k) if k > 0 else 0
        want = self.owed + int(k)
        if live is None:
            live = len(self._tq_live(tr))
        n = min(want, self.room(live, tr.global_steps))
        if n > 0:
            orig(tr, n)
        if k > 0 and n < want:
            self.cur["refused_calls"] += 1
        new_def = max(0, (want - n) - self.owed)
        self.cur["deferred_new"] += new_def
        self.tot["deferred"] += new_def
        if k == 0:
            self.cur["released"] += n
        self.owed = want - n
        self.cur["admitted"] += n
        self.tot["admitted"] += n
        return n

    # ------------------------------------------------------------------ sampling side
    def _cut(self, uid, reason, s, info):
        if uid in self.cut_logged:
            return
        self.cut_logged.add(uid)
        self.cut[uid] = info
        self.n_cut += 1
        self.cur["dropped"] += 1
        over = False
        if reason == "deadline_s":
            self.n_cut_time += 1
        else:
            self.n_cut_mandatory += 1
            over = self.n_cut > self.budget()
            self.n_cut_over_budget += over
        rec = {"event": "verl_cut", "t": time.time(), "step": int(s), "uid": uid, "reason": reason, "over_budget": over,
               "cut_total": self.n_cut, **info}
        print(f"[asrt] ticket cut: step {s} group {uid} ({reason}, age {info.get('age_steps')} steps / "
              f"{info.get('age_s')} s, start version {info.get('start_version')}, deadline step {info.get('deadline')}); "
              f"cut {self.n_cut} so far, budget {self.budget()} (compare: verl untrained in the same window)", flush=True)
        SC.log_admit(rec)

    def _age(self, uid, tag_steps, s):
        t = self.submit_t.get(uid, (None, None))[0]
        return {"age_steps": int(s) - int(tag_steps), "age_s": round(time.time() - t, 1) if t is not None else None,
                "enqueue_step": int(tag_steps)}

    def overdue_finished(self, rb, s, pid):
        """Finished groups that cannot be trained at step s or later (+ finished groups cut while running)."""
        pgs = rb.prompt_global_steps.get(pid, {})
        fin = set(rb.finished_keys.get(pid, set())) | set(rb.failure_keys.get(pid, set()))
        tm = self.traj_min(rb.partitions.get(pid, {}))
        out = set()
        for u in fin:
            if u in self.cut:
                out.add(u)
                continue
            if u in tm:
                v = tm[u]
                d = SC.deadline(v, self.S)
                if d < int(s):
                    self._cut(u, "overdue_finished", s, {"start_version": v, "deadline": d,
                                                          **self._age(u, pgs.get(u, s), s)})
                    out.add(u)
        return out

    def blocking(self, rb, s, pid):
        """Running groups the batch of step s has to wait for (EDF feasibility), [] if none."""
        pgs = rb.prompt_global_steps.get(pid, {})
        fin = set(rb.finished_keys.get(pid, set()))
        run = set(rb.pending_keys.get(pid, set())) | set(rb.running_keys.get(pid, set()))
        tm = self.traj_min(rb.partitions.get(pid, {}))
        dl = self.deadlines({u: pgs[u] for u in (fin | run) if u in pgs}, fin, tm)
        ds = sorted({d for d in dl.values() if d >= int(s)})
        n = f = 0
        block_d = None
        by_d = collections.Counter(dl.values())
        fin_d = collections.Counter(d for u, d in dl.items() if u in fin)
        # deadlines below s: finished ones are evicted; running ones can no longer be trained (their exact version
        # may still turn out newer) - not waited for
        for d in ds:
            n += by_d[d]
            f += fin_d[d]
            # this sample takes up to Bs finished groups in deadline order, each later step B: groups with deadline
            # <= d that this sample must take = n_d - (d - s) x B (more than Bs cannot be helped: those get cut)
            need = n - (d - int(s)) * self.B
            if f < min(need, self.Bs):
                block_d = d
                break
        if block_d is None:
            return [], dl
        return [u for u in run if u in dl and int(s) <= dl[u] <= block_d], dl

    def step_done(self, s, batch, metrics):
        now = time.time()
        if self._t_last is not None:
            self.step_s.append(now - self._t_last)
        self._t_last = now
        mins = []
        for tag in (getattr(batch, "tags", None) or []):
            if isinstance(tag, dict) and not tag.get("is_padding", False) and tag.get("min_global_steps") is not None:
                mins.append(int(tag["min_global_steps"]))
        ws = [SC.worst(s, m) for m in mins]
        mx = max(ws) if ws else None
        mean = sum(ws) / len(ws) if ws else 0.0
        over = sum(1 for w in ws if w > self.S)
        if self.wait_since is not None:
            self.cur["wait_s"] += now - self.wait_since
            self.wait_since = None
        self.tot["wait_s"] += self.cur["wait_s"]
        self.tot["refused"] += self.cur["refused_calls"]
        rec = {"event": "verl_step", "t": now, "step": int(s), "admitted": self.cur["admitted"],
               "released": self.cur["released"], "deferred": self.owed, "deferred_new": self.cur["deferred_new"],
               "refused": self.cur["refused_calls"], "dropped": self.cur["dropped"], "dropped_total": self.n_cut,
               "dropped_time": self.n_cut_time, "dropped_mandatory": self.n_cut_mandatory,
               "over_budget": self.n_cut_over_budget, "budget": self.budget(), "wait_s": round(self.cur["wait_s"], 1),
               "n_traj": len(ws), "worst_max": mx, "worst_mean": round(mean, 4), "over_cap": over, "S": self.S,
               "admitted_total": self.tot["admitted"], "deferred_total": self.tot["deferred"]}
        SC.log_admit(rec)
        print(SC.staleness_line(int(s), mean, mx if mx is not None else -1, self.S, over / max(1, len(ws)),
                                f" (n {len(ws)}, verl step {s}; admitted {rec['admitted']} deferred {self.owed} "
                                f"dropped {self.cur['dropped']} (total {self.n_cut}) wait {rec['wait_s']} s)"), flush=True)
        if over:
            print(f"[asrt] verl ceiling BREACH: step {s} {over} trajectories above S={self.S}", flush=True)
        if isinstance(metrics, dict):
            metrics.update({"asrt_ceiling/worst_max": float(mx if mx is not None else -1), "asrt_ceiling/over_cap": float(over),
                            "asrt_ceiling/dropped_total": float(self.n_cut), "asrt_ceiling/deferred": float(self.owed),
                            "asrt_ceiling/wait_s": float(self.cur["wait_s"])})
        self.cur = self._fresh()


_C = [None]


def _tq_module():
    m = sys.modules.get("verl.trainer.ppo.v1.replay_buffer")
    return getattr(m, "tq", None) if m is not None else None


def state() -> Ceiling | None:
    return _C[0]


# ---------------------------------------------------------------------- patches
def patch_replay_buffer(C) -> bool:
    """Patch an async ReplayBuffer class (own __dict__ flag: subclasses inherit attributes, see verl_offpol)."""
    if C.__dict__.get(_FLAG, False):
        return False
    if "_has_enough_samples" not in C.__dict__ and "wait_for_sampleable" not in C.__dict__:
        return False  # the sync ReplayBuffer: on-policy, nothing to cap
    o_reasons = C._terminal_eviction_reasons
    o_enough = C._has_enough_samples
    o_select = C._select_prompt_uids
    o_sample = C.sample

    def _terminal_eviction_reasons(self, global_steps, partition_id, _o=o_reasons):
        r = _o(self, global_steps, partition_id)
        c = _C[0]
        if c is None or not c.active or partition_id != "train" or c.B is None:
            return r
        extra = c.overdue_finished(self, global_steps, partition_id)
        if not extra:
            return r
        return (set(r[0]) | extra,) + tuple(r[1:])

    def _has_enough_samples(self, global_steps, partition_id, batch_size, sampleable_keys, _o=o_enough):
        ok = _o(self, global_steps, partition_id, batch_size, sampleable_keys)
        c = _C[0]
        if c is None or not c.active or partition_id != "train" or c.B is None:
            return ok
        tr = getattr(getattr(self, "refill_fn", None), "__self__", None)
        if c.owed > 0 and tr is not None:  # deferred prompts: release while there is room (cuts free slots)
            live = sum(1 for u in self.prompt_global_steps.get(partition_id, {}) if u not in c.cut)
            c.submit(tr, 0, _ORIG["add_prompts"], live=live)
        blk, dl = c.blocking(self, global_steps, partition_id)
        if blk:
            dsec = SC.drop_deadline(c.step_s, "auto")
            budget = c.budget()
            pgs = self.prompt_global_steps.get(partition_id, {})
            if dsec is not None:
                for u in sorted(blk, key=lambda u: c.submit_t.get(u, (time.time(),))[0]):
                    t0 = c.submit_t.get(u, (None,))[0]
                    if t0 is None or time.time() - t0 <= dsec or c.n_cut_time >= budget:
                        continue
                    c._cut(u, "deadline_s", global_steps, {"start_version": c.lb(pgs.get(u, global_steps)),
                                                           "deadline": dl.get(u), "drop_s": round(dsec, 1),
                                                           **c._age(u, pgs.get(u, global_steps), global_steps)})
                blk, _ = c.blocking(self, global_steps, partition_id)
        if not blk and c.wait_since is not None:
            c.cur["wait_s"] += time.time() - c.wait_since
            c.wait_since = None
        if blk:
            if c.wait_since is None:
                c.wait_since = time.time()
            c.cur["wait_polls"] += 1
            if c.cur["wait_polls"] in (1, 30) or c.cur["wait_polls"] % 150 == 0:
                print(f"[asrt] verl ceiling: step {global_steps} waits: needs more finished groups among {len(blk)} "
                      f"running ones due by step {max(dl.get(u, 0) for u in blk)} (e.g. {sorted(blk)[:2]}), finished "
                      f"sampleable {len(sampleable_keys)}", flush=True)
            return False
        return ok

    def _select_prompt_uids(self, partition_id, sampleable_keys, batch_size, _o=o_select):
        c = _C[0]
        if c is None or not c.active or partition_id != "train" or c.B is None:
            return _o(self, partition_id, sampleable_keys, batch_size)
        pgs = dict(self.prompt_global_steps[partition_id])
        part = dict(self.partitions[partition_id])
        tm = c.traj_min(part)

        def key(u):
            v = tm.get(u, c.lb(pgs.get(u, 0)))
            return (SC.deadline(v, c.S), pgs.get(u, 0))
        ordered = sorted(sampleable_keys, key=key)
        return ordered[:batch_size], part, pgs

    def sample(self, global_steps, partition_id, batch_size, _o=o_sample):
        out = _o(self, global_steps, partition_id, batch_size)
        c = _C[0]
        if c is None or not c.active or partition_id != "train" or c.B is None:
            return out
        try:
            batch, metrics = (out if isinstance(out, tuple) else (out, None))
            c.step_done(global_steps, batch, metrics)
        except Exception as e:  # never break training for a log line
            print(f"[asrt] verl ceiling step log failed: {e!r}", flush=True)
        return out

    set_kept(C, "_terminal_eviction_reasons", _terminal_eviction_reasons)
    set_kept(C, "_has_enough_samples", _has_enough_samples)
    set_kept(C, "_select_prompt_uids", _select_prompt_uids)
    set_kept(C, "sample", sample)
    setattr(C, _FLAG, True)
    return True


_ORIG = {}


def patch_trainer(T) -> bool:
    if T.__dict__.get(_FLAG, False):
        return False
    o_add = T._add_prompts_to_generate
    o_batch = T._add_batch_to_generate
    o_submit = T._submit_batch_to_rollout
    _ORIG["add_prompts"] = o_add

    def _add_prompts_to_generate(self, num_prompts):
        c = _C[0]
        if c is None or not c.active:
            return o_add(self, num_prompts)
        return c.submit(self, num_prompts, o_add)

    def _add_batch_to_generate(self):
        c = _C[0]
        if c is None or not c.active:
            return o_batch(self)
        c._verify(self)
        if not c.active:
            return o_batch(self)
        return c.submit(self, int(self.config.data.train_batch_size), o_add)

    def _submit_batch_to_rollout(self, batch):
        c = _C[0]
        if c is not None:
            try:
                now = time.time()
                for u in list(batch["uid"]):
                    c.submit_t[str(u)] = (now, int(self.global_steps))
            except Exception:
                pass
        return o_submit(self, batch)

    set_kept(T, "_add_prompts_to_generate", _add_prompts_to_generate)
    set_kept(T, "_add_batch_to_generate", _add_batch_to_generate)
    set_kept(T, "_submit_batch_to_rollout", _submit_batch_to_rollout)
    setattr(T, _FLAG, True)
    return True


def install(S=None):
    from ._hooks import when_imported

    S = S if S is not None else _S()
    if S is None:
        print("[asrt] verl ceiling: ASRT_MAX_STALENESS=off -> not installed", flush=True)
        return False
    if _C[0] is None:
        _C[0] = Ceiling(S)

    def on_rb(mod):
        n = 0
        for v in list(vars(mod).values()):
            if isinstance(v, type) and v.__module__ == mod.__name__ and hasattr(v, "_terminal_eviction_reasons"):
                n += bool(patch_replay_buffer(v))
        print(f"[asrt] verl ceiling: patched {n} async replay-buffer classes, S={S} (pid {os.getpid()})", flush=True)

    def on_tb(mod):
        patch_trainer(mod.PPOTrainer)

    when_imported("verl.trainer.ppo.v1.replay_buffer", on_rb)
    when_imported("verl.trainer.ppo.v1.trainer_base", on_tb)
    return True
