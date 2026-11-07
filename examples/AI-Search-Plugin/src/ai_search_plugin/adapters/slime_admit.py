"""Batch-aware admission for slime's fully-async rollout.

What slime does (slime/rollout/fully_async_rollout.py, 2026-10):
  * the worker keeps up to C = sglang_server_concurrency x engines / n_samples groups in flight and only stops
    pulling prompts once the *completed* queue also holds C groups.  In-flight groups keep finishing after that,
    so the queue reaches ~2C.  Example (C = 128 groups, batch B = 68): 255 completed groups were waiting
    when training asked for batch 1 -> every batch is 3-4 weight versions old (slime rollout/staleness) and a
    4-step run ends with ~2-2.5 batches generated but never trained.
  * the trainer still waits ~80-120 s per step: the RolloutManager only starts building the batch
    (convert, R3 tensors, build_dp_schedule, ray.put per DP rank) when training asks, although the groups have
    been sitting complete in the queue for minutes (measured: gen_wait 0.0 s, convert 74-123 s).

What this does (both monkey patches, no slime source change, off by default):
  * ``cap``   outstanding groups (admitted - handed to training, i.e. in flight + completed-unused + prebuilt)
              <= (1 + ASRT_ADMIT_AHEAD) x rollout_batch_size (default AHEAD 1: the next batch + one more).  The
              AReaL-style staleness bound (admitted <= (consumed + 1 + eta) x B).  slime's own in-flight cap
              (engine concurrency) stays; when rollout is the bottleneck the completed queue is empty and the cap
              is not binding (C <= 2B), so throughput is unchanged; when training is the bottleneck it stops
              slime from running 3-4 batches ahead.  Groups are still admitted in prompt order (the batch the
              trainer needs next is always the oldest in flight).
  * ``eager`` right after handing batch i to training, a manager thread waits until B completed groups are
              queued and builds batch i+1 (slime's own RolloutManager._generate path, FIFO like slime) while the
              trainer trains batch i.  generate(i+1) then returns the prebuilt refs.  Batch content is identical
              (slime hands out the first B completed groups either way); only *when* the conversion runs changes.
              The staleness metric is computed against the version the trainer will hold (engine version at
              generate(i) + 1).  If generate(i+1) arrives before B groups are complete the prebuild is cancelled
              and slime's normal path runs.  Off automatically with straw transport, training recovery,
              dynamic/sample filters, eval, offload/colocate, and right before a checkpoint save.
  * ``staleness`` hard staleness ceiling (ASRT_MAX_STALENESS=S), in verl's units: per sample, the
              version of the weights that TRAIN on it minus the EARLIEST weight version it was generated with (verl
              training/off_policy/trajectory_staleness_worst; = slime's rollout/staleness when slime hands batches out
              after the weight sync).  Measured verl (1 version/step): mean ~1.0, max 2 (one-step off-policy plus
              stragglers).  The ceiling bounds the groups that are admitted but not yet handed to training: <= (S-1) x batch
              when the asrt overlap driver hands batch i+1 out during the weight sync after train(i) (ASRT_PREFETCH has
              ``overlap``; groups admitted then generate during train(i+1)), <= S x batch otherwise (slime's serial order).
              Rollout waits for training once the ceiling is reached (no new groups; in-flight groups finish); samples, TIS
              and the loss are untouched.  Every rollout logs mean / max / share over S in these units to the driver log
              ("[asrt] staleness ...") and admit.jsonl, next to slime's own metric (which, under the overlap driver, is read
              before the sync and so reads 1 low).  Setting ASRT_MAX_STALENESS turns ``cap`` on by itself; off/0/empty =
              no ceiling.
  * ``profile`` one line per rollout in $ASRT_LOG_DIR/admit.jsonl (queue/in-flight/outstanding, prebuilt hit,
              build seconds, versions).

Turn on (one line, chains with slime_prefetch / slime_r3pack / slime_toolcache):
    --rollout-function-path ai_search_plugin.adapters.slime_admit.generate_rollout_fully_async
    ASRT_ADMIT=cap,eager,profile      ASRT_ADMIT_AHEAD=1
    ASRT_ADMIT_INNER=<the rollout function you used before>   (default slime's fully_async one; e.g.
                     ai_search_plugin.adapters.slime_prefetch.generate_rollout_fully_async)
Requires slime's fully-async worker (non-distributed reader).  ASRT_ADMIT unset/empty = slime unchanged.
"""
from __future__ import annotations

import collections
import copy
import json
import os
import threading
import time

DEFAULT_INNER = "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
_tls = threading.local()


from . import staleness_core as SC  # version math / budget / logging shared with the verl ceiling
from .staleness_core import max_staleness  # noqa: F401  (ASRT_MAX_STALENESS=S, re-exported)


def _overlap_driver() -> bool:
    """slime_prefetch 'overlap': batch i+1 is handed out during the weight sync after train(i)."""
    return "overlap" in {x.strip() for x in os.environ.get("ASRT_PREFETCH", "").split(",")}


def opts() -> set:
    o = {s.strip() for s in os.environ.get("ASRT_ADMIT", "").split(",") if s.strip()}
    if max_staleness() is not None:
        o |= {"cap", "staleness"}
    return o


def _log(rec: dict):
    d = os.environ.get("ASRT_LOG_DIR")
    if not d or not ({"profile", "staleness"} & opts()):
        return
    SC.log_admit(rec)


# ----------------------------------------------------------------------------------------------- admission cap
class AdmitCounter:
    """Groups admitted / requeued (aborted) / handed to training.  outstanding = admitted - requeued - handed."""

    def __init__(self, batch: int, ahead: float):
        self.batch = int(batch)
        self.cap = max(1, int(round((1.0 + ahead) * batch)))
        self.lock = threading.Lock()
        self.admitted = self.requeued = self.handed = 0
        self.refused = 0

    def outstanding(self) -> int:
        with self.lock:
            return self.admitted - self.requeued - self.handed

    def may_admit(self) -> bool:
        ok = self.outstanding() < self.cap
        if not ok:
            self.refused += 1
        return ok

    def add(self, field: str, k: int):
        with self.lock:
            setattr(self, field, getattr(self, field) + k)


class GatedBuffer:
    """Stands in for the worker's ``data_buffer``: the worker's top-up loop calls ``get_samples(1)`` and stops
    when it returns []; aborted groups come back through ``add_samples``.  Everything else is delegated."""

    def __init__(self, inner, counter: AdmitCounter):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_ctr", counter)

    def get_samples(self, num_samples):
        if not self._ctr.may_admit():
            return []
        out = self._inner.get_samples(num_samples)
        self._ctr.add("admitted", len(out) if out else 0)
        return out

    def add_samples(self, samples):
        r = self._inner.add_samples(samples)
        self._ctr.add("requeued", len(samples))
        return r

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __setattr__(self, k, v):
        setattr(self._inner, k, v)


# ----------------------------------------------------------------------------------------------- ticket mode
def _ticket_mode() -> bool:
    # ticket: default (ceiling held in every tested workload, faster than count); count = old cap
    return max_staleness() is not None and SC.stale_mode() == "ticket"


def _gkey(group):
    try:
        x = group[0]
        x = x[0] if isinstance(x, list) else x
        i = getattr(x, "index", None)
        return ("i", i) if i is not None else ("o", id(x))
    except Exception:
        return ("o", id(group))


def _versions(group):
    vs = []
    for x in group:
        for y in (x if isinstance(x, list) else [x]):
            for v in (getattr(y, "weight_versions", None) or []):
                try:
                    vs.append(int(v))
                except (TypeError, ValueError):
                    pass
    return vs


class Tickets:
    """ version-ticketed admission (replaces the group-count cap under ASRT_MAX_STALENESS).

    Every admitted group gets the ticket of the batch it WILL be trained in (= rollout id t, trained at version
    v0 + t, one weight update per step) and is handed out exactly in that batch (oldest ticket first; a younger
    ticket only fills a batch whose own groups were requeued).  A group may start only while
        t - (v_now - v0) <= S,
    v_now = newest weight version seen on any finished sample (a lower bound of the engines' version, so the
    check is conservative).  Its earliest version is >= v_now, its train version is v0 + t, so its staleness in
    verl units is <= S by construction - no slip into a later batch, unlike the count cap.  While batch m is in
    its tail, tickets m+1..m+S may already generate (the count cap allowed only m+1): the engines stay busy and
    the next batches are mostly done when their turn comes.  Requeued groups lose their ticket and are re-admitted
    under the same rule against their own earliest retained version."""

    def __init__(self, batch: int, S: int):
        self.B, self.S = int(batch), int(S)
        self.lock = threading.Lock()
        self.count = {}        # ticket -> live admitted groups
        self.total = {}        # ticket -> admitted - requeued (incl. already handed)
        self.pending = {}      # ticket -> admitted, not yet finished
        self.ticket_of = {}    # group key -> ticket
        self.start_rel = {}    # group key -> weight updates since batch 0 when it was admitted (lower bound)
        self.pend_keys = {}    # ticket -> keys admitted, not yet finished
        self.deferred = 0
        self.n_rollout = None  # set by the installer: run length (rollouts), for the whole-run cut budget
        self.gidx = {}         # group key -> slime group_index (slow-tool lookup)
        self.slow_keys = set() # groups that ran >= ASRT_TICKET_SLOW_S (ratio budget)      # ASRT_TICKET_DEFER swaps (straggler -> later ticket within the cap)
        self.t_admit = {}      # group key -> admission time
        self.t_fin = {}        # group key -> real completion time (done callback; own run time, not queue time)
        self.fresh = {}        # group key -> pristine copy of its inputs (first admission), see TicketBuffer
        self.refreshed = 0     # requeued groups restarted fresh because their retained samples could not meet S
        self.lat = collections.deque(maxlen=256)  # finished group latencies (s)
        self.dropped = set()   # keys cut from their batch (output discarded when they finish)
        self.n_dropped = 0
        self.drop_in_batch = {}  # batch -> groups cut from it
        self.drop_guarded = 0
        self.step_s = collections.deque(maxlen=32)  # hand-out intervals
        self._t_last_batch = None
        self.buf = {}          # ticket -> [(gid, group)] finished, not handed
        self.m = 0             # batch being collected (= batches handed)
        self.v0 = self.vmax = None
        self.v_eng = None
        self.refused = self.admitted = self.stale_requeue = 0
        self.peak_live = 0
        self.handed = {}       # group key -> ticket it was handed out under (for refund on filter reject)
        self.filter_refunds = 0
        self.cut_requeue_dropped = 0   # cut groups aborted at a sync: discarded, not requeued
        self.cut_readmit_ignored = 0   # cut groups that still reached admit() (defensive path): no ticket taken
        self.ledger_bad = 0            # batch_done() ledger checks that failed (see ledger())
        # start version (rel units, lower bound) of a group's FIRST admission - the oldest version any
        # of its tokens can carry (tokens are generated after admission; versions only grow).  Kept through requeues
        # (an aborted group's retained samples keep their tokens) and reset only when the group restarts from its
        # pristine copy.  SGLang's weight_version labels are the version at call RETURN: a call that runs across a
        # sync (flush 0 / pause in place / kvepoch / kvfresh retract-resume) is labelled newer than its oldest tokens,
        # so labels may only ever LOWER this bound, never raise it.
        self.sr_first = {}     # group key -> start_rel at first admission
        self.sr_by_gi = {}     # group_index -> same (staleness log: train version - this = token-level upper bound)
        # checkpoint resume (see snapshot / restore / adopt_without_book and the worker state_dict patch)
        self.resume_floor = None       # engine version at the save: every group finishing after the resume is newer
        self.resume_version_bad = 0    # groups finishing after a resume with all versions below that (version reset)
        self.resume_m_bad = 0          # restored batch counter != the rollout slime asks for after the resume
        self.v0_off = None             # resume without a book: v0 = min(newest version of a finished group) - v0_off
        self.resumed = None            # {"mode": ..., ...} what the last resume restored (logged)

    def vstart_rel(self, k, vs):
        """ lock held.  Start version of group k in rel units for the ceiling: min(first admission bound,
        min weight_version label - v0, rel() now).  Labels alone are the version at call return (may be newer than the
        oldest token); the admission bound is a true lower bound of every token's version."""
        c = [self.rel()]
        if vs and self.v0 is not None:
            c.append(min(vs) - self.v0)
        if k in self.sr_first:
            c.append(self.sr_first[k])
        return min(c)

    def _note_start(self, k, g, sr):
        """ lock held: record the first-admission start version (kept until a fresh restart)."""
        self.sr_first.setdefault(k, sr)
        try:
            x = g[0][0] if isinstance(g[0], list) else g[0]
            gi = getattr(x, "group_index", None)
            if gi is not None:
                self.sr_by_gi[gi] = self.sr_first[k]
        except Exception:
            pass

    def rel(self):
        """Weight updates since batch 0, lower bound: newest version on a finished sample, the slowest engine's
        own weight version (polled, see start_engine_poll) and the hand-outs so far (batch m is handed after >= m-1
        syncs in slime's serial order, >= m-2 under the asrt overlap driver, which hands out during the sync)."""
        seen = 0 if self.v0 is None else self.vmax - self.v0
        eng = 0 if (self.v0 is None or self.v_eng is None) else self.v_eng - self.v0
        return max(seen, eng, self.m - (2 if _overlap_driver() else 1), 0)

    def start_engine_poll(self, args):
        """With only finished samples as evidence the version estimate froze whenever nothing was
        in flight (seen: whole minutes at 0 tok/s until hand-outs moved the fallback).  Poll every engine's
        /get_weight_version (what slime's SGLangEngine reads) once a second; keep the MINIMUM over engines, so a
        group dispatched anywhere starts at >= that version.  Off with ASRT_TICKET_POLL=0; any error -> no value."""
        if os.environ.get("ASRT_TICKET_POLL", "1") == "0" or getattr(args, "sglang_router_ip", None) is None:
            return

        def loop():
            import requests
            urls, last = [], 0.0
            while True:
                try:
                    if not urls or time.time() - last > 60:
                        r = requests.get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers", timeout=10)
                        urls = sorted({w["url"] for w in r.json().get("workers", []) if w.get("url")})
                        last = time.time()
                    vs = []
                    for u in urls:
                        v = requests.get(f"{u}/get_weight_version", timeout=5).json().get("weight_version")
                        vs.append(int(v))
                    if vs:
                        with self.lock:
                            self.v_eng = min(vs)
                except Exception:
                    pass
                time.sleep(float(os.environ.get("ASRT_TICKET_POLL_S", "1")))

        threading.Thread(target=loop, name="asrt-ticket-version", daemon=True).start()

    def next_ticket(self, minv=None, sr=None):
        """Lowest ticket >= m with room whose version budget admits a group starting now (or at minv, absolute, or
        at sr, rel units - works before v0 is known)."""
        t = self.m
        while self.total.get(t, 0) >= self.B:  # a ticket is a batch: B groups in total, handed or not
            t += 1
        start_rel = self.rel() if minv is None or self.v0 is None else min(self.rel(), minv - self.v0)
        if sr is not None:
            start_rel = min(start_rel, sr)
        # ASRT_TICKET_ADMIT_SLACK=k (opt-in, non-cutting option): admit only with k versions of slack
        # (t - start_rel <= S - k), so every group can still be deferred k batches by DEFER without passing the
        # ceiling S.  320's step-7 straggler was admitted at zero slack (t - start_rel = S) -> no deferral possible.
        # only look-ahead tickets (t > m): batch m itself must stay admissible or admission deadlocks (CPU sim hung)
        slack = int(os.environ.get("ASRT_TICKET_ADMIT_SLACK", "0")) if (minv is None and sr is None and t > self.m) else 0
        return t if SC.within(t, start_rel, self.S, slack) else None

    def _trainable(self, minv=None, sr=None) -> bool:
        """A requeued group with earliest retained version minv (absolute) / sr (rel units) can still be trained within
        S (lock held).  Ticket mode: a ticket with room within its version budget (a full batch m means m+1 = staleness
        S+1, so no)."""
        return self.next_ticket(minv, sr=sr) is not None

    def restart_if_stale(self, g):
        """ lock held.  A requeued group (slime flush-cache-interval 1 aborts every in-flight
        request at each weight sync; prepare_retry_group keeps the finished siblings and the flat partial samples WITH
        their old weight_versions) that no batch can still train within S restarts from the pristine copy of its
        first-admission inputs, in place, so every sample is generated again under current weights.  Decided under
        the same lock as the ticket it then gets (deciding it outside: batch_done() in between could still
        force it into batch m).  Returns the group's versions after the decision (empty after a restart)."""
        vs = _versions(g)
        k = _gkey(g)
        if not vs and k not in self.sr_first:
            return vs
        sr = self.vstart_rel(k, vs)   # not the labels alone: they can be newer than the oldest token
        if self._trainable(sr=sr):
            return vs
        src = self.fresh.get(k)
        if src is None or not isinstance(g, list):
            # no pristine copy (deepcopy failed at first admission): cannot regenerate, counted as stale_requeue
            _log({"event": "requeue_stale_kept", "t": time.time(), "m": self.m, "min_version": min(vs) if vs else None,
                  "start_rel": sr, "v0": self.v0,
                  "sample_index": k[1] if k[0] == "i" else None})
            return vs
        g[:] = copy.deepcopy(src)
        self.sr_first.pop(k, None)    # a fresh restart regenerates every token: its start is the new admission
        self.refreshed += 1
        _log({"event": "requeue_refresh", "t": time.time(), "m": self.m, "sample_index": k[1] if k[0] == "i" else None,
              "min_version": min(vs) if vs else None, "start_rel": sr, "v0": self.v0, "refreshed": self.refreshed})
        return _versions(g)

    def admit(self, groups):
        with self.lock:
            for g in groups:
                if _gkey(g) in self.dropped:
                    # a cut group must not take a ticket again: its result is discarded in take(),
                    # so that ticket's pending would never reach 0 (batch never fills).  requeued() already keeps cut
                    # groups out of slime's buffer; this only guards other paths.  No ticket, stays cut.
                    self.cut_readmit_ignored += 1
                    _log({"event": "cut_readmit_ignored", "t": time.time(), "m": self.m,
                          "sample_index": _gkey(g)[1] if _gkey(g)[0] == "i" else None})
                    continue
                vs = self.restart_if_stale(g)
                k = _gkey(g)
                sr = self.vstart_rel(k, vs)   # first-admission bound, labels can only lower it
                requeue = bool(vs) or k in self.sr_first
                t = self.next_ticket(sr=sr if requeue else None)
                if t is None:  # requeued group with old retained samples, no room and no pristine copy: flag it
                    t = self.m
                    self.stale_requeue += 1
                self._note_start(k, g, sr)
                self.ticket_of[k] = t
                self.start_rel[k] = sr
                self.t_admit[k] = time.time()
                self.t_fin.pop(k, None)
                try:
                    x = g[0][0] if isinstance(g[0], list) else g[0]
                    self.gidx[k] = getattr(x, "group_index", None)
                except Exception:
                    pass
                self.pend_keys.setdefault(t, set()).add(k)
                self.count[t] = self.count.get(t, 0) + 1
                self.total[t] = self.total.get(t, 0) + 1
                self.pending[t] = self.pending.get(t, 0) + 1
                self.admitted += 1
            live = sum(c for t, c in self.count.items() if t >= self.m)
            self.peak_live = max(self.peak_live, live)

    def _drop_cut_requeue(self, k):
        """ lock held.  A group cut from its batch (_drop_overdue: its ticket slot and pending were
        already released / given to its replacement) was aborted at a weight sync (slime flush 1) before it finished.
        It is NOT requeued: re-admitting it took a new ticket whose pending was never decremented (take() discards a
        cut group's result) and own_age() was 0 for it (t_admit gone), so the batch could never fill - a deadlock found
        in review.  The prompt is skipped, as the cut documents; its copy is freed."""
        self.dropped.discard(k)
        for d in (self.fresh, self.start_rel, self.t_admit, self.t_fin, self.gidx, self.sr_first):
            d.pop(k, None)
        self.cut_requeue_dropped += 1
        _log({"event": "cut_requeue_dropped", "t": time.time(), "m": self.m,
              "sample_index": k[1] if k[0] == "i" else None, "n": self.cut_requeue_dropped})

    def requeued(self, groups):
        """Aborted groups slime puts back (done callback -> data_buffer.add_samples).  Returns the groups that may
        go back to slime's buffer: cut groups are discarded here (see _drop_cut_requeue)."""
        keep = []
        with self.lock:
            for g in groups:
                k = _gkey(g)
                if k in self.dropped:
                    self._drop_cut_requeue(k)
                    continue
                keep.append(g)
                t = self.ticket_of.pop(k, None)
                self.start_rel.pop(k, None)
                self.t_fin.pop(k, None)   # an aborted group also passes the done callback
                if t is not None:
                    self.count[t] -= 1
                    self.total[t] -= 1
                    self.pending[t] -= 1
                    self.pend_keys.get(t, set()).discard(k)
        return keep

    def _tracked(self, k) -> bool:
        """k is admitted and not yet handed / requeued (lock held)."""
        return k in self.ticket_of or k in self.dropped

    def finished(self, group):
        """Done callback (worker loop, at the real completion).  Also records the completion time - a
        group's own run time is admission -> completion, NOT -> the take() that pulls it from slime's queue (otherwise
        ~10x too many groups count as slow: normal groups that finished during a long step sat queued)."""
        vs = _versions(group)
        with self.lock:
            k = _gkey(group)
            if self._tracked(k):  # slime's own done callback runs first: an aborted group is already requeued (or
                self.t_fin[k] = time.time()   # discarded, if cut) - no completion time for it (was a stale entry)
            if vs:
                if self.resume_floor is not None and max(vs) <= self.resume_floor and self._tracked(k):
                    # a group generated after the resume carries a version newer than the engines had at
                    # the save (the first sync after a resume publishes start + 1); an older one means the version
                    # counter restarted (slime_resume not active) - the ceiling's version reference is then wrong
                    self.resume_version_bad += 1
                    if self.resume_version_bad == 1:
                        print(f"[asrt] ticket resume: group finished with versions {sorted(set(vs))} <= the save-time "
                              f"engine version {self.resume_floor} - the weight version restarted after the resume "
                              f"(ai_search_plugin.adapters.slime_resume not installed?); ledger BAD", flush=True)
                        _log({"event": "resume_version_bad", "t": time.time(), "versions": sorted(set(vs)),
                              "floor": self.resume_floor})
                cand = min(vs) if self.v0_off is None else max(vs) - self.v0_off
                self.v0 = cand if self.v0 is None else min(self.v0, cand)
                self.vmax = max(vs) if self.vmax is None else max(self.vmax, max(vs))

    def own_age(self, k, now) -> float:
        """Own run time of group k: admission -> completion if it has finished, else -> now (lock held)."""
        return min(self.t_fin.get(k, now), now) - self.t_admit.get(k, now)

    def take(self, items, limit):
        """items: newly finished (gid, group) from slime's queue; returns what the collector may hand out now."""
        with self.lock:
            for gid, g in items:
                k = _gkey(g)
                ta = self.t_admit.pop(k, None)
                tf = min(self.t_fin.pop(k, time.time()), time.time())   # real completion, not this poll
                if ta is not None and tf - ta >= SC.slow_s():
                    self.slow_keys.add(k)
                if ta is not None:
                    self.lat.append(tf - ta)
                if k in self.dropped:  # cut from its batch earlier: its result would exceed the ceiling, discard
                    self.dropped.discard(k)
                    self.sr_first.pop(k, None)
                    self.fresh.pop(k, None)
                    self.start_rel.pop(k, None)
                    self.gidx.pop(k, None)
                    continue
                t = self.ticket_of.get(k, self.m)
                self.pending[t] = self.pending.get(t, 0) - 1
                self.pend_keys.get(t, set()).discard(k)
                self.buf.setdefault(t, []).append((gid, g))
            # cut first: an overdue straggler at the ceiling must get a replacement before
            self._drop_overdue()  # _defer hands the finished later groups to deferrable stragglers
            if _defer_on():
                self._defer()
            out = []
            want = limit if limit is not None else 1 << 30
            for t in sorted(self.buf):
                if len(out) >= want:
                    break
                m = self.m
                if t > m and (self.pending.get(m, 0) > 0 or self.total.get(m, 0) < self.B):
                    break  # batch m still has groups to admit or generating: they go first
                lst = self.buf[t]
                k = min(len(lst), want - len(out))
                out += lst[:k]
                del lst[:k]
                if not lst:
                    del self.buf[t]
            if out:  # which groups this hand-out carries (batch m) + their start bound: lets the token-level
                # staleness be rebuilt offline from trainer-side update times, independently of SGLang's labels
                _log({"event": "ticket_handout", "t": time.time(), "batch": self.m,
                      "groups": [[self.gidx.get(_gkey(g)), self.start_rel.get(_gkey(g)), self.ticket_of.get(_gkey(g))]
                                 for _, g in out]})
            for _, g in out:
                self.fresh.pop(_gkey(g), None)
                self.sr_first.pop(_gkey(g), None)
                t = self.ticket_of.pop(_gkey(g), None)
                if t is not None:
                    self.count[t] -= 1
                    self.handed[_gkey(g)] = t   # remembered for refund() if the dynamic filter rejects it
            return out

    def refund(self, group):
        """slime's dynamic filter (DAPO-style, e.g. check_reward_nonzero_std) rejected a group this
        ticket book handed out: give its slot back to its ticket so a replacement can be admitted.  Without it the
        rejected groups still count toward the ticket's B, tickets m..m+S fill up with groups that never train,
        admission stops and the collector waits forever (seen: a rollout stuck at 26/32 groups).
        The replacement is admitted by the usual rule (lowest ticket >= m with room, t - start_rel <= S), so the
        ceiling is unchanged."""
        with self.lock:
            self.fresh.pop(_gkey(group), None)
            t = self.handed.pop(_gkey(group), None)
            if t is not None:
                self.total[t] = self.total.get(t, 0) - 1
                self.filter_refunds += 1

    def drop_deadline(self):
        """ASRT_TICKET_DROP_S: seconds, or 'auto' = max(60, ASRT_TICKET_DROP_STEPS (4, verl's version window) x the
        lower-quartile step time) after 3 steps; unset/0/off = never drop (default).  Under DEFER (default) the cut is
        on, budgeted to the baseline's untrained share (see _drop_overdue): stragglers at the ceiling are cut only while
        ours <= that count.  The rule itself lives in staleness_core.drop_deadline (shared with the verl ceiling): verl's
        window (max_off_policy_threshold versions) in wall time = ASRT_TICKET_DROP_STEPS x the LOWER-QUARTILE step time
        (stalled steps - the thing being fixed - inflate the median), floor ASRT_TICKET_DROP_FLOOR_S (60 s, above a
        normal group's admission -> finish), no warm-up deadline unless ASRT_TICKET_DROP_EARLY_S is set.  Measured: with
        cuts every arm trained to a slightly lower reward than without (they remove search-heavy groups), so correct
        training data comes first; the cut is budgeted, not free."""
        return SC.drop_deadline(self.step_s, "auto" if _defer_on() else "")

    def _drop_overdue(self):
        """ opt-in (ASRT_TICKET_DROP_S=<s>|auto).  A group of batch m that has been running longer than the deadline while finished
        ticket-(m+1) groups wait is cut from batch m: a finished younger group takes its place (one-for-one, like the
        slip) and the straggler's result is discarded when it arrives - it could only be trained beyond the ceiling
        (e.g. question-bound slow tools: 9-11 calls x 40-100 s per trajectory vs ~30 s steps; hedging re-sends to the
        same slow tool).  The prompt is skipped, not regenerated (it would
        be just as slow); counted in stats['dropped'] (lock held)."""
        dl = self.drop_deadline()
        if SC.ratio_p() is not None:
            dl = SC.slow_s()  # ratio mode: a group may be cut once it is slow (>= T), warm-up included
        m = self.m
        nxt = self.buf.get(m + 1) or []
        # DEFER: a deferred straggler can reach the ceiling in a batch whose next ticket cannot be admitted yet
        # (it waits for this hand-out) - then free the slot instead, a fresh group is admitted into batch m
        refill = _defer_on() and not nxt
        now = time.time()
        if dl is None or self.pending.get(m, 0) <= 0 or (not nxt and not refill):
            return
        late = [k for k in self.pend_keys.get(m, ()) if self.own_age(k, now) > dl]
        # guard: at most max(1, ASRT_TICKET_DROP_MAXFRAC (2%) x batch) groups cut per batch, so a bad deadline cannot
        # reshape the training distribution; beyond that the batch waits as usual (warned once per batch)
        cap = max(1, int(float(os.environ.get("ASRT_TICKET_DROP_MAXFRAC", "0.02")) * self.B))
        room = cap - self.drop_in_batch.get(m, 0)
        if _defer_on():  # deferral gathers stragglers of several batches into one: budget over the run instead.
            # Fairness rule: cuts <= the baseline's slow-not-trained count, in GROUPS over the WHOLE run.
            # verl never trains the slow groups inside the run (every trained traj has spans 1 / worst staleness <= 1):
            # measured ~1% of groups.  Budget = max(ASRT_TICKET_DROP_MIN=3, ASRT_TICKET_DROP_RUNFRAC=1% x
            # rollout_batch_size x num_rollout), available from the start (the slow groups come early, verl's too).
            total = (self.n_rollout or (m + 1)) * self.B
            room = SC.cut_budget(total) - self.n_dropped  # ASRT_TICKET_DROP_BUDGET=<n> overrides the formula
            p_ratio = SC.ratio_p()
            if p_ratio is not None:  # ratio:<p> - budget = p x slow groups so far; only slow groups may be cut
                T = SC.slow_s()
                # only the batch being collected: a later-ticket group can sit in the pipeline >= T without being slow
                # (counting those overcounted ~10x); finished groups are counted in take() by duration
                for k in self.pend_keys.get(m, ()):
                    if self.own_age(k, now) >= T:   # own run time (a finished-but-unpulled group stops at completion)
                        self.slow_keys.add(k)
                late = [k for k in late if k in self.slow_keys]
                room = int(p_ratio * len(self.slow_keys) + 1e-9) - self.n_dropped
        if late and room <= 0:
            if self.drop_in_batch.get(("warned", m)) is None:
                self.drop_in_batch[("warned", m)] = True
                self.drop_guarded += 1
                print(f"[asrt] ticket drop guard: batch {m} already cut {self.drop_in_batch.get(m, 0)} groups (cap {cap}); "
                      f"{len(late)} more overdue groups are waited for", flush=True)
                _log({"event": "ticket_drop_guard", "t": now, "batch": m, "cut": self.drop_in_batch.get(m, 0),
                      "overdue": len(late)})
            return
        late = late[:max(0, room)]
        for k in (late if refill else late[:len(nxt)]):
            if refill:  # slot back to admission: next_ticket() sees total[m] < B
                self.count[m] = self.count.get(m, 0) - 1
                self.total[m] = self.total.get(m, 0) - 1
            else:
                gid, g = nxt.pop(0)
                kg = _gkey(g)
                self.ticket_of[kg] = m
                self.buf.setdefault(m, []).append((gid, g))
                for d in (self.count, self.total):
                    d[m + 1] = d.get(m + 1, 0) - 1
            self.ticket_of.pop(k, None)
            self.pend_keys[m].discard(k)
            self.pending[m] -= 1
            self.dropped.add(k)
            self.n_dropped += 1
            self.drop_in_batch[m] = self.drop_in_batch.get(m, 0) + 1
            print(f"[asrt] ticket cut: batch {m} group (sample {k[1] if k[0] == 'i' else '?'}) after "
                  f"{self.own_age(k, now):.0f} s at the ceiling; cut {self.n_dropped} so far "
                  f"(compare: verl slow-not-trained in the same window)", flush=True)
            _log({"event": "ticket_drop", "t": now, "batch": m, "age_s": round(self.own_age(k, now), 1),
                  "deadline_s": round(dl, 1) if dl is not None else None, "sample_index": k[1] if k[0] == "i" else None})  # which question was cut
        if not nxt and (m + 1) in self.buf:
            del self.buf[m + 1]

    def _defer(self):
        """ ASRT_TICKET_DEFER=1: verl-style batching (v1 ReplayBuffer samples ANY finished groups; a slow
        group does not hold the batch, it keeps running and lands in a later batch, or is evicted once it exceeds the
        version window).  Batch m waits for stragglers while later tickets already have finished groups: each
        straggler swaps with a finished group of the EARLIEST later ticket t it can still be trained in,
        t - start_rel <= S (start_rel = lower bound of its earliest version, so its staleness stays <= S); the
        finished group moves into batch m.  Ticket sizes are unchanged, and a deferred group can be deferred again.
        A straggler no later ticket can take is at the ceiling: it is waited for (default), or cut by the
        overdue deadline with ASRT_TICKET_DROP_S=auto and counted as dropped - the ceiling S=2 is stricter than verl's 4-version
        window, so that is where the two still differ.  Lock held."""
        m = self.m
        if self.pending.get(m, 0) <= 0:
            return
        later = [t for t in sorted(self.buf) if t > m and self.buf[t]]
        if not later:
            return
        for k in sorted(self.pend_keys.get(m, ()), key=lambda k: self.t_admit.get(k, 0)):  # oldest first
            sr = self.start_rel.get(k)
            t2 = next((t for t in later if self.buf.get(t) and sr is not None and t - sr <= self.S), None)
            if t2 is None:
                continue
            gid, g = self.buf[t2].pop(0)
            self.ticket_of[_gkey(g)] = m
            self.buf.setdefault(m, []).append((gid, g))
            self.ticket_of[k] = t2
            self.pend_keys[m].discard(k)
            self.pend_keys.setdefault(t2, set()).add(k)
            self.pending[m] -= 1
            self.pending[t2] = self.pending.get(t2, 0) + 1
            self.deferred += 1
            _log({"event": "ticket_defer", "t": time.time(), "batch": m, "to": t2,
                  "sample_index": k[1] if k[0] == "i" else None})  # which questions get deferred
            if not self.buf[t2]:
                del self.buf[t2]
                later.remove(t2)
            if not later:
                break

    def buffered(self) -> int:
        return sum(len(v) for v in self.buf.values())

    def ledger(self) -> dict:
        """ ticket accounting check for the batch just handed out (lock held, before m moves):
        its pending must be 0, and for every ticket pending == its pending keys, nothing negative."""
        m = self.m
        mism = sorted(t for t in set(self.pending) | set(self.pend_keys)
                      if self.pending.get(t, 0) != len(self.pend_keys.get(t, ())))
        neg = sorted(t for d in (self.pending, self.count, self.total) for t, c in d.items() if c < 0)
        rec = {"batch": m, "pending_m": self.pending.get(m, 0), "pend_keys_m": len(self.pend_keys.get(m, ())),
               "count_m": self.count.get(m, 0), "total_m": self.total.get(m, 0),
               "pending_later": {str(t): c for t, c in sorted(self.pending.items()) if t > m and c},
               "mismatch": mism, "negative": neg, "cut_running": len(self.dropped),
               "cut_requeue_dropped": self.cut_requeue_dropped, "cut_readmit_ignored": self.cut_readmit_ignored}
        rec["ok"] = rec["pending_m"] == 0 and rec["pend_keys_m"] == 0 and not mism and not neg
        if self.resumed is not None or self.resume_version_bad or self.resume_m_bad:   #
            rec["resume_version_bad"], rec["resume_m_bad"] = self.resume_version_bad, self.resume_m_bad
            rec["ok"] = rec["ok"] and not self.resume_version_bad and not self.resume_m_bad
        return rec

    def batch_done(self):
        with self.lock:
            lg = self.ledger()
            if not lg["ok"]:
                self.ledger_bad += 1
            _log({"event": "ticket_ledger", "t": time.time(), **lg})
            print(f"[asrt] ticket ledger {'OK' if lg['ok'] else 'BAD'}: " + json.dumps(lg, sort_keys=True), flush=True)
            self.m += 1
            self.handed.clear()   # the dynamic filter runs inside the collection that handed them: no refund later
            now = time.time()
            if self._t_last_batch is not None:
                self.step_s.append(now - self._t_last_batch)
            self._t_last_batch = now

    def stats(self):
        with self.lock:
            return {"m": self.m, "rel_version": self.rel(), "v_eng": self.v_eng, "vmax": self.vmax, "v0": self.v0, "live": {t: c for t, c in self.count.items() if c},
                    "pending": {t: c for t, c in self.pending.items() if c}, "buffered": {t: len(v) for t, v in self.buf.items()},
                    "admitted": self.admitted, "refused": self.refused, "stale_requeue": self.stale_requeue, "deferred": self.deferred, "slow_groups": len(self.slow_keys), "dropped": self.n_dropped, "drop_guarded": self.drop_guarded,
                    "filter_refunds": self.filter_refunds, "refreshed": self.refreshed, "peak_live": self.peak_live,
                    "cut_requeue_dropped": self.cut_requeue_dropped, "cut_readmit_ignored": self.cut_readmit_ignored,
                    "cut_running": len(self.dropped), "ledger_bad": self.ledger_bad}

    # ---------------------------------------------------------------- checkpoint save / resume
    # slime saves the fully-async worker as {"ready": completed groups in its output queue} (fully_async_rollout.py
    # state_dict, after pause(drain): nothing in flight) next to the data-source cursor.  take() has already moved
    # every completed group into self.buf, so those groups were not saved although the cursor is past them (one batch
    # of prompts skipped per resume), and the book (m, pending, tickets, v0) restarted at zero (ledger BAD,
    # pending_m -493 in 929s).  The book is saved with the worker; the buf groups go back into "ready" (in ticket
    # order, so a reader without asrt still gets them) and count as admitted-not-taken again.
    _VOLATILE = ("lock", "v_eng", "_t_last_batch")

    def snapshot(self):
        """(book, extra ready groups).  Caller: the worker is paused and drained (slime's state_dict checks it)."""
        with self.lock:
            d = {}
            for k, v in self.__dict__.items():
                if k in self._VOLATILE:
                    continue
                if isinstance(v, (dict, set, list, collections.deque)):
                    v = copy.copy(v)
                d[k] = v
            d["pend_keys"] = {t: set(s) for t, s in self.pend_keys.items()}
            vs = [x for x in (self.vmax, self.v_eng) if x is not None]
            d["resume_floor"] = max(vs) if vs else self.resume_floor
            extra = []
            if "buf" in d:
                for t in sorted(self.buf):
                    for gid, g in self.buf[t]:
                        extra.append((gid, g))
                        k = _gkey(g)
                        d["pend_keys"].setdefault(t, set()).add(k)
                        d["pending"][t] = d["pending"].get(t, 0) + 1
                d["buf"] = {}
            d["__cls__"] = type(self).__name__
            return d, extra

    def restore(self, d, rid=None):
        """Book from snapshot(); the saved ready groups (incl. the former buf) are back in slime's output queue and
        go through take() again.  rid = the rollout slime asks for now (must equal the restored batch counter)."""
        with self.lock:
            n_rollout = self.n_rollout
            keep = {k: getattr(self, k) for k in ("B", "S") if hasattr(self, k)}   # this run's settings win
            for k, v in d.items():
                if k == "__cls__" or k in self._VOLATILE or k in keep:
                    continue
                setattr(self, k, v)
            self.n_rollout = n_rollout if n_rollout is not None else self.n_rollout
            for k, v in keep.items():
                if d.get(k, v) != v:
                    print(f"[asrt] ticket resume: {k} was {d.get(k)} at the save, {v} now - the new value is used", flush=True)
            if d.get("B", self.B) != self.B:   # tickets are batches: a different batch size breaks the saved book
                self.resume_m_bad += 1
                print(f"[asrt] ticket resume: rollout_batch_size changed {d.get('B')} -> {self.B}; ledger BAD", flush=True)
            self.v_eng, self._t_last_batch = None, None
            if rid is not None and int(rid) != self.m:
                self.resume_m_bad += 1
                print(f"[asrt] ticket resume: restored batch counter m={self.m} but slime asks for rollout {rid}; "
                      f"ledger BAD", flush=True)
            self.resumed = {"mode": "book", "m": self.m, "rid": rid, "v0": self.v0, "vmax": self.vmax,
                            "floor": self.resume_floor, "buffered_restored": sum(len(s) for s in self.pend_keys.values())}
            return self.resumed

    def adopt_without_book(self, rid, queued):
        """Resume from a checkpoint saved without the ticket book (slime's own fully-async state, or this module in
        count mode): the restored completed groups are adopted as tickets m, m+1, ... (handed out a batch EARLIER
        than planned: staleness only lower).  queued: [(gid, group)] restored into slime's output queue.
        Returns info."""
        with self.lock:
            self.m = int(rid)
            self.v0_off = int(rid)      # first sync after the resume publishes v0 + rid (slime_resume)
            info = {"mode": "no_book", "m": self.m, "restored_ready": len(queued)}
            # adopt: oldest first, B per ticket from m (no version budget: they were admitted for a ticket >= m)
            t = self.m
            for gid, g in queued:
                while self.total.get(t, 0) >= self.B:
                    t += 1
                k = _gkey(g)
                self.ticket_of[k] = t
                self.pend_keys.setdefault(t, set()).add(k)
                for dct in (self.count, self.total, self.pending):
                    dct[t] = dct.get(t, 0) + 1
                self.gidx[k] = _group_index(g)
                self.admitted += 1
            info["adopted"] = len(queued)
            self.resumed = info
            return info


def _group_index(group):
    try:
        x = group[0]
        x = x[0] if isinstance(x, list) else x
        return getattr(x, "group_index", None)
    except Exception:  # noqa: BLE001
        return None


_TK = [None]


def defer_line() -> str:
    """ startup line of the ticket reader's straggler handling (DEFER / cut)."""
    return (f"[asrt] ticket defer: {'on' if _defer_on() else 'off'} (ASRT_TICKET_DEFER="
            f"{os.environ.get('ASRT_TICKET_DEFER', '<default 1>')}); straggler cut ASRT_TICKET_DROP_S="
            f"{os.environ.get('ASRT_TICKET_DROP_S', 'auto' if _defer_on() else '<off>')}")


def _defer_on() -> bool:
    # default on (verl-style batching): measured ~15-25% faster steps than cut-only on a slow-tool workload, same
    # number of untrained groups as verl, staleness ceiling held.  NOTE: on another cluster DEFER/cut was 15-20% SLOWER;
    # the benefit is workload/cluster dependent (status: undecided).  ASRT_TICKET_DEFER=0 = plain ticket order (cut off unless ASRT_TICKET_DROP_S is set).
    return os.environ.get("ASRT_TICKET_DEFER", "1").strip().lower() in ("1", "on", "true")


class TicketBuffer:
    """data_buffer stand-in for ticket mode (see Tickets)."""

    def __init__(self, inner, tk: Tickets):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_tk", tk)

    def get_samples(self, num_samples):
        with self._tk.lock:
            ok = self._tk.next_ticket() is not None
        if not ok:
            self._tk.refused += 1
            return []
        out = (self._inner.get_samples(num_samples) or []) if num_samples > 0 else []
        if out:
            self._refresh(out)
            self._tk.admit(out)
        return out

    def _refresh(self, groups):
        """ Keep a pristine copy of every fresh group's inputs (first
        admission, before generation mutates them - like slime's own retry copy).  Tickets.restart_if_stale (under the
        ticket lock, inside admit) puts it back when a requeued group's retained samples can no longer meet S."""
        tk = self._tk
        for g in groups:
            if _versions(g):
                continue
            k = _gkey(g)
            if k not in tk.fresh:
                try:
                    tk.fresh[k] = copy.deepcopy(g)
                except Exception:
                    pass

    def add_samples(self, samples):
        # release first; groups cut from their batch do not go back to slime's buffer
        keep = self._tk.requeued(samples)
        return self._inner.add_samples(keep) if keep else None

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __setattr__(self, k, v):
        setattr(self._inner, k, v)


_RID = [None]          # rollout id of the generate() call in progress (the worker starts inside the first)


def _resume_tickets(w, tk, raw):
    """ worker start() after slime restored it from a checkpoint (load_state_dict ran first).  Book saved
    with it -> restore it; checkpoint without a book (plain slime / count mode) -> Tickets.adopt_without_book."""
    if not getattr(w, "_asrt_loaded", False):
        return None
    rid = _RID[0]
    st = getattr(w, "_asrt_restore", None) or {}
    book = st.get("book")
    if book is not None and book.get("__cls__") == type(tk).__name__:
        info = tk.restore(book, rid)
        _vfirst[0] = st.get("vfirst", _vfirst[0])
    elif book is not None:
        info = {"mode": "none", "why": f"book of a {book.get('__cls__')} reader, this run uses {type(tk).__name__}"}
        tk.resumed = info
        tk.resume_m_bad += 1
        print(f"[asrt] ticket resume WARNING: {info['why']} (ASRT_STALE_MODE changed?); ledger BAD", flush=True)
    else:
        with w.output_queue.mutex:
            queued = list(w.output_queue.queue)
        info = tk.adopt_without_book(rid if rid is not None else 0, queued)
        try:  # prompts drawn before the save that are neither in the restored queue nor slime's requeue buffer
            lost = int(raw.sample_group_index) - int(rid) * tk.B - len(queued) - len(getattr(raw, "buffer", []) or [])
        except Exception:  # noqa: BLE001
            lost = None
        info["lost_estimate"] = lost
        print(f"[asrt] ticket resume (checkpoint without an asrt ticket book): {json.dumps(info, sort_keys=True)}"
              + (f"; WARNING about {lost} prompts drawn before the save are lost" if lost else ""), flush=True)
    if rid is not None:
        _stale_n[0] = int(rid)
    _log({"event": "resume", "t": time.time(), "rollout": rid, **{k: v for k, v in (info or {}).items()}})
    if info and info.get("mode") == "book":
        print(f"[asrt] ticket resume: {json.dumps(info, sort_keys=True, default=str)}", flush=True)
    return info


_worker_patched = False
_CTR = [None]          # live AdmitCounter (count mode)
_COLLECT = {"on": False, "deferred": 0}   # groups popped during the current batch collection
_ACTIVE = {}           # reader -> group budget the staleness ceiling is enforced with (verified after rollout 0)


def _install_filter_refund(far):
    """Ticket mode + slime's dynamic filter (DAPO-style, e.g.
    check_reward_nonzero_std): a rejected group gives its slot back to its ticket (Tickets.refund), otherwise tickets
    m..m+S fill with groups that never train and admission deadlocks (seen: a rollout stuck at 26/32 groups).
    Wraps call_dynamic_filter in every loaded module that holds slime's function (the collector looks it up as a
    module global; a custom ASRT_ADMIT_INNER may have imported it by name). """
    import sys
    try:
        from slime.rollout.filter_hub import base_types as BT
    except Exception:
        BT = None
    orig = getattr(BT, "call_dynamic_filter", None) or getattr(far, "call_dynamic_filter", None)
    if orig is None or getattr(orig, "_asrt_filter_refund", False):
        return

    def call_dynamic_filter(fn, args_, group, *a, **kw):
        v = orig(fn, args_, group, *a, **kw)
        try:
            tk = _TK[0]
            if tk is not None and not getattr(v, "keep", True) and hasattr(tk, "refund"):
                tk.refund(group)
        except Exception as e:  # never break the collector
            print(f"[asrt] ticket filter refund failed: {e!r}", flush=True)
        return v
    call_dynamic_filter._asrt_filter_refund = True
    call_dynamic_filter.__wrapped__ = orig
    n = 0
    for m in list(sys.modules.values()):
        try:
            if getattr(m, "call_dynamic_filter", None) is orig:
                setattr(m, "call_dynamic_filter", call_dynamic_filter)
                n += 1
        except Exception:
            pass
    print(f"[asrt] slime_admit: dynamic-filter rejects refund their ticket slot ({n} module(s))", flush=True)


def install_worker_cap(args) -> bool:
    """Patch AsyncRolloutWorker.start / get_completed_groups (before the worker exists)."""
    global _worker_patched
    if _worker_patched or "cap" not in opts():
        return _worker_patched
    from slime.rollout import fully_async_rollout as far

    W = far.AsyncRolloutWorker
    orig_start, orig_get = W.start, W.get_completed_groups
    S = max_staleness()
    if S is not None:  # groups outstanding <= (ahead + 1) x batch
        # (S-1) batches under the overlap driver, S in slime's serial order.  S batches under the overlap
        # driver were tried: staleness 3 on 3-9% of the samples in 7 of 11 rollouts -> reverted.  The count cap cannot sit exactly on max 2: it is
        # max 1 (safe, slow on long tails) or max 3; ASRT_STALE_MODE=ticket is the exact one.
        ahead = float(max(1, S - 1) - 1) if _overlap_driver() else float(S - 1)
    else:
        ahead = float(os.environ.get("ASRT_ADMIT_AHEAD", "1"))

    def start(self):
        if _ticket_mode() and not isinstance(self.data_buffer, TicketBuffer):
            tk = Tickets(self.args.rollout_batch_size, S)
            tk.n_rollout = getattr(self.args, "num_rollout", None)   # run length for the whole-run cut budget
            tk.start_engine_poll(self.args)
            _TK[0] = tk
            _install_filter_refund(far)
            _ACTIVE["single-process(ticket)"] = (S + 1) * tk.B
            raw = self.data_buffer
            self.data_buffer = TicketBuffer(self.data_buffer, tk)
            _resume_tickets(self, tk, raw)   # no-op unless slime restored this worker from a checkpoint
            print(f"[asrt] slime_admit ticket mode: batch {tk.B}, max staleness {S} verl units (group of batch t "
                  f"starts only while t - (v_now - v0) <= {S}, handed out in batch t); slime in-flight cap "
                  f"{self.concurrency} groups", flush=True)
            print(defer_line(), flush=True)
            return orig_start(self)
        if not _ticket_mode() and not isinstance(self.data_buffer, GatedBuffer):
            self._asrt_ctr = AdmitCounter(self.args.rollout_batch_size, ahead)
            _CTR[0] = self._asrt_ctr
            _ACTIVE["single-process"] = self._asrt_ctr.cap
            self.data_buffer = GatedBuffer(self.data_buffer, self._asrt_ctr)
            if getattr(self, "_asrt_loaded", False):   # restored completed groups are admitted, not handed
                with self._asrt_ctr.lock:
                    self._asrt_ctr.admitted += self.output_queue.qsize()
                print(f"[asrt] slime_admit cap resume: {self.output_queue.qsize()} restored completed groups count as "
                      f"outstanding", flush=True)
            print(f"[asrt] slime_admit cap: outstanding groups <= {self._asrt_ctr.cap} "
                  f"(batch {self._asrt_ctr.batch}, ahead {ahead}, max staleness {S} verl units, overlap driver "
                  f"{_overlap_driver()}); slime in-flight cap {self.concurrency}",
                  flush=True)
        return orig_start(self)

    def get_completed_groups(self, limit=None):
        tk = _TK[0] if isinstance(self.data_buffer, TicketBuffer) else None
        if tk is not None and not getattr(_tls, "eager", False):
            return tk.take(orig_get(self, None), limit)
        out = orig_get(self, limit)
        ctr = getattr(self, "_asrt_ctr", None)
        if ctr is not None and out and not getattr(_tls, "eager", False):
            if _COLLECT["on"] and max_staleness() is not None:
                # slime's collector pops groups one by one while it waits for the batch; counting them at
                # once let new groups start during a long collection (long tail / drift: some workloads reached staleness
                # 3-4 at 1 batch outstanding).  Under the ceiling they count as handed when the whole batch is out.
                with ctr.lock:
                    _COLLECT["deferred"] += len(out)
            else:
                ctr.add("handed", len(out))  # slime's collector pops at hand-out; eager counts at hand-out (below)
        return out

    orig_cb = W._make_done_cb

    def _make_done_cb(self, gid, input_group):
        cb = orig_cb(self, gid, input_group)

        def _cb(done_task):
            cb(done_task)
            tk = _TK[0]
            if tk is not None:
                try:
                    tk.finished(done_task.result())
                except BaseException:
                    pass

        return _cb

    orig_qs = W.queue_size

    def queue_size(self):
        tk = _TK[0] if isinstance(self.data_buffer, TicketBuffer) else None
        n = orig_qs(self)
        if tk is not None:
            with tk.lock:
                n += tk.buffered()
        return n

    # slime saves this worker as {"ready": its output queue} (fully_async_rollout.py state_dict) and
    # restores it before start() (DataSource.register_consumer -> load_state_dict).  Ticket mode keeps completed groups
    # in Tickets.buf, outside that queue: save them and the ticket book too.
    orig_sd, orig_ld = getattr(W, "state_dict", None), getattr(W, "load_state_dict", None)

    def state_dict(self):
        st = orig_sd(self)   # raises unless paused and drained
        tk = _TK[0] if isinstance(self.data_buffer, TicketBuffer) else None
        if tk is not None:
            book, extra = tk.snapshot()
            st["ready"] = list(extra) + list(st["ready"])
            st["asrt"] = {"v": 1, "mode": "ticket", "book": book,
                          "stale_n": _stale_n[0], "vfirst": _vfirst[0], "buf_saved": len(extra)}
            _log({"event": "resume_save", "t": time.time(), "m": tk.m, "ready": len(st["ready"]),
                  "buf_saved": len(extra), "v0": tk.v0, "vmax": tk.vmax, "floor": book.get("resume_floor")})
            print(f"[asrt] slime_admit save: batch counter {tk.m}, {len(st['ready'])} completed groups "
                  f"({len(extra)} from the ticket buffer), versions v0 {tk.v0} max {tk.vmax}", flush=True)
        elif getattr(self, "_asrt_ctr", None) is not None:
            st["asrt"] = {"v": 1, "mode": "count", "stale_n": _stale_n[0], "vfirst": _vfirst[0]}
        return st

    def load_state_dict(self, state):
        object.__setattr__(self, "_asrt_restore", state.get("asrt"))
        object.__setattr__(self, "_asrt_loaded", True)
        return orig_ld(self, state)

    if orig_sd is not None and orig_ld is not None:
        W.state_dict, W.load_state_dict = state_dict, load_state_dict
    W.start, W.get_completed_groups, W._make_done_cb, W.queue_size = start, get_completed_groups, _make_done_cb, queue_size
    _worker_patched = True
    return True


def _worker(mgr):
    return getattr(getattr(mgr, "data_source", None), "consumers", {}).get("fully_async")


# ----------------------------------------------------------------------------------------------- eager batch build
class _Prebuild:
    def __init__(self, rid: int, version):
        self.rid, self.version = rid, version
        self.lock = threading.Lock()
        self.started = self.cancel = False
        self.done = threading.Event()
        self.out = self.err = None
        self.t_sched = time.time()
        self.t_start = self.build_s = None


def _eager_allowed(mgr, rid: int) -> str | None:
    a = mgr.args
    if getattr(a, "rollout_data_transport", None) == "straw":
        return "straw"
    if getattr(mgr, "recovery", None) is not None:
        return "recovery"
    if getattr(a, "dynamic_sampling_filter_path", None) or getattr(a, "rollout_sample_filter_path", None):
        return "filters"
    if getattr(a, "eval_interval", None) is not None:
        return "eval"
    if getattr(a, "offload_rollout", False) or getattr(a, "colocate", False) or getattr(a, "debug_rollout_only", False):
        return "offload/colocate/debug"
    if rid >= getattr(a, "num_rollout", 1 << 30):
        return "last"
    try:  # checkpoint after train(rid-1) snapshots the worker queue: do not pop it before that
        from slime.utils.misc import should_run_periodic_action

        if should_run_periodic_action(rid - 1, getattr(a, "save_interval", None), None, getattr(a, "num_rollout", None)):
            return "save"
    except Exception:
        pass
    from slime.rollout.fully_async_rollout import AsyncRolloutWorker

    if not isinstance(_worker(mgr), AsyncRolloutWorker):
        return "worker"
    return None


def _eager_main(mgr, pb: _Prebuild, orig_generate):
    w = _worker(mgr)
    need = mgr.args.rollout_batch_size
    while True:
        with pb.lock:
            if pb.cancel:
                pb.done.set()
                return
            if w.queue_size() >= need:
                pb.started = True
                break
        if w.error is not None:
            pb.done.set()
            return
        time.sleep(0.02)
    pb.t_start = time.time()
    _tls.eager, _tls.version = True, (pb.version + 1 if pb.version is not None else None)
    try:
        mgr.batch_builder.rollout_id = pb.rid
        pb.out = orig_generate(mgr, pb.rid, pb.t_start)
    except BaseException as e:  # surfaced by the generate call that takes it
        pb.err = e
    finally:
        _tls.eager = False
        pb.build_s = time.time() - pb.t_start
        pb.done.set()


_mgr_patched = False


def install_manager_eager() -> bool:
    """Patch RolloutManager._generate / _get_rollout_data on every live class copy (Ray binds remote methods at
    actor creation, so ``generate`` itself cannot be patched; ``_generate`` is called through ``self``)."""
    global _mgr_patched
    if _mgr_patched or "eager" not in opts():
        return _mgr_patched
    from ._hooks import patch_method
    from .slime_prefetch import _live_classes

    classes = _live_classes("RolloutManager", "_generate")
    if not classes:
        return False

    def make_generate(orig):
        def _generate(self, rollout_id, start_time):
            if getattr(_tls, "eager", False):  # the prebuild thread itself
                return orig(self, rollout_id, start_time)
            t_ask = time.time()
            w = _worker(self)
            ctr = getattr(w, "_asrt_ctr", None)
            rec = {"rollout": rollout_id, "t_ask": t_ask, "queue": w.queue_size() if w else None,
                   "inflight": getattr(w, "active", None), "outstanding": ctr.outstanding() if ctr else None}
            pb = getattr(self, "_asrt_pb", None)
            out = None
            if pb is not None and pb.rid != rollout_id:  # not expected in slime's loop; never drop popped groups
                with pb.lock:
                    pb.cancel = not pb.started
                if pb.started:
                    raise RuntimeError(f"asrt prebuild for rollout {pb.rid} but rollout {rollout_id} was asked")
                pb = None
            if pb is not None:
                with pb.lock:
                    if not pb.started:
                        pb.cancel = True
                if pb.started:
                    pb.done.wait()
                    if pb.err is not None:
                        raise pb.err
                    out = pb.out
                    if pb.version is not None:
                        self.weight_version = pb.version + 1
                    if ctr is not None:
                        ctr.add("handed", self.args.rollout_batch_size)
                    rec.update(prebuilt=True, build_s=pb.build_s, ready_before_ask_s=max(0.0, t_ask - (pb.t_start + pb.build_s)),
                               join_s=time.time() - t_ask)
                else:
                    rec.update(prebuilt=False, why="not enough completed groups when asked")
            else:
                rec.update(prebuilt=False, why="none scheduled")
            self._asrt_pb = None
            if out is None:
                t0 = time.time()
                out = orig(self, rollout_id, start_time)
                rec["build_s"] = time.time() - t0  # incl. waiting for generation
            rec["weight_version"] = self.weight_version
            rec["handout_s"] = time.time() - t_ask
            # schedule the next prebuild (batch rollout_id + 1)
            why = _eager_allowed(self, rollout_id + 1)
            if why is None:
                nxt = _Prebuild(rollout_id + 1, self.weight_version)
                self._asrt_pb = nxt
                threading.Thread(target=_eager_main, args=(self, nxt, orig), name="asrt-prebuild", daemon=True).start()
            else:
                rec["next_prebuild_off"] = why
            if ctr is not None:
                rec.update(admitted=ctr.admitted, requeued=ctr.requeued, handed=ctr.handed, cap=ctr.cap)
            _log(rec)
            return out

        return _generate

    def make_grd(orig):
        def _get_rollout_data(self, rollout_id):
            data, metrics = orig(self, rollout_id=rollout_id)
            if getattr(_tls, "eager", False) and getattr(_tls, "version", None) is not None:
                self.weight_version = _tls.version  # the version training will hold for this batch
            return data, metrics

        return _get_rollout_data

    n = 0
    for C in classes:
        n += patch_method(C, "_generate", make_generate)
        if "_get_rollout_data" in C.__dict__:
            patch_method(C, "_get_rollout_data", make_grd)
    _mgr_patched = n > 0
    print(f"[asrt] slime_admit eager prebuild installed on {n} RolloutManager class(es)", flush=True)
    return _mgr_patched


# ----------------------------------------------------------------------------------------------- staleness log
_stale_patched = False
_stale_n = [0]


_vfirst = [None]


def staleness_summary(samples, version, cap, rollout=None) -> dict | None:
    """verl units: (version that trains this batch) - (earliest weight version of the sample).  The training version
    of rollout k is first_version + k (one weight update per step; = slime's reference when it hands out after the
    sync), never below slime's own reference or any version a sample already carries."""
    if version is not None and _vfirst[0] is None:
        # version of rollout 0 (one sync per rollout); a resumed run first records rollout = start id
        _vfirst[0] = int(version) - (int(rollout) if rollout is not None else 0)
    ref = int(version) if version is not None else None
    if _vfirst[0] is not None and rollout is not None:
        ref = max(ref if ref is not None else -1, _vfirst[0] + int(rollout))
    ages, ahead, slime_ages = [], 0, []
    for x in samples:
        try:
            vs = [int(v) for v in (x.weight_versions or [])]
        except (TypeError, ValueError):
            continue
        if not vs or ref is None:
            continue
        if version is not None and max(vs) <= int(version):
            slime_ages.append(int(version) - min(vs))
        r = ref
        if max(vs) > r:
            ahead += 1
            r = max(vs)
        ages.append(r - min(vs))
    if not ages:
        return None
    # token-level: train version - the group's START version (first-admission lower bound, see
    # Tickets.sr_first), not the return-time labels.  Upper bound of (train - oldest token version) per sample.
    tok = []
    tk = _TK[0]
    if tk is not None and ref is not None and tk.v0 is not None:
        for x in samples:
            gi = getattr(x, "group_index", None)
            sr = tk.sr_by_gi.get(gi) if gi is not None else None
            try:
                vs = [int(v) for v in (x.weight_versions or [])]
            except (TypeError, ValueError):
                vs = []
            if sr is None and not vs:
                continue
            st = min(([tk.v0 + sr] if sr is not None else []) + ([min(vs)] if vs else []))
            tok.append(max(ref, max(vs) if vs else ref) - st)
    over = sum(1 for a in ages if cap is not None and a > cap)
    return {"n": len(ages), "mean": round(sum(ages) / len(ages), 4), "max": max(ages),
            "frac_over_cap": round(over / len(ages), 4), "cap": cap, "train_version": ref, "slime_version": version,
            "newer_than_train": ahead,
            "slime_mean": round(sum(slime_ages) / len(slime_ages), 4) if slime_ages else None,
            "slime_max": max(slime_ages) if slime_ages else None, "slime_unknown": len(samples) - len(slime_ages),
            "tok_n": len(tok), "tok_mean": round(sum(tok) / len(tok), 4) if tok else None,
            "tok_max": max(tok) if tok else None,
            "tok_frac_over_cap": round(sum(1 for a in tok if cap is not None and a > cap) / len(tok), 4) if tok else None}


def install_staleness_log() -> bool:
    """Wrap slime.observability.rollout_metrics.compute_staleness_metrics (called once per rollout in the
    RolloutManager with the batch samples and the version they are scored against)."""
    global _stale_patched
    if _stale_patched or "staleness" not in opts():
        return _stale_patched
    try:
        from slime.observability import rollout_metrics as rm
    except Exception as e:  # older slime: no per-rollout staleness metric
        print(f"[asrt] slime_admit staleness log unavailable: {e!r}", flush=True)
        _stale_patched = True
        return False
    orig = rm.compute_staleness_metrics

    def compute_staleness_metrics(samples, current_weight_version):
        out = orig(samples, current_weight_version)
        try:
            S = max_staleness()
            rid = _stale_n[0]
            _stale_n[0] += 1
            rec = staleness_summary(samples, current_weight_version, S, rid) if current_weight_version is not None else None
            if rec is not None:
                rec["rollout"] = rid
                out["staleness/frac_over_cap"] = rec["frac_over_cap"]
                if rec.get("tok_max") is not None:   # token-level (start-version) basis
                    out["staleness/tok_max"] = rec["tok_max"]
                    out["staleness/tok_mean"] = rec["tok_mean"]
                    out["staleness/tok_frac_over_cap"] = rec["tok_frac_over_cap"]
                print(SC.staleness_line(rid, rec['mean'], rec['max'], S, rec['frac_over_cap'], f" (n {rec['n']}, token-level "
                      f"max {rec.get('tok_max')} mean {rec.get('tok_mean')} over {rec.get('tok_frac_over_cap')}; train version "
                      f"{rec['train_version']}; slime metric mean {rec['slime_mean']} max {rec['slime_max']} "
                      f"unknown {rec['slime_unknown']})"),
                      flush=True)
                _log({"event": "staleness", "rollout": rid, "t": time.time(), **rec})
        except Exception as e:  # never break the metric path
            print(f"[asrt] slime_admit staleness log failed: {e!r}", flush=True)
        return out

    rm.compute_staleness_metrics = compute_staleness_metrics
    _stale_patched = True
    return True


# ----------------------------------------------------------------------------------------------- distributed reader
_dist_patched = False


def install_distributed_cap() -> bool:
    """ The ceiling for slime's DISTRIBUTED fully-async reader (fully_async_distributed.DistributedRollout,
    used with --rollout-data-transport straw on multi-node rollout).  There every node runs a _GenerationActor whose
    slots keep generating (each slot holds at most one finished group) and the collector prefetches up to the whole
    group capacity, so outstanding groups reach ~2x capacity (measured 3.8 batches each -> staleness 7-9) and the
    worker-level cap above never sees them.  Bound it the same way: total slot capacity = the ceiling's group budget
    ((S-1) or S batches, see ``staleness``) via the per-engine concurrency the reader sizes its pools from, and no
    collector prefetch beyond the batch being collected.  Only while the reader is built (args restored after)."""
    global _dist_patched
    S = max_staleness()
    if _dist_patched or S is None:
        return _dist_patched
    try:
        from slime.rollout import fully_async_distributed as fad
    except Exception:
        return False
    DR, RS = fad.DistributedRollout, fad.RolloutScheduler
    orig_init, orig_gen = DR.__init__, RS.generate

    def __init__(self, args, data_source, *a, **kw):
        import math

        B, n = int(args.rollout_batch_size), max(1, int(getattr(args, "n_samples_per_prompt", 1)))
        target = (max(1, S - 1) if _overlap_driver() else S) * B
        try:
            engines = int(fad.get_rollout_num_engines(args))
        except Exception:
            from slime.utils.http_utils import get_rollout_num_engines
            engines = int(get_rollout_num_engines(args))
        old_c, old_f = args.sglang_server_concurrency, getattr(args, "fully_async_group_prefetch_factor", 1)
        new_c = max(1, math.ceil(target * n / max(1, engines)))
        changed = new_c * max(1, old_f) < old_c * max(1, old_f) or old_f != 1
        if changed:
            args.sglang_server_concurrency, args.fully_async_group_prefetch_factor = min(old_c, new_c), 1
        try:
            orig_init(self, args, data_source, *a, **kw)
        finally:
            args.sglang_server_concurrency, args.fully_async_group_prefetch_factor = old_c, old_f
        self._asrt_cap = True
        _ACTIVE["distributed"] = sum(self.capacities)
        print(f"[asrt] slime_admit distributed cap: group slots {sum(self.capacities)} (target {target} = ceiling {S} "
              f"verl units, batch {B}; per-engine concurrency {old_c} x prefetch {old_f} -> "
              f"{min(old_c, new_c) if changed else old_c} x 1), collector prefetch 0", flush=True)
        _log({"event": "distributed_cap", "t": time.time(), "slots": sum(self.capacities), "target": target,
              "conc_before": old_c, "conc_after": min(old_c, new_c) if changed else old_c})

    def generate(self, rollout_id, *, prefetch):
        if getattr(self, "_asrt_cap", False):
            prefetch = 0
        return orig_gen(self, rollout_id, prefetch=prefetch)

    DR.__init__, RS.generate = __init__, generate
    _dist_patched = True
    return True


# ----------------------------------------------------------------------------------------------- entry point
def _eager_collect(args, rollout_id, data_buffer):
    """Rollout function body inside the prebuild thread: B completed groups are already queued; pop them FIFO
    and finalize exactly like slime's collector does without filters (no worker.resume(): the driver owns
    pause/resume around weight updates)."""
    from slime.rollout.base_types import finalize_rollout_groups
    from slime.rollout.fully_async_rollout import _get_worker

    w = _get_worker(args, data_buffer)
    B = args.rollout_batch_size
    groups = [g for _, g in w.get_completed_groups(limit=B)]
    if len(groups) != B:
        raise RuntimeError(f"asrt prebuild expected {B} completed groups, got {len(groups)}")
    out = finalize_rollout_groups(args, rollout_id, groups, None, controller=getattr(data_buffer, "controller", None))
    return out.samples


_inner_fn = None


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """``--rollout-function-path`` entry: installs the patches, then runs ASRT_ADMIT_INNER."""
    global _inner_fn
    if opts():
        install_worker_cap(args)
        install_distributed_cap()
        install_manager_eager()
        install_staleness_log()
    if getattr(_tls, "eager", False):
        return _eager_collect(args, rollout_id, data_buffer)
    if _inner_fn is None:
        from slime.utils.misc import load_function

        _inner_fn = load_function(os.environ.get("ASRT_ADMIT_INNER") or DEFAULT_INNER)
    _COLLECT["on"] = not evaluation
    if not evaluation:
        _RID[0] = rollout_id
        if _TK[0] is None and _CTR[0] is None and _stale_n[0] == 0 and rollout_id:
            _stale_n[0] = int(rollout_id)   # staleness log numbers rollouts like slime after a resume
    try:
        out = _inner_fn(args, rollout_id, data_buffer, evaluation=evaluation)
    finally:
        _COLLECT["on"] = False
        if _TK[0] is not None and not evaluation:
            _TK[0].batch_done()
            _log({"event": "ticket", "rollout": rollout_id, "t": time.time(), **_TK[0].stats()})
        ctr = _CTR[0]
        if ctr is not None and _COLLECT["deferred"]:
            with ctr.lock:
                ctr.handed += _COLLECT["deferred"]
                _COLLECT["deferred"] = 0
    _verify_ceiling(args, evaluation)
    return out


_verified = [False]


def _verify_ceiling(args, evaluation=False):
    """ after the first rollout: the configured ceiling must sit on the reader that actually produced it
    (single-process AsyncRolloutWorker or the distributed straw reader), else fail instead of running uncapped."""
    S = max_staleness()
    if _verified[0] or evaluation or S is None:
        return
    _verified[0] = True
    if not _ACTIVE:
        raise RuntimeError(f"[asrt] staleness ceiling ASRT_MAX_STALENESS={S} is configured but was installed on no "
                           f"rollout reader (rollout fn {getattr(args, 'rollout_function_path', None)}, transport "
                           f"{getattr(args, 'rollout_data_transport', None)}) - refusing to run uncapped")
    B = int(getattr(args, "rollout_batch_size", 0) or 0)
    desc = ", ".join(f"{k}: {v} groups ({v / B:.2f} batches)" if B else f"{k}: {v} groups" for k, v in _ACTIVE.items())
    mode = "ticket" if _ticket_mode() else "count"
    print(SC.active_line(S, f"mode {mode}, overlap driver {_overlap_driver()}, reader {desc}"), flush=True)
    _log({"event": "ceiling_active", "t": time.time(), "S": S, "readers": dict(_ACTIVE), "batch": B, "mode": mode})


# The RolloutManager imports the rollout function in its __init__ (load_function), so installing at import time
# patches the manager before generate(0) runs; the calls in generate_rollout_fully_async are the fallback.
if opts():
    for _f in (install_worker_cap, install_distributed_cap, install_manager_eager, install_staleness_log):
        try:
            _f(None) if _f is install_worker_cap else _f()
        except Exception as _e:  # never break the framework import
            print(f"[asrt] slime_admit: {_f.__name__} at import failed: {_e!r}", flush=True)
