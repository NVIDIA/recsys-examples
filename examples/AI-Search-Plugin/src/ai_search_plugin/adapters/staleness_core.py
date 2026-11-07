"""Framework-independent parts of the asrt staleness ceiling (shared by slime_admit and verl_ceiling).

Unit (both frameworks): verl's ``training/off_policy/trajectory_staleness_worst`` = (weight version that TRAINS the
sample) - (EARLIEST weight version the sample was generated with).  With one weight update per step, verl trains step
s on version s-1, so a trajectory whose earliest version is v may be trained at step s only while (s-1) - v <= S,
i.e. up to step ``deadline(v, S) = v + S + 1``.  slime's ticket form: batch t (trained at v0 + t) may start a group at
relative version r only while t - r <= S.  Both are ``within(train_version, start_version, S)``.

Env knobs (same names for every framework):
  ASRT_MAX_STALENESS=S           ceiling (unset / off / 0 -> none)
  ASRT_TICKET_DROP_S=<s>|auto|0  cut a straggler that holds a batch at the ceiling after <s> seconds since it started
                                 (auto = max(ASRT_TICKET_DROP_FLOOR_S 60, ASRT_TICKET_DROP_STEPS 4 x lower-quartile step)
                                 after 3 measured steps, ASRT_TICKET_DROP_EARLY_S before; 0/off = never)
  ASRT_TICKET_DROP_MIN=3, ASRT_TICKET_DROP_RUNFRAC=0.01   whole-run cut budget max(MIN, RUNFRAC x batch x steps)
  ASRT_TICKET_DROP_BUDGET=<n>    explicit whole-run budget in groups (e.g. the baseline's measured untrained count);
                                 unset = the formula above
Logs: $ASRT_LOG_DIR/admit.jsonl (one JSON record per line).
"""
from __future__ import annotations

import json
import os


def max_staleness() -> int | None:
    """ASRT_MAX_STALENESS=S (>=1) -> hard ceiling; unset / off / 0 -> None."""
    v = os.environ.get("ASRT_MAX_STALENESS", "").strip().lower()
    if v in ("", "off", "none", "0", "no"):
        return None
    return max(1, int(v))


def stale_mode() -> str:
    """ASRT_STALE_MODE: 'ticket' (default) or 'count' (old group-count cap).  Only slime_admit / slime_pool read it;
    the verl ceiling has its own ticket reader and ignores it."""
    return os.environ.get("ASRT_STALE_MODE", "ticket").strip().lower()


def within(train_version: int, start_version: int, S: int, slack: int = 0) -> bool:
    """staleness (train_version - start_version) <= S - slack (slack clipped to S)."""
    return train_version - start_version <= S - min(slack, S)


def deadline(start_version: int, S: int) -> int:
    """Last verl global step that may train a trajectory first generated with weights `start_version`."""
    return int(start_version) + int(S) + 1


def worst(global_steps: int, min_global_steps: int) -> int:
    """verl trainer_base: trajectory_staleness_worst = (global_steps - 1) - min_global_steps."""
    return (int(global_steps) - 1) - int(min_global_steps)


def drop_deadline(step_s, default: str = "") -> float | None:
    """Seconds after which a straggler at the ceiling may be cut (see module doc); None = never.
    step_s: recent batch hand-out intervals (s).  default: value when ASRT_TICKET_DROP_S is unset."""
    v = os.environ.get("ASRT_TICKET_DROP_S", default).strip().lower()
    if v in ("", "0", "off"):
        return None
    if v == "auto":
        floor = float(os.environ.get("ASRT_TICKET_DROP_FLOOR_S", "60"))
        if len(step_s) < 3:
            e = os.environ.get("ASRT_TICKET_DROP_EARLY_S", "").strip()
            return float(e) if e else None
        # lower quartile, not median: stalled steps (the thing being fixed) inflate the median
        return max(floor, float(os.environ.get("ASRT_TICKET_DROP_STEPS", "4")) * sorted(step_s)[len(step_s) // 4])
    return float(v)


def cut_budget(total_groups: int) -> int:
    """Whole-run cut budget in groups: ASRT_TICKET_DROP_BUDGET if set, else max(ASRT_TICKET_DROP_MIN (3),
    ASRT_TICKET_DROP_RUNFRAC (1%) x total_groups) - the baseline's slow-not-trained share."""
    b = os.environ.get("ASRT_TICKET_DROP_BUDGET", "").strip()
    if b:
        return max(0, int(b))
    return max(int(os.environ.get("ASRT_TICKET_DROP_MIN", "3")),
               int(float(os.environ.get("ASRT_TICKET_DROP_RUNFRAC", "0.01")) * total_groups))


def log_admit(rec: dict):
    """Append one record to $ASRT_LOG_DIR/admit.jsonl (no-op without ASRT_LOG_DIR; never raises)."""
    d = os.environ.get("ASRT_LOG_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "admit.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def active_line(S: int, detail: str) -> str:
    """The startup line every capped reader prints once it is verified installed (grepped by the staleness audit)."""
    return f"[asrt] staleness ceiling ACTIVE: S={S} (verl units: train version - earliest sample version), {detail}"


def staleness_line(rollout, mean, mx, S, frac_over, tail: str = "") -> str:
    """Per-rollout / per-step staleness line in the format the staleness audit parses."""
    return (f"[asrt] staleness rollout {rollout} (train version - earliest sample version): mean {mean:.3f} "
            f"max {mx} over cap {S}: {100 * frac_over:.1f}%{tail}")


# Cut accounting by SLOW-GROUP SHARE.  A group is slow when it has been running
# (admission -> last trajectory, or still running) longer than ASRT_TICKET_SLOW_S (300 s, same T as the verl-side
# accounting).  ASRT_TICKET_DROP=ratio:<p> (p = the baseline's untrained slow groups / its slow groups, measured on
# the baseline for the same workload): cuts <= floor(p x our slow groups so far),
# and only a group that has itself reached T may be cut (a group cut earlier would not count as slow).
def slow_s() -> float:
    return float(os.environ.get("ASRT_TICKET_SLOW_S", "300"))


def ratio_p():
    v = os.environ.get("ASRT_TICKET_DROP", "").strip().lower()
    if v.startswith("ratio:"):
        try:
            return float(v.split(":", 1)[1])
        except ValueError:
            return None
    return None
