"""The staleness ceiling and its log use a group's START version, not SGLang's return-time labels.

SGLang labels each generate call with the weight version at RETURN (meta_info weight_version).  A call that runs across
a weight sync (slime flush 0 / pause in place / kvepoch / kvfresh retract-resume) is labelled newer than its oldest
tokens.  Token-level compliance : train version - oldest version of any token <= S.
  1. (the deprecated ooo mode's checks are not ported)
  2. ticket: a requeued group whose retained sample was labelled newer than its first admission must be judged by
     the first-admission bound (old code: re-admitted into a batch S+1 after its oldest token).
  3. the staleness log reports the token-level (start-version) basis next to the label basis.
  4. fresh groups / labels older than admission: unchanged behaviour (labels can only LOWER the start).
usage: python3 tests/scripts/vstart_cpu.py"""
import copy
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.environ.get("VSTART_TEST_NO_PATH"):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
os.environ.update(ASRT_MAX_STALENESS="2", ASRT_TICKET_DROP="ratio:1", ASRT_TICKET_SLOW_S="1000",
                  ASRT_TICKET_POLL="0")
os.environ.pop("ASRT_PREFETCH", None)
os.environ.pop("ASRT_TICKET_DEFER", None)
from ai_search_plugin.adapters import slime_admit as sa  # noqa: E402

N = 2


def mk(gi, labels=None):
    return [types.SimpleNamespace(index=gi * N + j, group_index=gi,
                                  weight_versions=[] if labels is None else [str(v) for v in labels]) for j in range(N)]


def label(g, v):
    for x in g:
        x.weight_versions = [str(v)]




def test_ticket_requeue_uses_first_admission():
    os.environ["ASRT_STALE_MODE"] = "ticket"
    tk = sa.Tickets(1, 2)
    tk.v0 = tk.vmax = 0
    g = mk(0)
    tk.fresh[sa._gkey(g)] = copy.deepcopy(g)   # TicketBuffer._refresh keeps the pristine inputs
    tk.admit([g])                             # ticket 0, start 0
    assert tk.ticket_of[sa._gkey(g)] == 0
    label(g, 2)                               # a retained sample continued across 2 syncs, labelled 2 at return
    tk.requeued([g])                          # aborted later (e.g. a flush-1 sync), back to slime's buffer
    tk.m = 3
    tk.vmax = 3
    tk.admit([g])
    k = sa._gkey(g)
    # must not train the retained tokens (version 0) in batch >= 3: restart from the pristine copy instead
    assert tk.refreshed == 1, "retained tokens from version 0 would be trained at batch 3 (staleness 3 > S)"
    assert not sa._versions(g), "restarted group carries no old samples"
    assert tk.ticket_of[k] - tk.start_rel[k] <= tk.S


def test_fresh_and_older_labels_unchanged():
    os.environ["ASRT_STALE_MODE"] = "ticket"
    tk = sa.Tickets(2, 2)
    tk.v0 = tk.vmax = 0
    tk.m = 1
    tk.vmax = 1
    g = mk(1)
    tk.admit([g])
    k = sa._gkey(g)
    assert tk.start_rel[k] == tk.rel() == 1          # fresh: start = rel() at admission, as before
    h = mk(2, labels=[0])                            # requeued group with an OLDER label than rel: label lowers it
    tk.admit([h])
    assert tk.start_rel[sa._gkey(h)] == 0


def test_staleness_log_token_basis():
    os.environ["ASRT_STALE_MODE"] = "ticket"
    tk = sa.Tickets(1, 2)
    tk.v0 = tk.vmax = 0
    sa._TK[0] = tk
    try:
        g = mk(0)
        tk.admit([g])                  # start 0
        label(g, 2)                    # labelled 2 at return
        sa._vfirst[0] = None           # module state: first version seen by the staleness log
        rec = sa.staleness_summary(g, 2, 2, rollout=0)   # train version 2
        assert rec["max"] == 0, rec    # label basis: 2 - 2
        assert rec["tok_max"] == 2, rec   # token basis: 2 - 0
        assert rec["tok_frac_over_cap"] == 0.0
    finally:
        sa._TK[0] = None


def test_ticket_requeue_before_v0_known():
    """With no labelled sample finished yet (v0 None) the first-admission bound must still
    hold for a requeued group (it was dropped when the start was passed as an absolute version)."""
    os.environ["ASRT_STALE_MODE"] = "ticket"
    tk = sa.Tickets(1, 2)
    g = mk(0)
    tk.fresh[sa._gkey(g)] = copy.deepcopy(g)
    tk.admit([g])                      # v0 None, start 0
    for x in g:                        # retained partial tokens, no version label
        x.tokens = [1, 2, 3]
    tk.requeued([g])
    tk.m = 3
    tk.admit([g])
    k = sa._gkey(g)
    assert tk.refreshed == 1 or tk.ticket_of[k] - tk.start_rel[k] <= tk.S, (tk.ticket_of[k], tk.start_rel[k])


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    print("ALL OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
