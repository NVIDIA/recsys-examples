"""slime_pool.plan: resize only when the in-flight group pool is below one batch."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ai_search_plugin.adapters.slime_pool import plan  # noqa: E402


def test_n16_single_node_resized():
    p = plan(64, 2, 16, 32, 128)
    assert p["changed"] and p["conc"] == 128 and p["groups_after"] >= 32 and p["groups_before"] == 8


def test_batch_sized_pool_untouched():
    for a in [(68, 4, 4, 68, 128), (68, 4, 16, 17, 128), (128, 8, 4, 68, 128)]:
        p = plan(*a)
        assert not p["changed"] and p["groups_after"] == p["groups_before"]


def test_never_lowers_and_caps_at_max_running():
    p = plan(64, 2, 16, 32, 96)
    assert p["conc"] == 96 and p["groups_after"] >= 32
    assert plan(200, 2, 16, 64, 128)["conc"] == 200


def test_override_and_no_max_running():
    p = plan(64, 2, 16, 32, None)
    assert p["conc"] == 64 and p["groups_after"] >= 32
    assert plan(64, 2, 16, 32, 128, sem_override=192)["conc"] == 192


# staleness ceiling (ASRT_MAX_STALENESS) budget
def test_cap_budget_glm_single_engine():
    assert plan(128, 1, 4, 68, 128)["groups_after"] == 96          # no ceiling: unchanged (async mode)
    p = plan(128, 1, 4, 68, 128, cap=1.0)                          # S=2 + overlap driver: 1 batch outstanding
    assert p["groups_after"] == 64 and p["groups_after"] <= p["cap_groups"] == 68
    p = plan(128, 1, 4, 68, 128, cap=2.0)                          # S=2, serial order: 2 batches
    assert p["groups_after"] == 96 and p["groups_after"] <= 136


def test_cap_never_exceeded():
    import itertools
    for conc, E, n, rbs, mr, cap in itertools.product((16, 64, 128), (1, 2, 4, 8), (1, 4, 8, 16), (17, 32, 68),
                                                       (None, 128, 512), (1.0, 2.0, 3.0)):
        p = plan(conc, E, n, rbs, mr, cap=cap)
        assert p["groups_after"] <= max(p["groups_before"], p["cap_groups"]), (conc, E, n, rbs, mr, cap, p)
        assert p["conc"] >= conc


def test_cap_helpers(monkeypatch):
    from ai_search_plugin.adapters.slime_pool import cap_batches, max_staleness, plan
    monkeypatch.setenv("ASRT_STALE_MODE", "count")          # count cap: (ahead + 1) batches
    assert cap_batches(None, True) is None and cap_batches(2, True) == 1.0 and cap_batches(2, False) == 2.0
    assert cap_batches(1, True) == 1.0 and cap_batches(3, True) == 2.0
    monkeypatch.setenv("ASRT_STALE_MODE", "ticket")         # ticket mode, fill default on: tickets m..m+S live
    monkeypatch.delenv("ASRT_POOL_FILL", raising=False)
    assert cap_batches(2, True) == 3.0 and cap_batches(2, False) == 3.0
    monkeypatch.setenv("ASRT_POOL_FILL", "off")             # off: the old max(1, S-1)-batch pool
    assert cap_batches(2, True) == 1.0
    # 4 engines, conc 68, n 4, batch 68, max_running 128: fill the engines -> 128 groups (<= 3 batches)
    p = plan(68, 4, 4, 68, 128, 1, 1.0, 0, cap=3.0, fill_engines=True)
    assert p["conc"] == 128 and p["groups_after"] == 128 and p["groups_after"] <= p["cap_groups"], p
    assert not plan(68, 4, 4, 68, 128, 1, 1.0, 0, cap=3.0)["changed"]   # without fill: unchanged (one batch)
    monkeypatch.setenv("ASRT_MAX_STALENESS", "off")
    assert max_staleness() is None
    monkeypatch.setenv("ASRT_MAX_STALENESS", "2")
    assert max_staleness() == 2
