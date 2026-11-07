"""verl_offpol counts submitted / trained / dropped on a fake v1 ReplayBuffer (no verl needed).
Run: python3 tests/scripts/verl_offpol_cpu.py"""
import io, contextlib, json, types
from ai_search_plugin.adapters import verl_offpol as vo

class Batch:
    def __init__(self, keys): self.keys = keys

class RB:
    max_off_policy_threshold = 4; max_off_policy_strategy = "drop"
    def __init__(self):
        self.partitions = {"train": {"a_0_0": {}, "a_1_0": {}, "b_0_0": {}, "b_1_0": {}, "c_0_0": {}}}
        self.prompt_global_steps = {"train": {"a": 1, "b": 5, "c": 6}}
    def _evict_terminal_groups(self, global_steps, partition_id, reasons):
        for u in reasons[0]:
            for k in [k for k in self.partitions[partition_id] if k.startswith(u + "_")]:
                del self.partitions[partition_id][k]
        return reasons[0], len(reasons[0]), 0, {}
    def sample(self, global_steps, partition_id, batch_size):
        self._evict_terminal_groups(global_steps, partition_id, ({"a"}, set(), set(), None))
        return Batch(["b_0_0", "b_1_0", "c_0_0"]), {}

mod = types.ModuleType("fake_rb"); RB.__module__ = "fake_rb"; mod.RB = RB
assert vo._patch_class(RB)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    RB().sample(6, "train", 2)
rec = json.loads(buf.getvalue().split("[asrt] verl offpol ", 1)[1])
assert rec["dropped_stale_groups"] == 1 and rec["dropped_stale_traj"] == 2, rec
assert rec["trained_groups"] == 2 and rec["trained_traj"] == 3 and rec["submitted_groups_new"] == 3, rec
assert rec["dropped_span_max"] == 6 and abs(rec["drop_share"] - 0.4) < 1e-9, rec
print("ok", rec)

# ReplayBufferAsync shape: a subclass with its own sample() must be patched too (the flag is not inherited)
class RBA(RB):
    def sample(self, global_steps, partition_id, batch_size):
        return Batch(["c_0_0"]), {}
assert vo._patch_class(RBA), "subclass skipped"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    RBA().sample(7, "train", 1)
assert "[asrt] verl offpol " in buf.getvalue(), "subclass sample not wrapped"
print("ok subclass")
