import sys, types, os
from types import SimpleNamespace
os.environ.update(ASRT_MAX_STALENESS="2", ASRT_PREFETCH="overlap,r3c,profile")
for k in [k for k in sys.modules if k.startswith("slime")]: del sys.modules[k]
def mod(n, **a):
    m = types.ModuleType(n); m.__path__ = []; m.__dict__.update(a); sys.modules[n] = m; return m
mod("slime"); mod("slime.rollout"); mod("slime.utils")
class RS:
    def generate(self, rid, *, prefetch): return ("gen", rid, prefetch)
class DR(RS):
    def __init__(self, args, ds):
        tc = args.sglang_server_concurrency * 8 // args.n_samples_per_prompt
        self.capacities = [tc // 4 * args.fully_async_group_prefetch_factor] * 4
fad = mod("slime.rollout.fully_async_distributed", DistributedRollout=DR, RolloutScheduler=RS, get_rollout_num_engines=lambda a: 8)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from ai_search_plugin.adapters import slime_admit as sa
assert sa.install_distributed_cap()
args = SimpleNamespace(rollout_batch_size=68, n_samples_per_prompt=4, sglang_server_concurrency=128, fully_async_group_prefetch_factor=1)
d = DR(args, None)
print("slots", sum(d.capacities), "args restored", args.sglang_server_concurrency, "gen", d.generate(3, prefetch=256))
assert sum(d.capacities) <= 72 and args.sglang_server_concurrency == 128 and d.generate(3, prefetch=256)[2] == 0
print("ok")
# configured but installed nowhere -> fail at the first rollout
sa._ACTIVE.clear(); sa._verified[0] = False
try:
    sa._verify_ceiling(SimpleNamespace(rollout_batch_size=68, rollout_function_path="x", rollout_data_transport="straw"))
    raise SystemExit("expected RuntimeError")
except RuntimeError as e:
    print("refuses uncapped:", str(e)[:80])
sa._ACTIVE["distributed"] = 68; sa._verified[0] = False
sa._verify_ceiling(SimpleNamespace(rollout_batch_size=68))
print("ok2")
