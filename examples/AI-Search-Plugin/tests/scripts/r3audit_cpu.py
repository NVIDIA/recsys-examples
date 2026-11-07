"""slime_r3check audit: counts rows, row mismatches and fully-missing route rows; result unchanged.
Run: python3 tests/scripts/r3audit_cpu.py"""
import os, sys, types, io, contextlib, json
import torch
os.environ.update(ASRT_R3AUDIT="1", ASRT_R3AUDIT_EVERY="2", ASRT_R3CHECK="")
calls = []
def orig(experts, args, *, sample_index=None, expected_rows=None):
    calls.append(experts.shape[0]); return None
m = types.ModuleType("slime.utils.routed_experts"); m.validate_routed_experts_tensor = orig
for name in ("slime", "slime.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["slime.utils.routed_experts"] = m
from ai_search_plugin.adapters import slime_r3check as rc
assert rc.install()
v = m.validate_routed_experts_tensor
args = types.SimpleNamespace(num_layers=2, moe_router_topk=2)
a = torch.randint(1, 8, (5, 2, 2), dtype=torch.int16)
b = a.clone(); b[3] = 0                     # one token without any route
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    assert v(a, args, expected_rows=5) is None and v(b, args, expected_rows=6) is None
rec = json.loads(buf.getvalue().split("[asrt] r3audit ", 1)[1])
assert rec["samples"] == 2 and rec["rows"] == 10 and rec["hole_rows"] == 1 and rec["row_mismatch"] == 1, rec
assert calls == [5, 5]
print("ok", rec)
