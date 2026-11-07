"""CPU test of ai_search_plugin.adapters.verl response_logits guard (_guard_response_logits / _rlogits_kept_rows) and
the engine hook of the `response_logits` feature. Fake response_logits package; no GPU, no verl needed.
Run: python3 tests/scripts/verl_rlogits_guard_cpu.py"""
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

# ---- fake response_logits (adapters.verl._wrap_forward_fn marks the compact path; adapters.rows_mode from env) ----
calls = []
import ai_search_plugin  # noqa: E402  (real package; its response_logits subpackage is replaced by fakes below)
rl = types.ModuleType("ai_search_plugin.response_logits"); rl.__path__ = []
ad = types.ModuleType("ai_search_plugin.response_logits.adapters"); ad.__path__ = []
ad.rows_mode = lambda: os.environ.get("RESPONSE_LOGITS_ROWS", "response")
rv = types.ModuleType("ai_search_plugin.response_logits.adapters.verl")
rv.install = lambda: (calls.append("rl_install") or {"installed": True})


def _wrap_forward_fn(fn):
    def compact(*a, **k):
        calls.append("compact")
        return fn(*a, **k)
    return compact


rv._wrap_forward_fn = _wrap_forward_fn
for m in (rl, ad, rv):
    sys.modules[m.__name__] = m
rl.adapters, ad.verl = ad, rv
ai_search_plugin.response_logits = rl

from ai_search_plugin.adapters import verl as V  # noqa: E402

assert V._guard_response_logits() is True


def forward_model_engine(model, input_ids, multi_modal_inputs, logits_processor=None, logits_processor_args=None):
    calls.append("stock")
    return None


guarded = rv._wrap_forward_fn(forward_model_engine)


def nested(rows):
    return torch.nested.nested_tensor([torch.tensor(r) for r in rows], layout=torch.jagged)


# multi-turn batch: 2 samples, prompt 200 / 300, response 800 / 900 of which 600 / 700 are tool / search tokens
resp = [[1] * 100 + [0] * 600 + [1] * 100, [1] * 100 + [0] * 700 + [1] * 100]
ids = nested([[5] * 1000, [5] * 1200])  # 2200 tokens, response 1700 (77%), loss rows 400 (18%)


def run(lm, ram=None):
    calls.clear()
    lpa = {"label": None, "temperature": None, "loss_mask": lm, "response_attention_mask": ram}
    guarded(None, ids, {}, logits_processor=lambda *a, **k: None, logits_processor_args=lpa)
    return calls[0]


def old_guard(lm, ids, thr=0.5):  # the criterion before this change (counted every response position)
    if getattr(lm, "is_nested", False) and getattr(ids, "is_nested", False):
        return "compact" if lm.values().numel() <= thr * ids.values().numel() else "stock"
    return "compact"


lm_n = nested(resp)
R = 900
lm_p = torch.zeros(2, R, dtype=torch.bool)
ram = torch.zeros(2, R, dtype=torch.long)
for i, r in enumerate(resp):
    lm_p[i, :len(r)] = torch.tensor(r).bool(); ram[i, :len(r)] = 1

# nested multi-turn batch (only nested input is judged): the old criterion counted every response position
assert old_guard(lm_n, ids) == "stock", "old guard (nested): tool tokens counted -> tier off on a multi-turn batch"
assert run(lm_n) == "compact", "new guard keeps the tier (loss rows 400 / 2200 = 18%)"
assert V._rlogits_kept_rows(lm_n) == 400
long_ans = nested([[1] * 900, [1] * 1000])  # single-turn, long answers all with loss: 1900 / 2200 = 86% -> stock
assert run(long_ans) == "stock"
# padded loss_mask (the production verl v1 path): unchanged, always compact, also when > 50% of rows carry loss
lm_q = torch.ones(2, R, dtype=torch.bool); ram_q = torch.zeros(2, R, dtype=torch.long); ram_q[0, :800] = 1; ram_q[1, :900] = 1
assert old_guard(lm_q, ids) == "compact" and run(lm_q, ram_q) == "compact" and run(lm_p, ram) == "compact"
assert V._rlogits_kept_rows(lm_p) is None
assert run(None) == "compact"
# threshold env is read when the guard is installed: 0.9 keeps the long single-turn batch compact
V._RL.update(compact=0, stock=0)
rv._wrap_forward_fn = _wrap_forward_fn
os.environ["ASRT_VERL_RLOGITS_MAX_FRAC"] = "0.9"
assert V._guard_response_logits() is True
guarded = rv._wrap_forward_fn(forward_model_engine)
assert run(long_ans) == "compact"
os.environ.pop("ASRT_VERL_RLOGITS_MAX_FRAC")
print("guard: nested multi-turn compact (old: stock); nested long single-turn stock; padded unchanged (compact); threshold env OK")

# ---- feature hook: response_logits installs the adapter when verl's Megatron engine module is imported ----
hooks = []
V.when_imported = lambda name, fn: hooks.append((name, fn))
V._install_response_logits()
assert [n for n, _ in hooks] == ["verl.workers.engine.megatron.transformer_impl"], hooks
calls.clear()
hooks[0][1](types.ModuleType("verl.workers.engine.megatron.transformer_impl"))
assert calls == ["rl_install"], calls
print("feature hook: response_logits adapter installed on the Megatron engine import OK")
print("ALL OK")
