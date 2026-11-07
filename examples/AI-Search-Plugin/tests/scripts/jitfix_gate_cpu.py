"""CPU test: slime_jitfix skips itself on a dense model (7B / 14B dense models: eager
replacement of Megatron @jit_fuser made the train segment ~10-12% slower) and still applies on MoE / GDN models or
with ASRT_JITFIX_FORCE=1. A fake megatron module with one Dynamo-wrapped function stands in for @jit_fuser.
Run: python3 tests/scripts/jitfix_gate_cpu.py"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ai_search_plugin.adapters import slime_jitfix as J  # noqa: E402


def fake_megatron():
    def orig(x):
        return x + 1

    def wrapped(x):
        return orig(x)
    wrapped._torchdynamo_orig_callable = orig
    m = types.ModuleType("megatron.fakefused")
    m.bias_swiglu = wrapped
    sys.modules["megatron.fakefused"] = m
    return m, orig


def hf(cfg):
    d = tempfile.mkdtemp(prefix="asrt-jitgate-")   # left in place (no deletes in tests)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def main():
    os.environ.pop("ASRT_JITFIX_FORCE", None)
    dense = types.SimpleNamespace(num_experts=None, hf_checkpoint=hf({"model_type": "qwen3", "num_hidden_layers": 40}))
    moe = types.SimpleNamespace(num_experts=128, hf_checkpoint=None)
    gdn = types.SimpleNamespace(num_experts=None, hf_checkpoint=hf({"text_config": {"linear_num_value_heads": 32,
                                                                                    "linear_num_key_heads": 16}}))
    hf_moe = types.SimpleNamespace(num_experts=None, hf_checkpoint=hf({"n_routed_experts": 64}))
    assert J.model_kind(dense) == "dense" and J.model_kind(moe) == "moe" and J.model_kind(gdn) == "gdn"
    assert J.model_kind(hf_moe) == "moe" and J.model_kind(None) is None

    m, orig = fake_megatron()
    rep = J.install("eager", args=dense)
    assert rep.get("skipped") and not rep["active"] and m.bias_swiglu is not orig, rep
    for a in (moe, gdn, None):        # MoE / GDN / unknown (no args, old callers): applied as before
        m, orig = fake_megatron()
        rep = J.install("eager", args=a)
        assert rep["active"] and m.bias_swiglu is orig, (a, rep)
    os.environ["ASRT_JITFIX_FORCE"] = "1"
    m, orig = fake_megatron()
    rep = J.install("eager", args=dense)
    assert rep["active"] and m.bias_swiglu is orig and rep["model"] == "dense", rep
    os.environ.pop("ASRT_JITFIX_FORCE")
    print("jitfix model gate cpu test OK")


if __name__ == "__main__":
    main()
