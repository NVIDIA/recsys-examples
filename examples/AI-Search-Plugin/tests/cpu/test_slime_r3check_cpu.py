"""fast R3 validation == slime's: passes the same tensors and raises the same errors (slime's own function is used
for anything irregular). Uses a local copy of slime's checker semantics when slime is not importable."""
from types import SimpleNamespace

import pytest
import torch

from ai_search_plugin.adapters.slime_r3check import make_fast

calls = {"orig": 0}


def orig(experts, args, *, sample_index=None, expected_rows=None):  # minimal stand-in with slime's error classes
    calls["orig"] += 1
    if experts.ndim != 3 or tuple(experts.shape[1:]) != (args.num_layers, args.moe_router_topk):
        raise ValueError("shape")
    if expected_rows is not None and experts.shape[0] != expected_rows:
        raise ValueError("rows")
    m = experts.to(torch.int64)
    if bool(((m < 0) | (m >= args.num_experts)).any()):
        raise ValueError("range")
    present = (m != 0).any(-1)
    if bool((~present.any(0)).any()):
        raise ValueError("missing layer")


ARGS = SimpleNamespace(num_layers=4, moe_router_topk=3, num_experts=16, moe_layer_freq=[1, 1, 1, 1])


def test_good_tensor_skips_original():
    calls["orig"] = 0
    f = make_fast(orig)
    x = torch.randint(1, 16, (100, 4, 3), dtype=torch.int16)
    assert f(x, ARGS, sample_index=0, expected_rows=100) is None and calls["orig"] == 0


@pytest.mark.parametrize("bad,err", [("range", "range"), ("layer", "missing layer"), ("rows", "rows")])
def test_irregular_goes_to_original(bad, err):
    f = make_fast(orig)
    x = torch.randint(1, 16, (100, 4, 3), dtype=torch.int32)
    rows = 100
    if bad == "range":
        x[5, 1, 2] = 16
    elif bad == "layer":
        x[:, 2, :] = 0
    else:
        rows = 99
    with pytest.raises(ValueError, match=err):
        f(x, ARGS, sample_index=0, expected_rows=rows)
