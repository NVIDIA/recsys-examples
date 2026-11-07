# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""hybridep_overlap chunked path on CPU tensors (needs torch; skipped otherwise).

A single-rank stand-in for HybridEPBuffer (dispatch = gather rows per expert, combine = scatter-add back) checks the
autograd wiring of the chunk schedule: output and the grads of hidden states, router probs and expert weights equal
an unchunked dense reference for 1..4 chunks (uneven chunk sizes included), the buffer sees fixed slots and saved
handles, and the patched MoELayer.forward (plain and under activation checkpointing) matches the reference.  The CUDA
stream ordering itself needs a GPU and is not covered here."""
import types

import pytest

torch = pytest.importorskip("torch")

from _hybridep_fakes import FakeMoELayerBase, install_fake_megatron  # noqa: E402
from ai_search_plugin.adapters import megatron_hybridep as mh  # noqa: E402

E, H, K = 4, 6, 2


class SingleRankHybridEP:
    """EP size 1: dispatch_with_permute sorts the routed (token, expert) pairs by expert; combine sums them back."""

    def __init__(self, num_experts):
        self.E = num_experts
        self.calls = []

    def dispatch_with_permute(self, hidden, routing_map=None, probs=None, scaling_factor=None,
                              num_of_experts_per_rank=None, pad_multiple=None, num_permuted_tokens=None,
                              non_blocking=False, num_of_tokens_per_rank=None, handle=None):
        if handle is None:
            assert num_of_experts_per_rank == self.E and hidden.shape[0] <= num_of_tokens_per_rank
            self.calls.append(("dispatch", hidden.shape[0], num_of_tokens_per_rank))
            rows = [routing_map[:, e].nonzero().flatten() for e in range(self.E)]
            idx = torch.cat(rows)
            eid = torch.cat([torch.full((len(r),), e, dtype=torch.long) for e, r in enumerate(rows)])
            counts = torch.tensor([len(r) for r in rows], dtype=torch.int32)
            handle = (idx, eid, hidden.shape[0])
            return hidden[idx], probs[idx, eid], None, counts, handle
        idx, eid, _ = handle
        assert num_permuted_tokens == len(idx)
        self.calls.append(("dispatch_bwd", len(idx)))
        return hidden[idx], None, None, None, None

    def combine_with_unpermute(self, hidden, probs=None, handle=None):
        idx, eid, T = handle
        self.calls.append(("combine", T))
        out = torch.zeros(T, hidden.shape[1], dtype=hidden.dtype).index_add_(0, idx, hidden)
        hp = None
        if probs is not None:
            hp = torch.zeros(T, self.E, dtype=probs.dtype)
            hp[idx, eid] = probs
        return out, hp


class Experts:
    """Per-expert linear map scaled by the routing prob (the prob is applied in the experts, as in Megatron)."""

    def __init__(self, W):
        self.W = W
        self.seen = []

    def __call__(self, x, tokens_per_expert, probs):
        self.seen.append(tokens_per_expert.tolist())
        ys, o = [], 0
        for e, c in enumerate(tokens_per_expert.tolist()):
            ys.append((x[o:o + c] @ self.W[e]) * probs[o:o + c, None])
            o += c
        return torch.cat(ys), None


def _inputs(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, dtype=torch.float64, generator=g)
    logits = torch.randn(T, E, generator=g)
    top = logits.topk(K, dim=-1).indices
    rmap = torch.zeros(T, E, dtype=torch.bool).scatter_(1, top, True)
    probs = torch.softmax(logits, -1) * rmap
    W = torch.randn(E, H, H, dtype=torch.float64, generator=g)
    return x, rmap, probs, W


def _reference(x, rmap, probs, W):
    y = torch.einsum("th,ehk->tek", x, W)                       # every token through every expert
    return (y * (probs.to(y.dtype) * rmap)[..., None]).sum(1)


def _leaves(*ts):
    return [t.detach().clone().requires_grad_(True) for t in ts]


def _layer(W, tp_size=1):
    return types.SimpleNamespace(token_dispatcher=types.SimpleNamespace(tp_size=tp_size, num_local_experts=E,
                                                                        tp_ep_group=None),
                                 config=types.SimpleNamespace(), experts=Experts(W))


@pytest.mark.parametrize("T,n", [(13, 1), (13, 2), (13, 3), (16, 4), (3, 4)])
def test_chunked_matches_dense_reference(T, n):
    x0, rmap, p0, W0 = _inputs(T, seed=T * 10 + n)
    x, p, W = _leaves(x0, p0, W0)
    xr, pr, Wr = _leaves(x0, p0, W0)
    buf = SingleRankHybridEP(E)
    lay = _layer(W)
    out = mh.chunked_hybridep(lay, x, rmap, p, n, max_tokens=64, buffer=buf)
    ref = _reference(xr, rmap, pr, Wr)
    torch.testing.assert_close(out, ref)
    g = torch.randn_like(ref)
    out.backward(g)
    ref.backward(g)
    torch.testing.assert_close(x.grad, xr.grad)
    torch.testing.assert_close(p.grad, pr.grad.to(p.grad.dtype))
    torch.testing.assert_close(W.grad, Wr.grad)
    # schedule seen by the buffer: n dispatches into fixed slots, n combines, n dispatches of grads with saved handles
    d = [c for c in buf.calls if c[0] == "dispatch"]
    assert [c[1] for c in d] == mh.split_sizes(T, n) and {c[2] for c in d} == {mh.chunk_slots(64, n)}
    assert sum(c[0] == "combine" for c in buf.calls) == 2 * n           # forward combines + dispatch backward
    assert sum(c[0] == "dispatch_bwd" for c in buf.calls) == n
    assert len(lay.experts.seen) == n


def test_forward_order_dispatch_next_before_combine_prev():
    x0, rmap, p0, W0 = _inputs(12)
    buf = SingleRankHybridEP(E)
    mh.chunked_hybridep(_layer(W0), x0, rmap, p0, 3, max_tokens=64, buffer=buf)
    assert [c[0] for c in buf.calls] == ["dispatch", "dispatch", "combine", "dispatch", "combine", "combine"]


def test_guards():
    x0, rmap, p0, W0 = _inputs(10)
    with pytest.raises(RuntimeError, match="ASRT_HYBRIDEP_MAX_TOKENS"):
        mh.chunked_hybridep(_layer(W0), torch.zeros(65, H, dtype=x0.dtype), torch.zeros(65, E, dtype=torch.bool),
                            torch.zeros(65, E), 2, max_tokens=64, buffer=SingleRankHybridEP(E))
    with pytest.raises(RuntimeError, match="bool routing map"):
        mh.chunked_hybridep(_layer(W0), x0, rmap.long(), p0, 2, max_tokens=64, buffer=SingleRankHybridEP(E))
    with pytest.raises(RuntimeError, match="expert tensor parallel"):
        mh.chunked_hybridep(_layer(W0, tp_size=2), x0, rmap, p0, 2, max_tokens=64, buffer=SingleRankHybridEP(E))


# ------------------------------------------------------------------------------- patched MoELayer.forward
class FakeMoELayer(FakeMoELayerBase):
    def __init__(self, W, router, recompute=False):
        self.config = types.SimpleNamespace(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep",
                                            moe_expert_capacity_factor=None, moe_hybridep_num_sms_preprocessing=108)
        self.training, self.shared_expert_overlap, self.moe_layer_recompute = True, False, recompute
        self.token_dispatcher = types.SimpleNamespace(tp_size=1, num_local_experts=E, tp_ep_group=None)
        self.experts = Experts(W)
        self.router = router

    def route(self, hidden_states, padding_mask=None):
        logits = hidden_states.reshape(-1, H).float() @ self.router
        top = logits.topk(K, dim=-1).indices
        rmap = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, top, True)
        return torch.softmax(logits, -1) * rmap, rmap

    def shared_experts_compute(self, hidden_states):
        return hidden_states * 0.5

    def postprocess(self, output, shared):
        return output.view(self.token_dispatcher.hidden_shape) + shared


@pytest.fixture
def patched(monkeypatch):
    mh.uninstall()
    buf = SingleRankHybridEP(E)
    ns = install_fake_megatron(monkeypatch, moe_layer_cls=FakeMoELayer, buffer=buf)
    mh.install(2, max_tokens=64)
    yield ns, buf
    mh.uninstall()


@pytest.mark.parametrize("recompute", [False, True])
def test_patched_forward_matches_reference(patched, recompute):
    ns, buf = patched
    g = torch.Generator().manual_seed(3)
    W0 = torch.randn(E, H, H, dtype=torch.float64, generator=g)
    router = torch.randn(H, E, generator=g)
    hs0 = torch.randn(5, 2, H, dtype=torch.float64, generator=g)        # [s, b, H]
    W, hs = _leaves(W0, hs0)
    Wr, hsr = _leaves(W0, hs0)
    layer = FakeMoELayer(W, router, recompute=recompute)
    out, bias = layer.forward(hs)
    assert bias is None and out.shape == hs.shape
    probs, rmap = layer.route(hsr)
    ref = _reference(hsr.reshape(-1, H), rmap, probs, Wr).view(hs.shape) + hsr * 0.5
    torch.testing.assert_close(out, ref)
    out.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(hs.grad, hsr.grad)
    torch.testing.assert_close(W.grad, Wr.grad)
    assert mh.snapshot()["calls"] == 1 and mh.snapshot()["fallback"] == 0
    assert ns.tensor_parallel.calls == ([False] if recompute else [])
    assert sum(c[0] == "dispatch" for c in buf.calls) == 2


def test_patched_forward_falls_back(patched):
    ns, buf = patched
    layer = FakeMoELayer(torch.zeros(E, H, H), torch.zeros(H, E))
    layer.config.moe_hybridep_routing_map_mode = "indices"
    assert layer.forward(torch.zeros(2, 1, H)) == ("orig", None)
    assert mh.snapshot()["fallback_why"] == {"index routing map": 1} and not buf.calls
