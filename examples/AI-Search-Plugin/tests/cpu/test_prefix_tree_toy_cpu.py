# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prefix_tree numerics on CPU (float64, no Megatron, CP = TP = 1): tree path == flat path for loss and gradients.

Toy Megatron-like block: layers named *Layer with a ``self_attention`` sequence mixer
  - softmax attention with GQA (4 query heads, 2 KV heads), RoPE from a raw [max_seqlen] table applied per THD
    document (as Megatron does), QKV / output projections, a ``core_attention`` sub-module;
  - a gated-delta-net mixer (A_log, conv1d; causal short conv per document, then a module-global
    ``chunk_gated_delta_rule`` with FLA's varlen signature, here a naive recurrence);
  - token-wise MLP and RMS norms.
The flat path packs every sample as its own document.  The tree path uses the real ported code: the slime adapter's
``SlimeTree`` (grouping, flat<->tree row maps, RankPlan) and ``model.install_on_model`` (layer wrappers, virtual
packs, GDN state fork).  Loss = weighted squares of the block output on response rows.
"""
import sys

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from ai_search_plugin.prefix_tree import model as ptm  # noqa: E402
from ai_search_plugin.prefix_tree import plan as P  # noqa: E402
from ai_search_plugin.prefix_tree.adapters import slime as S  # noqa: E402
from ai_search_plugin.prefix_tree.adapters import verl as V  # noqa: E402

DT = torch.float64
D, HQ, HKV, DH, CONV = 16, 4, 2, 4, 4
_THIS = sys.modules[__name__]


# ---- toy modules ---------------------------------------------------------------------------------------------

def rope_table(n):
    inv = 1.0 / (10000 ** (torch.arange(0, DH, 2, dtype=DT) / DH))
    f = torch.outer(torch.arange(n, dtype=DT), inv)
    return torch.cat([f, f], -1)[:, None, None, :]          # [n, 1, 1, DH] (Megatron's raw table layout)


def apply_rope(x, freqs):                                  # x [L, H, DH], freqs [L, 1, 1, DH]
    f = freqs[:, 0]
    x1, x2 = x.chunk(2, -1)
    return x * f.cos() + torch.cat([-x2, x1], -1) * f.sin()


def docs(psp):
    cu = psp.cu_seqlens_q.tolist()
    return list(zip(cu[:-1], cu[1:]))


class ToyCore(torch.nn.Module):
    def forward(self, q, k, v, attention_mask, packed_seq_params=None):
        out = []
        r = HQ // HKV
        for a, b in docs(packed_seq_params):
            qq, kk, vv = q[a:b], k[a:b].repeat_interleave(r, 1), v[a:b].repeat_interleave(r, 1)
            s = torch.einsum("qhd,khd->hqk", qq, kk) / DH ** 0.5
            s = s + torch.triu(torch.full_like(s[0], -1e30), 1)
            out.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vv).reshape(b - a, HQ * DH))
        return torch.cat(out)


class ToyAttention(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.linear_qkv = torch.nn.Parameter(torch.randn(D, (HQ + 2 * HKV) * DH, generator=g, dtype=DT) * 0.3)
        self.linear_proj = torch.nn.Parameter(torch.randn(HQ * DH, D, generator=g, dtype=DT) * 0.3)
        self.core_attention = ToyCore()

    def forward(self, hidden_states, attention_mask=None, rotary_pos_emb=None, packed_seq_params=None):
        x = hidden_states[:, 0]
        qkv = x @ self.linear_qkv
        q = qkv[:, :HQ * DH].reshape(-1, HQ, DH)
        k = qkv[:, HQ * DH:(HQ + HKV) * DH].reshape(-1, HKV, DH)
        v = qkv[:, (HQ + HKV) * DH:].reshape(-1, HKV, DH)
        if rotary_pos_emb is not None:                     # raw table, positions restart per THD document
            qs, ks = [], []
            for a, b in docs(packed_seq_params):
                qs.append(apply_rope(q[a:b], rotary_pos_emb[:b - a]))
                ks.append(apply_rope(k[a:b], rotary_pos_emb[:b - a]))
            q, k = torch.cat(qs), torch.cat(ks)
        o = self.core_attention(q, k, v, None, packed_seq_params=packed_seq_params)
        return (o @ self.linear_proj)[:, None], None


def naive_gdr(q, k, v, g=None, beta=None, scale=None, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, cu_seqlens=None):
    """FLA-signature gated delta rule, [1, L, H, K/V], varlen via cu_seqlens; state [N, H, K, V]."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    cu = cu_seqlens.tolist()
    H, K, Vd = q.shape[2], q.shape[3], v.shape[3]
    outs, finals = [], []
    for n, (a, b) in enumerate(zip(cu[:-1], cu[1:])):
        Sm = (initial_state[n].to(q.dtype) if initial_state is not None else q.new_zeros(H, K, Vd))
        for t in range(a, b):
            kt, vt, qt = k[0, t], v[0, t], q[0, t] * scale        # [H, K], [H, V]
            Sm = torch.exp(g[0, t])[:, None, None] * Sm
            pred = torch.einsum("hk,hkv->hv", kt, Sm)
            Sm = Sm + beta[0, t][:, None, None] * torch.einsum("hk,hv->hkv", kt, vt - pred)
            outs.append(torch.einsum("hk,hkv->hv", qt, Sm))
        finals.append(Sm)
    o = torch.stack(outs)[None] if outs else v.new_zeros(1, 0, H, Vd)
    return o, (torch.stack(finals) if output_final_state else None)


chunk_gated_delta_rule = naive_gdr      # module global looked up at call time (the state mode wraps it)


class ToyGDN(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        r = lambda *s: torch.nn.Parameter(torch.randn(*s, generator=g, dtype=DT) * 0.3)
        self.A_log = r(2)
        self.conv1d = torch.nn.Conv1d(D, D, CONV, groups=D, bias=False).to(DT)
        with torch.no_grad():
            self.conv1d.weight.copy_(torch.randn(D, 1, CONV, generator=g, dtype=DT) * 0.3)
        self.conv_kernel_dim = CONV
        self.wq, self.wk, self.wv, self.wo = r(D, 2 * DH), r(D, 2 * DH), r(D, 2 * DH), r(2 * DH, D)
        self.wg, self.wb = r(D, 2), r(D, 2)

    def forward(self, hidden_states, attention_mask=None, rotary_pos_emb=None, packed_seq_params=None):
        x = hidden_states[:, 0]
        w = self.conv1d.weight[:, 0]                           # [D, CONV]
        hs = []
        for a, b in docs(packed_seq_params):                   # causal depthwise conv, zero left pad per document
            xp = torch.cat([x.new_zeros(CONV - 1, D), x[a:b]])
            hs.append(torch.stack([(xp[t:t + CONV] * w.t()).sum(0) for t in range(b - a)]))
        h = torch.nn.functional.silu(torch.cat(hs))
        L = h.shape[0]
        q = (h @ self.wq).reshape(1, L, 2, DH)
        k = torch.nn.functional.normalize((h @ self.wk).reshape(1, L, 2, DH), dim=-1)
        v = (h @ self.wv).reshape(1, L, 2, DH)
        gg = (-torch.exp(self.A_log) * torch.nn.functional.softplus(h @ self.wg))[None]
        beta = torch.sigmoid(h @ self.wb)[None]
        o, _ = _THIS.chunk_gated_delta_rule(q, k, v, g=gg, beta=beta, cu_seqlens=packed_seq_params.cu_seqlens_q)
        return (o.reshape(L, 2 * DH) @ self.wo)[:, None], None


class ToyLayer(torch.nn.Module):
    def __init__(self, mixer, g):
        super().__init__()
        self.self_attention = mixer
        self.w1 = torch.nn.Parameter(torch.randn(D, 2 * D, generator=g, dtype=DT) * 0.3)
        self.w2 = torch.nn.Parameter(torch.randn(2 * D, D, generator=g, dtype=DT) * 0.3)

    @staticmethod
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def forward(self, h, packed_seq_params, rotary_pos_emb):
        a, _ = self.self_attention(self.norm(h), attention_mask=None, rotary_pos_emb=rotary_pos_emb,
                                   packed_seq_params=packed_seq_params)
        h = h + a
        return h + torch.tanh(self.norm(h) @ self.w1) @ self.w2


class ToyBlock(torch.nn.Module):
    def __init__(self, seed=0, gdn=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(64, D, generator=g, dtype=DT))
        mixers = [ToyAttention(g), ToyGDN(g), ToyAttention(g)] if gdn else [ToyAttention(g), ToyAttention(g)]
        self.layers = torch.nn.ModuleList([ToyLayer(m, g) for m in mixers])
        self.head = torch.nn.Parameter(torch.randn(D, 3, generator=g, dtype=DT))

    def forward(self, hidden_states, packed_seq_params, rotary_pos_emb):
        h = hidden_states
        for layer in self.layers:
            h = layer(h, packed_seq_params, rotary_pos_emb)
        return h


# ---- data ---------------------------------------------------------------------------------------------------------

def make_batch(seed=0):
    """GRPO-like micro-batch: 2 groups (one with siblings sharing their first response tokens), a singleton."""
    rng = np.random.default_rng(seed)
    seqs, plens = [], []
    for P_, n, shared in ((21, 3, 0), (13, 2, 3)):
        prompt = rng.integers(1, 64, P_)
        common = rng.integers(1, 64, shared)
        for _ in range(n):
            seqs.append(np.concatenate([prompt, common, rng.integers(1, 64, int(rng.integers(2, 9)))]))
            plens.append(P_)
    seqs.insert(3, rng.integers(1, 64, 11))                   # singleton between the groups
    plens.insert(3, 6)
    lens = [len(s) for s in seqs]
    cu = np.concatenate([[0], np.cumsum(lens)]).tolist()
    pad = (-cu[-1]) % 8                                       # slime-style trailing pad segment
    if pad:
        cu.append(cu[-1] + pad)
    return seqs, plens, cu


def loss_of(y, seqs, plens, cu, head):
    """Weighted squares on the response rows (rows predicting response tokens) of the flat-layout output."""
    L = 0
    for i, (s, p) in enumerate(zip(seqs, plens)):
        r = slice(cu[i] + p - 1, cu[i] + len(s) - 1)
        L = L + ((y[r, 0] @ head).pow(2).sum()) * (i + 1)
    return L


def flat_and_tree(block, seqs, plens, cu, rope=True, min_trunk=4):
    ids = torch.as_tensor(np.concatenate(seqs + [np.zeros(cu[-1] - cu[len(seqs)], np.int64)]))
    rot = rope_table(64) if rope else None
    params = [p for p in block.parameters()]
    # flat
    x = block.emb[ids][:, None]
    psp = ptm._make_psp(cu, len(cu) - 1, max(b - a for a, b in zip(cu[:-1], cu[1:])), "cpu")
    lf = loss_of(block(x, psp, rot), seqs, plens, cu, block.head)
    gf = torch.autograd.grad(lf, params, allow_unused=True)
    # tree (ported slime adapter + model wrappers)
    tree = S.SlimeTree(seqs, plens, cu, tp=1, pad_multiple=8, device="cpu", sp=False, min_trunk=min_trunk,
                       halo=ptm.required_halo(1))
    x = block.emb[ids][:, None]
    y = tree.to_flat(block(tree.to_tree(x), tree.psp, rot))
    lt = loss_of(y, seqs, plens, cu, block.head)
    gt = torch.autograd.grad(lt, params, allow_unused=True)
    return lf, gf, lt, gt, tree


def rel(a, b):
    if a is None or b is None:
        return 0.0 if a is None and b is None else 1.0
    return float((a - b).norm() / (b.norm() + 1e-30))


@pytest.fixture
def modes():
    saved = (dict(ptm.ATTN_MODE), dict(ptm.GDN_MODE), dict(ptm.FAST_ROWS), sys.modules[__name__].chunk_gated_delta_rule)
    yield
    ptm.ATTN_MODE.clear(); ptm.ATTN_MODE.update(saved[0])
    ptm.GDN_MODE.clear(); ptm.GDN_MODE.update(saved[1])
    ptm.FAST_ROWS.clear(); ptm.FAST_ROWS.update(saved[2])
    _THIS.chunk_gated_delta_rule = saved[3]


@pytest.mark.parametrize("attn,gdn,fast,rope", [
    ("full", "full", False, True),      # default: every mixer on the virtual 'full' pack
    ("full", "state", False, True),     # GDN state fork (trunk once, branches from its final state)
    ("full", "state", True, True),      # + deterministic row-map backward
    ("core", "full", False, False),     # core mode (no RoPE): only core_attention on the virtual pack
    ("core", "state", True, True),      # core mode requested with a RoPE table -> attention layers fall back to full
])
def test_tree_equals_flat(modes, attn, gdn, fast, rope):
    ptm.ATTN_MODE["mode"], ptm.GDN_MODE["mode"], ptm.FAST_ROWS["on"] = attn, gdn, fast
    seqs, plens, cu = make_batch(0)
    block = ToyBlock(0)
    before = ptm.snapshot()
    rep = ptm.install_on_model(block, gdn_core_modules=(__name__,))
    assert rep["layers_wrapped"] == 3 and ptm.usable(), rep
    assert rep["gdn_mode"]["mode"] == gdn
    lf, gf, lt, gt, tree = flat_and_tree(block, seqs, plens, cu, rope=rope)
    # the tree really shares rows: 2 groups, fewer rows than the flat pack
    assert sum(1 for T, _ in tree.groups if T) == 2 and tree.layout.real_total < cu[len(seqs)]
    assert abs(float(lt - lf)) / abs(float(lf)) < 1e-12
    worst = max(rel(a, b) for a, b in zip(gt, gf))
    assert worst < 1e-10, worst
    after = ptm.snapshot()
    if gdn == "state":
        assert after["gdn_state_calls"] > before["gdn_state_calls"]
    if attn == "core" and not rope:
        assert after["core_calls"] > before["core_calls"]
    if attn == "core" and rope:
        assert after["attn_core_mode_calls"] == before["attn_core_mode_calls"]


def test_tree_attention_mask_reference(modes):
    """Dense tree mask + tree position ids on the tree rows == per-document attention on the flat pack."""
    seqs, plens, cu = make_batch(1)
    groups, lay, t2f, f2t, tree_pos, *_ = S.build_maps(seqs, plens, cu, tp=1, pad_multiple=8, min_trunk=4)
    assert np.array_equal(tree_pos, P.tree_position_ids(lay))
    g = torch.Generator().manual_seed(1)
    ids = torch.as_tensor(np.concatenate(seqs + [np.zeros(cu[-1] - cu[len(seqs)], np.int64)]))
    xf = torch.randn(64, HQ, DH, generator=g, dtype=DT)[ids]    # a function of the token (siblings share the prompt)
    rot = rope_table(64)
    psp = ptm._make_psp(cu, len(cu) - 1, 64, "cpu")
    qf = torch.cat([apply_rope(xf[a:b], rot[:b - a]) for a, b in docs(psp)])
    of = ToyCore()(qf, qf[:, :HKV], qf[:, :HKV], None, packed_seq_params=psp)
    xt = xf[torch.as_tensor(t2f)]
    qt = apply_rope(xt, rot[torch.as_tensor(tree_pos)])
    m = torch.as_tensor(P.tree_attention_mask(lay))
    kt = qt[:, :HKV].repeat_interleave(HQ // HKV, 1)
    s = torch.einsum("qhd,khd->hqk", qt, kt) / DH ** 0.5
    s = s.masked_fill(~m, -1e30)
    ot = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), kt).reshape(-1, HQ * DH)
    real = cu[len(seqs)]
    back = ot[torch.as_tensor(f2t[:real])]
    assert float((back - of[:real]).abs().max()) < 1e-12


def test_verl_tree_build_and_map_back():
    """verl adapter host logic on nested tensors: segments, rolled labels, mapping back to per-sample outputs."""
    rng = np.random.default_rng(3)
    prompt = rng.integers(1, 64, 40)
    samples = [np.concatenate([prompt, rng.integers(1, 64, n)]) for n in (5, 9, 3)]
    samples.append(rng.integers(1, 64, 17))                       # unrelated sample
    ids = torch.nested.as_nested_tensor([torch.as_tensor(s) for s in samples], layout=torch.jagged)
    tree = V.build_tree(ids, prompt_lens=[40, 40, 40, 6], align=2, cp=1, tp=1, halo=2)
    assert tree is not None and tree["groups"] == [(38, [0, 1, 2]), (0, [3])]
    lay = tree["layout"]
    assert lay.total == 38 + sum(-(-(len(s) - 38) // 2) * 2 for s in samples[:3]) + 18
    # per-segment rolled labels mapped back == per-sample rolled labels (except each sample's last row, never used)
    rolled = torch.nested.as_nested_tensor([torch.roll(t, -1) for t in tree["seg_lab"].unbind()], layout=torch.jagged)
    per = V.map_back({"lab": rolled, "other": 1}, tree)
    assert per["other"] == 1
    for i, s in enumerate(samples):
        got = per["lab"].unbind()[i]
        assert got.numel() == len(s)
        assert torch.equal(got[:-1], torch.as_tensor(s[1:]))
    seg = V.segments_of(ids, tree["groups"])
    assert [t.numel() for t in seg.unbind()] == [38, 7, 11, 5, 17]
    lm = torch.nested.as_nested_tensor([torch.ones(n) for n in (5, 9, 3, 11)], layout=torch.jagged)
    ro = V.reorder_nested(lm, [None, 0, 1, 2, 3])
    assert ro.offsets().tolist() == [0, 0, 5, 14, 17, 28]
    assert V.build_tree(torch.nested.as_nested_tensor([torch.as_tensor(samples[3])], layout=torch.jagged),
                        align=2, cp=1, tp=1, halo=2) is None


def test_negative_control_without_wrappers(modes):
    """Sanity of the check itself: the same tree pack through an UNwrapped block (attention per tree segment, branch
    positions restarting at 0, no trunk context) must not match the flat path."""
    seqs, plens, cu = make_batch(0)
    block = ToyBlock(5)                       # never passed to install_on_model
    lf, gf, lt, gt, _ = flat_and_tree(block, seqs, plens, cu, rope=True)
    assert abs(float(lt - lf)) / abs(float(lf)) > 1e-3
