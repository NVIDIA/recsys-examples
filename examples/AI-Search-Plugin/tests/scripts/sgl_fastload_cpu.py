"""CPU test for the SGLang-side fastload (record/replay) in ai_search_plugin.adapters.slime_sglang_hooks.

A toy model whose ``load_weights`` mimics SGLang's: per-expert TP-sharded narrow+copy into
fused w13/w2 params (FusedMoE style), a merged qkv param, a scalar loaded with ``fill_``
(must fall back) and a transposed-contiguous temporary (must fall back).  Replay of a new
bucket with the same layout must give exactly the parameters the original loader gives."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ai_search_plugin.adapters import slime_sglang_hooks as H  # noqa: E402

E, I, HID, TP, RANK = 8, 6, 4, 2, 1


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w13 = torch.nn.Parameter(torch.zeros(E, 2 * I // TP, HID, dtype=torch.bfloat16), requires_grad=False)
        self.w2 = torch.nn.Parameter(torch.zeros(E, HID, I // TP, dtype=torch.bfloat16), requires_grad=False)
        self.qkv = torch.nn.Parameter(torch.zeros(3 * HID, HID, dtype=torch.bfloat16), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        self.tw = torch.nn.Parameter(torch.zeros(HID, 5, dtype=torch.float32), requires_grad=False)
        self.router = torch.nn.Parameter(torch.zeros(E, HID, dtype=torch.bfloat16), requires_grad=False)
        self.norm = torch.nn.Parameter(torch.zeros(HID, dtype=torch.bfloat16), requires_grad=False)
        self.register_buffer("gemma", torch.ones(HID, dtype=torch.bfloat16), persistent=True)

    def load_weights(self, weights):
        for name, w in weights:
            if name.startswith("experts."):
                _, e, proj = name.split(".")[:3]
                e = int(e)
                if proj == "down_proj":
                    sh = w.shape[1] // TP
                    self.w2.data[e].copy_(w.narrow(1, sh * RANK, sh))
                else:
                    sh = w.shape[0] // TP
                    start = 0 if proj == "gate_proj" else I // TP
                    self.w13.data[e].narrow(0, start, I // TP).copy_(w.narrow(0, sh * RANK, sh))
            elif name in ("q", "k", "v"):
                k = "qkv".index(name)
                self.qkv.data.narrow(0, k * HID, HID).copy_(w)
            elif name == "scale":
                self.scale.data.fill_(w.item())
            elif name == "tw":
                self.tw.data.copy_(w.t().contiguous())
            elif name == "router":
                self.router.data.copy_(w)  # fp32 -> bf16 conversion inside copy_
            elif name == "norm":  # SGLang GemmaRMSNorm: copy + derived buffer written with out=
                self.norm.data.copy_(w)
                torch.add(self.norm.data, 1.0, out=self.gemma)
            elif name == "ignored":
                pass


def layout():
    out = []
    for e in range(E):  # interleaved per expert, like slime's HF conversion
        out += [(f"experts.{e}.gate_proj.weight", torch.bfloat16, (I, HID)),
                (f"experts.{e}.up_proj.weight", torch.bfloat16, (I, HID)),
                (f"experts.{e}.down_proj.weight", torch.bfloat16, (HID, I))]
    out += [("q", torch.bfloat16, (HID, HID)), ("k", torch.bfloat16, (HID, HID)), ("v", torch.bfloat16, (HID, HID)),
            ("scale", torch.float32, (1,)), ("tw", torch.float32, (5, HID)), ("router", torch.float32, (E, HID)),
            ("norm", torch.bfloat16, (HID,)), ("ignored", torch.bfloat16, (3,))]
    out.sort(key=lambda x: -x[1].itemsize)  # sender order: largest element size first (alignment)
    return out


def make_bucket(seed):
    g = torch.Generator().manual_seed(seed)
    lay = layout()
    parts = [torch.randn(s, generator=g).to(dt).view(-1).view(torch.uint8) for _, dt, s in lay]
    flat = torch.cat(parts)
    names, dts, shapes = [n for n, _, _ in lay], [d for _, d, _ in lay], [s for _, _, s in lay]
    return H._FlatBucket(flat, names, dts, shapes, (tuple(names),))


def same(a, b):
    return all(torch.equal(x.view(torch.uint8) if x.dtype != torch.float32 else x, y.view(torch.uint8) if y.dtype != torch.float32 else y)
               for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_record_replay_bit_exact():
    fast, ref = Toy(), Toy()
    b1 = make_bucket(1)
    plan = H._record(fast, b1, fast.load_weights)
    ref.load_weights(list(b1))
    assert plan is not None and same(fast, ref)
    names = [n for n, _ in b1]
    assert sorted(names[i] for i in plan.fallback) == ["norm", "scale", "tw"]
    # 3 expert projections coalesce into one strided copy each; q/k/v coalesce too
    assert len(plan.ops) <= 3 + 1 + 1, plan.ops
    for seed in (2, 3):
        b = make_bucket(seed)
        assert plan.valid(b.flat)
        H._replay(plan, b.flat)
        fast.load_weights([b[i] for i in plan.fallback])
        ref.load_weights(list(b))
        assert same(fast, ref)


def test_moved_param_invalidates():
    m = Toy()
    b = make_bucket(1)
    plan = H._record(m, b, m.load_weights)
    m.w2.data = torch.zeros_like(m.w2.data)
    assert not plan.valid(b.flat)


def test_overlapping_writes_keep_order():
    class Dup(Toy):
        def load_weights(self, weights):
            for name, w in weights:
                if name in ("q", "k"):
                    self.qkv.data.narrow(0, 0, HID).copy_(w)  # both write the same rows: last one wins

    fast, ref = Dup(), Dup()
    b1 = make_bucket(1)
    plan = H._record(fast, b1, fast.load_weights)
    b2 = make_bucket(2)
    H._replay(plan, b2.flat)
    ref.load_weights(list(b2))
    assert torch.equal(fast.qkv.view(torch.uint8), ref.qkv.view(torch.uint8))


def test_fingerprint_catches_missed_write():
    m = Toy()
    b1 = make_bucket(1)
    plan = H._record(m, b1, m.load_weights)
    names = [n for n, _ in b1]
    plan.fallback = [i for i in plan.fallback if names[i] != "norm"]  # pretend the out= write was missed
    b2 = make_bucket(2)
    H._replay(plan, b2.flat)
    m.load_weights([b2[i] for i in plan.fallback])
    fp0 = H._fingerprint(m)
    m.load_weights(list(b2))
    fp1 = H._fingerprint(m)
    assert sorted(n for n in fp1 if fp0[n] != fp1[n]) == ["gemma", "norm"]


if __name__ == "__main__":
    test_fingerprint_catches_missed_write()
    test_record_replay_bit_exact()
    test_moved_param_invalidates()
    test_overlapping_writes_keep_order()
    print("ok")


# ---------------------------------------------------------------- tensor path (verl: update_weights_from_tensor)
class FusedToy(Toy):
    """verl sends HF names with FUSED experts: experts.gate_up_proj [E, 2I, H], experts.down_proj [E, H, I], each its
    own CUDA-IPC tensor; SGLang's qwen3.5 loader slices them per expert (load_fused_expert_weights)."""

    def load_weights(self, weights):
        for name, w in weights:
            if name == "experts.gate_up_proj":
                for e in range(E):  # gate half, then up half, TP shard of each
                    for k in range(2):
                        half = w[e].narrow(0, k * I, I)
                        sh = I // TP
                        self.w13.data[e].narrow(0, k * sh, sh).copy_(half.narrow(0, sh * RANK, sh))
            elif name == "experts.down_proj":
                for e in range(E):
                    sh = I // TP
                    self.w2.data[e].copy_(w[e].narrow(1, sh * RANK, sh))
            else:  # inline, like SGLang's loaders (a loader that defers writes past the generator is not replayed)
                super().load_weights([(name, w)])


def tensor_bucket(seed, views=False):
    g = torch.Generator().manual_seed(seed)
    lay = [("experts.gate_up_proj", torch.bfloat16, (E, 2 * I, HID)), ("experts.down_proj", torch.bfloat16, (E, HID, I)),
           ("q", torch.bfloat16, (HID, HID)), ("k", torch.bfloat16, (HID, HID)), ("v", torch.bfloat16, (HID, HID)),
           ("scale", torch.float32, (1,)), ("tw", torch.float32, (5, HID)), ("router", torch.float32, (E, HID)),
           ("norm", torch.bfloat16, (HID,))]
    ts = [torch.randn(s, generator=g).to(dt) for _, dt, s in lay]
    if views:  # all inputs views of one shared buffer at different offsets (other senders may do that)
        buf = torch.cat([t.view(-1).view(torch.uint8) for t in ts])
        out, off = [], 0
        for t in ts:
            nb = t.numel() * t.element_size()
            out.append(buf[off:off + nb].view(t.dtype).view(t.shape))
            off += nb
        ts = out
    return [(n, t) for (n, _, _), t in zip(lay, ts)]


def test_tensor_path_record_replay_bit_exact():
    for views in (False, True):
        fast, ref = FusedToy(), FusedToy()
        b1 = H._TBucket(tensor_bucket(1, views))
        plan = H._record_tensors(fast, b1, fast.load_weights)
        ref.load_weights(list(b1))
        assert plan is not None and same(fast, ref)
        names = [n for n, _ in b1]
        assert sorted(names[i] for i in plan.fallback) == ["norm", "scale", "tw"]
        # 2*E gate/up copies -> a few strided ops; E down copies -> 1
        ops = {names[i]: len(o) for i, o in plan.by_src}
        assert ops["experts.down_proj"] == 1 and ops["experts.gate_up_proj"] <= 2, ops
        for seed in (2, 3):
            b = H._TBucket(tensor_bucket(seed, views))
            assert b.key == b1.key and plan.valid(b)
            H._replay_tensors(plan, b)
            fast.load_weights([b[i] for i in plan.fallback])
            ref.load_weights(list(b))
            assert same(fast, ref)
