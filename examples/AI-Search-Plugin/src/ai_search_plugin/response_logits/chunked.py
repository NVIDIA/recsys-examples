"""Chunked output layer + vocab-parallel log-prob / entropy (chunked CE with recompute).

The stock path (and the plain response_logits adapters) materialise logits [n, V/TP] in fp32, keep them for
backward and allocate a second full-size fp32 gradient; with n = 99k response rows and a ~124k-entry vocab shard (V/TP) that is
45.8 GiB per buffer. This op never holds more than one [chunk, V/TP] bf16 block:

  forward : for each row block  z = (x_c @ W^T) (bf16, same GEMM as the output layer) -> fp32 / temperature
            -> online max / sum-exp / sum(e*z) / target logit (Triton, one read of the bf16 block), block freed.
            Per-row stats are combined over TP once for all rows (all-reduce MAX, then one all-reduce SUM of
            [sum-exp, sum(e*z), target logit]); saved for backward: x, W (references), targets, lse, E[z].
  backward: for each row block recompute z, dz = g_lp (onehot - p) + g_ent p (E[z] - z), / temperature, cast to
            bf16 in place, then dX_c = dz @ W (written into dX) and dW += dz^T @ x_c (fp32 accumulate: straight
            into weight.main_grad with Megatron's wgrad_gemm_accum_fp32 when gradient_accumulation_fusion is on,
            exactly like ColumnParallelLinear; otherwise an fp32 buffer cast to the weight dtype once at the end).
            dX is all-reduced over TP once at the end (= the column-parallel layer's dgrad all-reduce).

Values: log_prob = (z_t - max) - log(sum exp(z - max)), entropy = max + log(sum) - E_p[z] -- the formulas of
slime's _VocabParallelLogProbEntropy, Megatron's vocab_parallel_cross_entropy and verl's vocab_parallel_entropy;
only the fp32 reduction order differs (online rescaling), i.e. ~1e-7 relative.

Config (environment, read per call):
  RESPONSE_LOGITS_CHUNK      auto (default): chunk only when the compact fp32 logits [n, V/TP] would exceed the
                             budget (small micro-batches keep the unchunked path, bit-identical to before);
                             1/on: always chunk; 0/off: never; <int > 1>: always chunk with that many rows.
  RESPONSE_LOGITS_CHUNK_MB   per-block budget in MiB (default 2048): rows = budget / (V/TP * bytes per element of
                             the working set), rounded down to a multiple of 128 (min 128).
  RESPONSE_LOGITS_CHUNK_IMPL triton (default when Triton imports) | torch (reference, ~4x the block memory).
"""
import os

import torch
import torch.distributed as dist

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False

_STATS = {'calls': 0, 'rows': 0, 'chunks': 0, 'bwd_calls': 0, 'fused_wgrad': 0, 'wgrad_impl': 'torch'}


def stats():
    return dict(_STATS)


# --------------------------------------------------------------------------------------------- config
def mode():
    v = os.environ.get('RESPONSE_LOGITS_CHUNK', 'auto').strip().lower()
    if v in ('', 'auto'):
        return 'auto', 0
    if v in ('0', 'off', 'false', 'no'):
        return 'off', 0
    if v in ('1', 'on', 'true', 'yes'):
        return 'on', 0
    n = int(v)
    if n < 0:
        raise ValueError(f'RESPONSE_LOGITS_CHUNK must be auto|on|off|<rows>, got {v!r}')
    return 'on', n


def impl():
    v = os.environ.get('RESPONSE_LOGITS_CHUNK_IMPL', 'triton' if _HAVE_TRITON else 'torch').strip().lower()
    if v == 'triton' and not _HAVE_TRITON:
        v = 'torch'
    return v


def budget_bytes():
    return int(float(os.environ.get('RESPONSE_LOGITS_CHUNK_MB', '2048')) * 2**20)


def auto_rows(vocab_local, impl_name=None):
    """Rows per block so that the block working set stays within the budget."""
    per = 2 if (impl_name or impl()) == 'triton' else 14  # bf16 block | bf16 + 3 fp32 temporaries
    rows = budget_bytes() // max(1, vocab_local * per)
    return max(128, rows // 128 * 128)


def decide(n_rows, vocab_local):
    """Chunk rows to use for a micro-batch with n_rows compact rows (0 = keep the unchunked path).
    Depends only on (n_rows, vocab_local, env), so every TP rank decides the same."""
    m, forced = mode()
    if m == 'off':
        return 0
    rows = forced or auto_rows(vocab_local)
    if m == 'auto' and n_rows * vocab_local * 4 <= budget_bytes():
        return 0
    return rows


# --------------------------------------------------------------------------------------------- kernels
if _HAVE_TRITON:
    @triton.jit
    def _fwd_stats_kernel(Z, stride_z, TGT, TEMP, M_OUT, S_OUT, T_OUT, ZT_OUT, V, vstart,
                          HAS_TEMP_VEC: tl.constexpr, temp_scalar, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        if HAS_TEMP_VEC:
            temp = tl.load(TEMP + row)
        else:
            temp = temp_scalar
        base = Z + row * stride_z
        cols0 = tl.arange(0, BLOCK)
        m = tl.full([], float('-inf'), tl.float32)
        s = tl.zeros([], tl.float32)
        t = tl.zeros([], tl.float32)
        for off in range(0, V, BLOCK):
            cols = off + cols0
            msk = cols < V
            z = tl.load(base + cols, mask=msk, other=0.0).to(tl.float32)
            z = tl.math.div_rn(z, temp)
            z = tl.where(msk, z, float('-inf'))
            m_new = tl.maximum(m, tl.max(z, 0))
            alpha = tl.exp(m - m_new)
            e = tl.exp(z - m_new)
            s = s * alpha + tl.sum(e, 0)
            t = t * alpha + tl.sum(tl.where(msk, e * z, 0.0), 0)
            m = m_new
        tgt = tl.load(TGT + row) - vstart
        own = (tgt >= 0) & (tgt < V)
        zt = tl.load(base + tl.where(own, tgt, 0)).to(tl.float32)
        zt = tl.where(own, tl.math.div_rn(zt, temp), 0.0)
        tl.store(M_OUT + row, m)
        tl.store(S_OUT + row, s)
        tl.store(T_OUT + row, t)
        tl.store(ZT_OUT + row, zt)

    @triton.jit
    def _bwd_kernel(Z, stride_z, TGT, TEMP, LSE, EZ, GLP, GENT, V, vstart,
                    HAS_TEMP_VEC: tl.constexpr, temp_scalar, HAS_ENT: tl.constexpr, BLOCK: tl.constexpr):
        """In place: Z (bf16 logits block) <- d loss / d logits (bf16)."""
        row = tl.program_id(0).to(tl.int64)
        if HAS_TEMP_VEC:
            temp = tl.load(TEMP + row)
        else:
            temp = temp_scalar
        lse = tl.load(LSE + row)
        glp = tl.load(GLP + row)
        if HAS_ENT:
            gent = tl.load(GENT + row)
            ez = tl.load(EZ + row)
        tgt = tl.load(TGT + row) - vstart
        base = Z + row * stride_z
        cols0 = tl.arange(0, BLOCK)
        for off in range(0, V, BLOCK):
            cols = off + cols0
            msk = cols < V
            z = tl.load(base + cols, mask=msk, other=0.0).to(tl.float32)
            z = tl.math.div_rn(z, temp)
            p = tl.exp(z - lse)
            g = tl.where(cols == tgt, 1.0, 0.0) - p
            g = g * glp
            if HAS_ENT:
                g = g + (p * (ez - z)) * gent
            g = tl.math.div_rn(g, temp)
            tl.store(base + cols, g.to(Z.dtype.element_ty), mask=msk)


def _block():
    return int(os.environ.get('RESPONSE_LOGITS_CHUNK_BLOCK', '4096'))


def _stats_triton(z, tgt, temp_vec, temp_scalar, vstart):
    c, V = z.shape
    out = torch.empty(4, c, device=z.device, dtype=torch.float32)
    if c:
        _fwd_stats_kernel[(c,)](z, z.stride(0), tgt, temp_vec if temp_vec is not None else out, out[0], out[1],
                                out[2], out[3], V, vstart, HAS_TEMP_VEC=temp_vec is not None,
                                temp_scalar=float(temp_scalar), BLOCK=_block(), num_warps=8)
    return out


def _stats_torch(z, tgt, temp_vec, temp_scalar, vstart):
    c, V = z.shape
    zf = z.float()
    if temp_vec is not None:
        zf.div_(temp_vec.unsqueeze(1))
    elif temp_scalar != 1.0:
        zf.div_(temp_scalar)
    m = zf.max(dim=1).values
    e = (zf - m.unsqueeze(1)).exp_()
    s = e.sum(dim=1)
    t = torch.einsum('ij,ij->i', e, zf)
    del e
    lt = tgt - vstart
    own = (lt >= 0) & (lt < V)
    zt = zf.gather(1, lt.clamp(0, V - 1).unsqueeze(1)).squeeze(1).masked_fill(~own, 0.0)
    return torch.stack([m, s, t, zt])


def _bwd_torch(z, tgt, temp_vec, temp_scalar, vstart, lse, ez, glp, gent):
    c, V = z.shape
    zf = z.float()
    tcol = temp_vec.unsqueeze(1) if temp_vec is not None else temp_scalar
    if temp_vec is not None or temp_scalar != 1.0:
        zf.div_(tcol)
    p = (zf - lse.unsqueeze(1)).exp_()
    if gent is not None:
        g = (ez.unsqueeze(1) - zf).mul_(p).mul_(gent.unsqueeze(1))
    else:
        g = None
    del zf
    p.neg_()
    lt = tgt - vstart
    own = ((lt >= 0) & (lt < V)).nonzero().squeeze(1)
    p[own, lt[own]] += 1.0
    p.mul_(glp.unsqueeze(1))
    if g is not None:
        p.add_(g)
        del g
    if temp_vec is not None or temp_scalar != 1.0:
        p.div_(tcol)
    z.copy_(p)


def _bwd_triton(z, tgt, temp_vec, temp_scalar, vstart, lse, ez, glp, gent):
    c, V = z.shape
    if c:
        _bwd_kernel[(c,)](z, z.stride(0), tgt, temp_vec if temp_vec is not None else lse, lse,
                          ez if ez is not None else lse, glp, gent if gent is not None else lse, V, vstart,
                          HAS_TEMP_VEC=temp_vec is not None, temp_scalar=float(temp_scalar),
                          HAS_ENT=gent is not None, BLOCK=_block(), num_warps=8)


# --------------------------------------------------------------------------------------------- wgrad
def _wgrad_accum_fn():
    """f(x [c, h], dy [c, V], out [V, h] fp32): out += dy^T @ x with fp32 accumulation and no temporary.
    apex fused_weight_gradient_mlp_cuda (what Megatron's ColumnParallelLinear uses) if present, else Transformer
    Engine's general_gemm(accumulate=True) (what TE linear layers use for main_grad), else None."""
    if 'wgrad' in _CACHE:
        return _CACHE['wgrad']
    fn = None
    if os.environ.get('RESPONSE_LOGITS_CHUNK_WGRAD', 'auto') in ('auto', 'apex'):
        try:
            import fused_weight_gradient_mlp_cuda as f
            fn = f.wgrad_gemm_accum_fp32
            _STATS['wgrad_impl'] = 'apex'
        except Exception:
            fn = None
    if fn is None and os.environ.get('RESPONSE_LOGITS_CHUNK_WGRAD', 'auto') in ('auto', 'te'):
        try:
            import inspect
            from transformer_engine.pytorch.cpp_extensions import general_gemm
            params = inspect.signature(general_gemm).parameters
            ws = None
            if 'workspace' in params:
                from transformer_engine.pytorch.module.base import get_workspace
                ws = get_workspace

            def fn(x, dy, out, _g=general_gemm, _ws=ws):
                if _ws is not None:
                    _g(x, dy, _ws(), out_dtype=torch.float32, layout='NT', grad=True, out=out, accumulate=True)
                else:
                    _g(x, dy, out_dtype=torch.float32, layout='NT', grad=True, out=out, accumulate=True)
            _STATS['wgrad_impl'] = 'te'
        except Exception:
            fn = None
    _CACHE['wgrad'] = fn
    return fn


_CACHE = {}


def _mm_fp32(a, b):
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a.float(), b.float())


def _accum_wgrad(out, x, dy):
    """out (fp32 [V, h]) += dy^T @ x."""
    fn = _wgrad_accum_fn() if out.dtype == torch.float32 else None
    if fn is not None:
        fn(x, dy, out)
    else:
        out.add_(_mm_fp32(dy.t(), x))


class Spec:
    """Non-tensor arguments of the op."""
    __slots__ = ('tp_group', 'vstart', 'temperature', 'with_entropy', 'entropy_grad', 'chunk', 'grad_accum_fusion',
                 'allreduce_dgrad', 'impl')

    def __init__(self, *, tp_group=None, vstart=0, temperature=1.0, with_entropy=True, entropy_grad=True,
                 chunk=4096, grad_accum_fusion=False, allreduce_dgrad=True, impl_name=None):
        self.tp_group = tp_group
        self.vstart = int(vstart)
        self.temperature = float(temperature)
        self.with_entropy = bool(with_entropy)
        self.entropy_grad = bool(entropy_grad) and bool(with_entropy)
        self.chunk = int(chunk)
        self.grad_accum_fusion = bool(grad_accum_fusion)
        self.allreduce_dgrad = bool(allreduce_dgrad)
        self.impl = impl_name or impl()


_WS = {}


def _block_buf(c, V, like):
    """[c, V] scratch for one logits block. With RESPONSE_LOGITS_CHUNK_WS=1 (default off: no reserved-memory gain was measured, +2 GiB allocated) one persistent buffer per device is
    reused by every call (forward and backward never overlap on a stream), so the caching allocator does not have to find
    a fresh multi-GiB hole for every block (repeated large allocations fragment the pool and grow the reserved memory)."""
    if os.environ.get('RESPONSE_LOGITS_CHUNK_WS', '0') != '1':
        return torch.empty(c, V, device=like.device, dtype=like.dtype)
    key = (like.device, like.dtype)
    buf = _WS.get(key)
    if buf is None or buf.numel() < c * V:
        _WS.pop(key, None)
        del buf
        buf = torch.empty(c * V, device=like.device, dtype=like.dtype)
        _WS[key] = buf
    return buf[:c * V].view(c, V)


def release_workspace():
    _WS.clear()


def _tp_size(g):
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(group=g)


class _ChunkedLMHeadLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, targets, temp_vec, spec):
        n, h = x.shape
        V = weight.size(0)
        stats_fn = _stats_triton if spec.impl == 'triton' else _stats_torch
        st = torch.empty(4, n, device=x.device, dtype=torch.float32)
        w_t = weight.t()
        cmax = min(n, spec.chunk)
        for a in range(0, n, spec.chunk):
            b = min(n, a + spec.chunk)
            z = torch.mm(x[a:b], w_t, out=_block_buf(cmax, V, x)[:b - a])  # [c, V/TP] (= the output layer's GEMM)
            st[:, a:b] = stats_fn(z, targets[a:b], None if temp_vec is None else temp_vec[a:b], spec.temperature,
                                  spec.vstart)
            del z
            _STATS['chunks'] += 1
        m_loc, s_loc, t_loc, zt = st[0], st[1], st[2], st[3]
        tp = _tp_size(spec.tp_group)
        if tp > 1:
            m = m_loc.clone()
            dist.all_reduce(m, op=dist.ReduceOp.MAX, group=spec.tp_group)
            scale = torch.exp(m_loc - m)
            red = torch.stack([s_loc * scale, t_loc * scale, zt])
            dist.all_reduce(red, op=dist.ReduceOp.SUM, group=spec.tp_group)
            s, t, zt = red[0], red[1], red[2]
        else:
            m, s, t = m_loc, s_loc, t_loc
        log_s = s.log()
        log_prob = (zt - m) - log_s
        ez = t / s
        lse = m + log_s
        entropy = (lse - ez) if spec.with_entropy else log_prob.new_zeros((0,))
        ctx.spec = spec
        ctx.save_for_backward(x, weight, targets, temp_vec, lse, ez)
        if not spec.entropy_grad:
            ctx.mark_non_differentiable(entropy)
        ctx.set_materialize_grads(False)
        _STATS['calls'] += 1
        _STATS['rows'] += n
        return log_prob, entropy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_lp, g_ent):
        x, weight, targets, temp_vec, lse, ez = ctx.saved_tensors
        spec = ctx.spec
        n, h = x.shape
        need_x, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if g_lp is None and g_ent is None:
            return None, None, None, None, None
        g_lp = torch.zeros_like(lse) if g_lp is None else g_lp.float().reshape(-1).contiguous()
        if not spec.entropy_grad or g_ent is None:
            g_ent = None
        else:
            g_ent = g_ent.float().reshape(-1).contiguous()
        bwd = _bwd_triton if spec.impl == 'triton' else _bwd_torch
        dx = torch.empty_like(x) if need_x else None
        fused = False
        dw32 = None
        if need_w:
            if spec.grad_accum_fusion and hasattr(weight, 'main_grad') and weight.main_grad is not None:
                fused = True  # accumulate straight into Megatron's main_grad (as ColumnParallelLinear does)
            else:
                dw32 = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
        w_t = weight.t()
        V = weight.size(0)
        cmax = min(n, spec.chunk)
        for a in range(0, n, spec.chunk):
            b = min(n, a + spec.chunk)
            z = torch.mm(x[a:b], w_t, out=_block_buf(cmax, V, x)[:b - a])
            bwd(z, targets[a:b], None if temp_vec is None else temp_vec[a:b], spec.temperature, spec.vstart,
                lse[a:b], ez[a:b], g_lp[a:b], None if g_ent is None else g_ent[a:b])
            if need_x:
                torch.mm(z, weight, out=dx[a:b])
            if need_w:
                _accum_wgrad(weight.main_grad if fused else dw32, x[a:b], z)
            del z
        if need_x and spec.allreduce_dgrad and _tp_size(spec.tp_group) > 1:
            dist.all_reduce(dx, group=spec.tp_group)
        dw = None
        if need_w:
            if fused:
                _STATS['fused_wgrad'] += 1
                if hasattr(weight, 'grad_added_to_main_grad'):  # Megatron convention (DDP skips the add)
                    if getattr(weight, 'zero_out_wgrad', False):
                        dw = torch.zeros(weight.main_grad.shape, dtype=x.dtype, device=x.device)
                    else:
                        dw = torch.empty(weight.main_grad.shape, dtype=x.dtype, device=x.device)
                    weight.grad_added_to_main_grad = True
            else:
                dw = dw32.to(weight.dtype)
        _STATS['bwd_calls'] += 1
        return dx, dw, None, None, None


def lm_head_logprob_entropy(x, weight, targets, spec, temp_vec=None):
    """x [n, h] (rows already TP-replicated), weight [V/TP, h] (vocab shard starting at spec.vstart), targets [n]
    global token ids, temp_vec optional fp32 [n] per-row temperature (else spec.temperature).
    Returns (log_prob [n] fp32, entropy [n] fp32 or [0] when spec.with_entropy is False)."""
    x = x.contiguous()
    targets = targets.reshape(-1).long().contiguous()
    if temp_vec is not None:
        temp_vec = temp_vec.reshape(-1).float().contiguous()
    return _ChunkedLMHeadLogProb.apply(x, weight, targets, temp_vec, spec)
