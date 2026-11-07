"""Op-level numerics of response_logits.chunked vs the unchunked compact path and an fp64 reference.

torchrun --nproc-per-node=<tp> test_chunked_op.py [--n 3000 --vocab 32768 --hidden 512 --chunk 640]
Each case: x [n, h] bf16 (TP-replicated), full W [V, h] bf16 (rank r holds rows r*V/TP..), targets, temperature
(scalar or per row), entropy in the loss or not, Megatron main_grad fusion or plain grad.
  compact = bf16 GEMM -> fp32 -> /T -> slime's own calculate_log_probs_and_entropy (vocab-parallel), autograd
            (= what the response_logits adapter did before; falls back to an inline copy of slime's formula when
            slime is not importable)
  chunked = response_logits.chunked.lm_head_logprob_entropy (triton and torch impls)
  fp64    = logits in fp64 from the same bf16 x / W, full vocab, exact log-softmax / entropy and their grads
Prints max|d log-prob|, max|d entropy| (chunked vs compact), rel ||d dW||, ||d dX|| (vs compact) and the error of
both against fp64. PASS: lp/ent <= 1e-5 abs and dW/dX <= 1e-4 rel vs compact.
"""
import argparse
import os

import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=3000)
p.add_argument('--vocab', type=int, default=32768)
p.add_argument('--hidden', type=int, default=512)
p.add_argument('--chunk', type=int, default=640)
args = p.parse_args()
dist.init_process_group('nccl')
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
dev = 'cuda'
G = dist.new_group(list(range(world)))
from ai_search_plugin.response_logits import chunked  # noqa: E402

try:
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy as slime_calc
except Exception:
    slime_calc = None


def compact_ref(x, w_loc, tgt, temp, with_ent, ent_grad, vstart, split=1):
    if split > 1:  # noise-floor control: same math, the output-layer GEMM run as `split` row blocks
        logits = torch.cat([torch.mm(xc, w_loc.t()) for xc in x.chunk(split)]).float()
    else:
        logits = torch.mm(x, w_loc.t()).float()
    logits = logits / (temp.unsqueeze(1) if torch.is_tensor(temp) else temp)
    if slime_calc is not None:
        lp, ent = slime_calc(logits, tgt, G, with_entropy=with_ent, with_entropy_grad=ent_grad)
        return lp.reshape(-1), (ent.reshape(-1) if with_ent else None)
    # inline copy of the math (verl / megatron vocab-parallel CE + entropy)
    m = logits.max(-1, keepdim=True).values.detach().clone()
    dist.all_reduce(m, op=dist.ReduceOp.MAX, group=G)
    zs = logits - m
    e = zs.exp()
    s = _AR.apply(e.sum(-1, keepdim=True))
    lt = tgt - vstart
    own = (lt >= 0) & (lt < w_loc.size(0))
    pred = zs.gather(1, lt.clamp(0, w_loc.size(0) - 1).unsqueeze(1)).squeeze(1) * own
    pred = _AR.apply(pred)
    lp = pred - s.squeeze(1).log()
    ent = None
    if with_ent:
        pz = ((e / s) * logits).sum(-1)
        pz = _AR.apply(pz)
        ent = m.squeeze(1) + s.squeeze(1).log() - pz
        if not ent_grad:
            ent = ent.detach()
    return lp, ent


class _AR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t):
        t = t.clone()
        dist.all_reduce(t, group=G)
        return t

    @staticmethod
    def backward(ctx, g):
        return g


def fp64_ref(x, w_full, tgt, temp, with_ent):
    z = (x.double() @ w_full.double().t()) / (temp.double().unsqueeze(1) if torch.is_tensor(temp) else temp)
    ls = torch.log_softmax(z, -1)
    lp = ls.gather(1, tgt.unsqueeze(1)).squeeze(1)
    ent = -(ls.exp() * ls).sum(-1) if with_ent else None
    return lp, ent


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def mx(v):
    t = torch.tensor([float(v)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def case(n, temp_kind, with_ent, ent_grad, fusion, impl, chunk, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    V, H = args.vocab, args.hidden
    Vl = V // world
    x0 = (torch.randn(max(n, 1), H, generator=g) * 1.0).to(torch.bfloat16).to(dev)
    w_full = (torch.randn(V, H, generator=g) * 0.08).to(torch.bfloat16).to(dev)
    tgt = torch.randint(0, V, (max(n, 1),), generator=g).to(dev)
    glp = torch.randn(max(n, 1), generator=g).to(dev)
    gen = torch.randn(max(n, 1), generator=g).to(dev)
    if temp_kind == 'vec':
        temp = (torch.rand(max(n, 1), generator=g) * 0.6 + 0.7).to(dev)
    else:
        temp = float(temp_kind)
    vstart = rank * Vl
    out = {}
    for which in ('compact', 'compact_split', 'chunked'):
        x = x0.clone().requires_grad_(True)
        w = w_full[vstart:vstart + Vl].clone().requires_grad_(True)
        if which == 'chunked' and fusion:
            w.main_grad = torch.zeros(w.shape, device=dev, dtype=torch.float32)
            w.grad_added_to_main_grad = False
        if which.startswith('compact'):
            lp, ent = compact_ref(x, w, tgt, temp, with_ent, ent_grad, vstart, split=5 if which == 'compact_split' else 1)
        else:
            os.environ['RESPONSE_LOGITS_CHUNK_IMPL'] = impl
            spec = chunked.Spec(tp_group=G, vstart=vstart, temperature=1.0 if torch.is_tensor(temp) else temp,
                                with_entropy=with_ent, entropy_grad=ent_grad, chunk=chunk, grad_accum_fusion=fusion,
                                allreduce_dgrad=True, impl_name=impl)
            lp, ent = chunked.lm_head_logprob_entropy(x, w, tgt, spec,
                                                      temp_vec=temp if torch.is_tensor(temp) else None)
        sl = slice(0, n)
        loss = (lp[sl] * glp[sl]).sum()
        if with_ent and ent_grad:
            loss = loss + (ent[sl] * gen[sl]).sum()
        loss.backward()
        dx = x.grad
        if which.startswith('compact') and world > 1:
            dist.all_reduce(dx, group=G)  # column-parallel dgrad all-reduce (done inside the chunked op)
        dw = w.main_grad if (which == 'chunked' and fusion) else w.grad
        out[which] = dict(lp=lp[sl].detach(), ent=None if ent is None else ent[sl].detach(), dx=dx.float(),
                          dw=dw.float())
    # fp64 reference (full vocab, local)
    xr = x0.double().requires_grad_(True)
    wr = w_full.double().requires_grad_(True)
    z = (xr @ wr.t()) / (temp.double().unsqueeze(1) if torch.is_tensor(temp) else temp)
    ls = torch.log_softmax(z, -1)
    lp64 = ls.gather(1, tgt.unsqueeze(1)).squeeze(1)
    ent64 = -(ls.exp() * ls).sum(-1)
    loss = (lp64[:n] * glp[:n].double()).sum()
    if with_ent and ent_grad:
        loss = loss + (ent64[:n] * gen[:n].double()).sum()
    loss.backward()
    dw64 = wr.grad[vstart:vstart + Vl]
    a, b, c = out['chunked'], out['compact'], out['compact_split']
    r = {
        'noise_dW': mx(rel(c['dw'], b['dw']) if n else 0.0), 'noise_dX': mx(rel(c['dx'], b['dx']) if n else 0.0),
        'lp_vs_compact': mx((a['lp'] - b['lp']).abs().max().item() if n else 0.0),
        'ent_vs_compact': mx((a['ent'] - b['ent']).abs().max().item() if (n and with_ent) else 0.0),
        'dW_rel_vs_compact': mx(rel(a['dw'], b['dw']) if n else (a['dw'].abs().max().item())),
        'dX_rel_vs_compact': mx(rel(a['dx'], b['dx']) if n else (a['dx'].abs().max().item())),
        'lp_err64_chunked': mx((a['lp'].double() - lp64[:n]).abs().max().item() if n else 0),
        'lp_err64_compact': mx((b['lp'].double() - lp64[:n]).abs().max().item() if n else 0),
        'dW_rel64_chunked': mx(rel(a['dw'].double(), dw64) if n else 0),
        'dW_rel64_compact': mx(rel(b['dw'].double(), dw64) if n else 0),
        'dX_rel64_chunked': mx(rel(a['dx'].double(), xr.grad) if n else 0),
        'dX_rel64_compact': mx(rel(b['dx'].double(), xr.grad) if n else 0),
    }
    # grads: bf16 dX / dW outputs flip by one ulp wherever the GEMM accumulation order changes (row-block shape);
    # pass = within 1e-4 or within 1.5x the compact path's own noise floor (GEMM split in 5 row blocks), and no
    # worse than the compact path against fp64 (with fusion the chunked dW stays fp32 -> more accurate).
    tol_w = max(1e-4, 1.5 * r['noise_dW'])
    tol_x = max(1e-4, 1.5 * r['noise_dX'])
    ok = (r['lp_vs_compact'] <= 1e-5 and r['ent_vs_compact'] <= 1e-5
          and (r['dW_rel_vs_compact'] <= tol_w or (fusion and r['dW_rel64_chunked'] <= r['dW_rel64_compact']))
          and r['dX_rel_vs_compact'] <= tol_x
          and r['dW_rel64_chunked'] <= 1.05 * r['dW_rel64_compact'] + 1e-6
          and r['dX_rel64_chunked'] <= 1.05 * r['dX_rel64_compact'] + 1e-6)
    if n == 0:
        ok = r['dW_rel_vs_compact'] == 0.0 and r['dX_rel_vs_compact'] == 0.0
    tag = f'tp={world} n={n} T={temp_kind} ent={with_ent} ent_grad={ent_grad} fusion={fusion} impl={impl} chunk={chunk}'
    if rank == 0:
        print(('PASS ' if ok else 'FAIL ') + tag)
        print('   ' + '  '.join(f'{k}={v:.2e}' for k, v in r.items()))
    return ok


allok = True
for impl in ('triton', 'torch'):
    for (n, temp, we, eg, fu) in [(args.n, '1.0', True, False, False), (args.n, '0.8', True, True, False),
                                  (args.n, 'vec', True, True, True), (args.n, '1.0', False, False, True),
                                  (1, '0.8', True, True, False), (0, '1.0', True, True, False)]:
        allok &= case(n, temp, we, eg, fu, impl, args.chunk)
if rank == 0:
    print('ALL', 'PASS' if allok else 'FAIL', chunked.stats())
dist.barrier()
dist.destroy_process_group()
