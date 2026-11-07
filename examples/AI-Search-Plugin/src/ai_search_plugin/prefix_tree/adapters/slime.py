# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-tree training inside slime's Megatron trainer.

Switch: ASRT_PREFIX_TREE=1, plus this module's ``megatron_init`` in slime's Megatron-init chain
(``ASRT_CHAIN_MEGATRON_INIT+=ai_search_plugin.prefix_tree.adapters.slime.megatron_init``, or
``--custom-megatron-init-path`` directly).

Per micro-batch (log-prob pass and training step alike), with NO change to slime's data, schedule, loss or the model
input / output layout:
  * slime's ``get_batch`` is wrapped: the micro-batch's per-sample tokens and prompt lengths are remembered on its
    PackedSeqParams;
  * Megatron's ``TransformerBlock.forward`` is wrapped: the embedded hidden states of the flat (slime) pack are
    gathered into a tree pack (per sibling group: the shared prefix once = trunk, then one branch per sample; a group =
    samples of the micro-batch with identical prompt tokens, i.e. GRPO siblings), the layers run on the tree pack, and
    the block output (after the final norm) is scattered back to the flat row order.  Output layer, log-probs,
    entropy, loss and every hook after the decoder see exactly the flat layout they saw before;
  * inside the layers, token-wise work runs on the tree rows only; the sequence mixers run on the virtual 'full' pack
    (``model.py``: one document per sample = trunk + its branch, the same documents as the flat pack, so the attention
    kernel and RoPE run unchanged).
Mathematically identical to the flat path: a row's hidden state depends only on its own prefix, so the n siblings'
copies of a shared-prefix row are equal; the trunk computes it once and its gradient is the sum over the siblings.
The trunk may extend into the response when siblings share their first response tokens (rows are mapped back per
flat position, so loss masks are not involved); every branch keeps >= 1 token.

Scope: context parallel size 1, pipeline parallel size 1 (pre- and post-process on the same stage), dense or MoE
transformer layers with softmax attention / GQA (TP and sequence parallel supported).  CP > 1, PP > 1, an unknown
sequence mixer, a micro-batch without siblings, or a virtual document longer than the RoPE table -> the flat path is
used (counted in ``snapshot()['fallback_reasons']``).  Savings: the shared rows skip embedding-free token-wise work
(QKV / output projections, MLP, norms); attention still runs per sample.

Check mode (numerics): PREFIX_TREE_SLIME_CHECK=K -> on the first K micro-batches per rank the decoder also runs the
flat pack and a re-cut flat pack (same samples, reversed order: the noise floor of packing), and per-sample
response-row log-prob differences (tree / re-cut vs flat) are written to $ASRT_LOG_DIR/prefix_tree_check-rank<r>.jsonl.
"""
import json
import os
import time

import numpy as np
import torch

from .. import model as ptm
from ..plan import TreeLayout, group_by_prompt

_STATE = {'installed': False, 'get_batch_calls': 0, 'block_calls': 0, 'tree_calls': 0, 'flat_fallback': 0,
          'fallback_reasons': {}, 'flat_tokens': 0, 'tree_tokens': 0, 'groups': 0, 'grouped_samples': 0,
          'singletons': 0, 'samples': 0, 'build_s': 0.0, 'check_calls': 0}
_CTX = {'gpt': None}
MIN_TRUNK = int(os.environ.get('PREFIX_TREE_MIN_TRUNK', '16'))
CHECK = int(os.environ.get('PREFIX_TREE_SLIME_CHECK', '0'))
LOG_EVERY = int(os.environ.get('PREFIX_TREE_LOG_EVERY', '0'))


def enabled():
    return os.environ.get('ASRT_PREFIX_TREE', '0') not in ('0', '', 'false', 'False')


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0


def _log(msg, all_ranks=False):
    if all_ranks or _rank() == 0:
        print(f'[asrt prefix_tree.slime] {msg}', flush=True)


def _fallback(why):
    _STATE['flat_fallback'] += 1
    r = _STATE['fallback_reasons']
    r[why] = r.get(why, 0) + 1


# ---- host-side plan ---------------------------------------------------------------------------------------------

def plan_groups(seqs, plens, min_trunk=None):
    """[(T, [sample idx])] for one slime micro-batch (see plan.group_by_prompt)."""
    return group_by_prompt(seqs, plens, MIN_TRUNK if min_trunk is None else min_trunk)


def build_maps(seqs, plens, cu_flat, tp, pad_multiple, min_trunk=None, halo=1):
    """Host-side tree plan of one flat slime pack: (groups, TreeLayout, t2f, f2t, tree_pos, sib, gid, t_zero, f_zero).
    t2f[tree row] = flat row it reads (trunk: sibling 0's rows); f2t[flat row] = tree row whose output it takes."""
    n = len(seqs)
    lens = [len(s) for s in seqs]
    assert len(cu_flat) - 1 in (n, n + 1), (len(cu_flat), n)
    assert all(cu_flat[i + 1] - cu_flat[i] == lens[i] for i in range(n)), 'slime pack != per-sample tokens'
    flat_total = int(cu_flat[-1])
    groups = plan_groups(seqs, plens, min_trunk)
    spec = [(T, [lens[i] - T for i in idx]) for T, idx in groups]
    lay = TreeLayout(spec, cp=1, tp=tp, pad_multiple=pad_multiple, halo=halo, align=1)
    tree_total = lay.total
    t2f = np.zeros(tree_total, dtype=np.int64)
    f2t = np.zeros(flat_total, dtype=np.int64)
    tree_pos = np.zeros(tree_total, dtype=np.int64)
    sib = np.full(n, -1, dtype=np.int64)      # sibling rank inside its tree group (-1: singleton)
    gid = np.full(n, -1, dtype=np.int64)
    for g, ((T, idx), ginfo) in enumerate(zip(groups, lay.groups)):
        if T:
            ts = ginfo['trunk']
            t2f[ts:ts + T] = cu_flat[idx[0]] + np.arange(T)
            tree_pos[ts:ts + T] = np.arange(T)
        for m, (i, (bs, b)) in enumerate(zip(idx, ginfo['branches'])):
            t2f[bs:bs + b] = cu_flat[i] + T + np.arange(b)
            tree_pos[bs:bs + b] = np.arange(T, T + b)
            a = cu_flat[i]
            if T:
                f2t[a:a + T] = ginfo['trunk'] + np.arange(T)
                sib[i], gid[i] = m, g
            f2t[a + T:a + lens[i]] = bs + np.arange(b)
    real_t, pad_t = lay.trailing
    real_f = int(cu_flat[n])
    pad_f = flat_total - real_f
    # pad rows read pad rows of the other pack; when that pack has none, they read row 0 but are ZEROED (t_zero /
    # f_zero masks applied after the index_select), so no value and no gradient crosses into a real row
    t_zero = np.zeros(tree_total, dtype=bool)
    f_zero = np.zeros(flat_total, dtype=bool)
    if pad_t:
        t2f[real_t:] = (real_f + np.arange(pad_t) % pad_f) if pad_f else 0
        t_zero[real_t:] = not pad_f
        tree_pos[real_t:] = np.arange(pad_t)
    if pad_f:
        f2t[real_f:] = (real_t + np.arange(pad_f) % pad_t) if pad_t else 0
        f_zero[real_f:] = not pad_t
    return groups, lay, t2f, f2t, tree_pos, sib, gid, t_zero, f_zero


class SlimeTree:
    """Tree pack of one slime micro-batch (CP 1) + flat<->tree row maps + the RankPlan for the model wrappers."""

    def __init__(self, seqs, plens, cu_flat, tp, pad_multiple, device, sp, min_trunk=None, halo=1):
        groups, lay, t2f, f2t, tree_pos, sib, gid, t_zero, f_zero = build_maps(seqs, plens, cu_flat, tp, pad_multiple,
                                                                               min_trunk, halo)
        self.groups, self.layout = groups, lay
        self.flat_total, self.tree_total = int(cu_flat[-1]), lay.total
        self.sib, self.gid = sib, gid
        self.tp, self.sp = tp, bool(sp) and tp > 1
        # Megatron GPTModel hands the layers a raw [max_seqlen] RoPE table: per_token_rotary stays False
        self.plan = ptm.RankPlan(lay, tree_pos, device, sp=self.sp)
        self.psp = ptm._make_psp(lay.cu, lay.num_samples, lay.max_seqlen, device)
        self.psp.prefix_tree = self.plan
        tpr = self.plan.tp_rank
        self.t2f_local = torch.as_tensor(self._chunk(t2f, self.tree_total, tpr), device=device)
        self.f2t_local = torch.as_tensor(self._chunk(f2t, self.flat_total, tpr), device=device)
        tz, fz = self._chunk(t_zero, self.tree_total, tpr), self._chunk(f_zero, self.flat_total, tpr)
        self.t_zero_local = torch.as_tensor(tz, device=device) if tz.any() else None
        self.f_zero_local = torch.as_tensor(fz, device=device) if fz.any() else None
        self.n_grouped = int((sib >= 0).sum())

    def _chunk(self, m, total, r):
        if not self.sp:
            return m
        k = total // self.tp
        return m[r * k:(r + 1) * k]

    def to_tree(self, x):
        """flat block input (SP chunk if sp) -> tree block input (SP chunk if sp), differentiable."""
        return _gather_select(x, self.t2f_local, self.sp, self.t_zero_local)

    def to_flat(self, y):
        """tree block output -> flat block output, differentiable."""
        return _gather_select(y, self.f2t_local, self.sp, self.f_zero_local)


def _gather_select(x, idx_local, sp, zero=None):
    """x: this rank's rows (its SP chunk if sp) of one pack -> this rank's rows of the other pack; idx_local = rows of
    the full source pack.  Differentiable (SP all-gather / reduce-scatter; index_select / index_add)."""
    if sp:
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
        x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True,
                                                 group=mpu.get_tensor_model_parallel_group())
    y = ptm.rows(x, idx_local)
    if zero is not None:
        y = y.masked_fill(zero.view(-1, *([1] * (y.dim() - 1))), 0)
    return y


# ---- slime / Megatron hooks -------------------------------------------------------------------------------------

def _wrap_get_batch(fn):
    def get_batch(data_iterator, keys, *a, **k):
        batch = fn(data_iterator, keys, *a, **k)
        _STATE['get_batch_calls'] += 1
        if not enabled():
            return batch
        psp = batch.get('packed_seq_params')
        tl, rl, toks = batch.get('total_lengths'), batch.get('response_lengths'), batch.get('unconcat_tokens')
        if psp is None or tl is None or rl is None or toks is None:
            return batch
        psp._pt_src = {'tokens': toks, 'plens': [int(t) - int(r) for t, r in zip(tl, rl)],
                       'flat_tokens': batch.get('tokens'), 'loss_mask': batch.get('full_loss_masks')}
        return batch
    get_batch._prefix_tree = True
    return get_batch


def _tree_of(block, psp):
    if hasattr(psp, '_pt_tree'):
        return psp._pt_tree
    from megatron.core import parallel_state as mpu
    if mpu.get_context_parallel_world_size() > 1:
        _fallback('cp>1')
        return None
    if getattr(block, 'pre_process', True) is False or getattr(block, 'post_process', True) is False:
        _fallback('pp>1')
        return None
    src = psp._pt_src
    t0 = time.perf_counter()
    toks = src['tokens']
    lens = [int(t.numel()) for t in toks]
    allt = torch.cat([t.reshape(-1) for t in toks]).to('cpu').numpy()
    off = np.concatenate([[0], np.cumsum(lens)])
    seqs = [allt[off[i]:off[i + 1]] for i in range(len(toks))]
    cu = list(getattr(psp, 'cu_seqlens_q_cpu', None) or psp.cu_seqlens_q.tolist())
    tp = mpu.get_tensor_model_parallel_world_size()
    sp = bool(getattr(block.config, 'sequence_parallel', False))
    pad_mult = int(os.environ.get('PREFIX_TREE_PAD_MULTIPLE', '128'))
    tree = SlimeTree(seqs, src['plens'], cu, tp, pad_mult, psp.cu_seqlens_q.device, sp, halo=ptm.required_halo(1))
    _STATE['build_s'] += time.perf_counter() - t0
    if not any(T for T, _ in tree.groups):
        _fallback('no siblings in micro-batch')
        psp._pt_tree = None
        return None
    # the RoPE table GPTModel built for this micro-batch covers max_seqlen of the flat pack; every virtual document
    # must fit (they are the original samples, only the virtual trailing pad document can be longer)
    if tree.layout.max_seqlen > int(getattr(psp, 'max_seqlen_q', 0) or 0):
        _fallback('virtual document longer than the flat max_seqlen')
        psp._pt_tree = None
        return None
    psp._pt_tree = tree
    _STATE['tree_calls'] += 1
    _STATE['flat_tokens'] += tree.flat_total
    _STATE['tree_tokens'] += tree.tree_total
    _STATE['groups'] += sum(1 for T, _ in tree.groups if T)
    _STATE['grouped_samples'] += tree.n_grouped
    _STATE['singletons'] += len(seqs) - tree.n_grouped
    _STATE['samples'] += len(seqs)
    n = _STATE['tree_calls']
    if n in (1, 2, 10, 100) or n % 500 == 0 or (LOG_EVERY and n % LOG_EVERY == 0):
        _log(json.dumps(snapshot()))
    return tree


def _wrap_block_forward(orig):
    def forward(self, *args, **kw):
        _STATE['block_calls'] += 1
        psp = kw.get('packed_seq_params')
        if (not enabled() or psp is None or getattr(psp, '_pt_src', None) is None or args
                or getattr(psp, 'prefix_tree', None) is not None):
            return orig(self, *args, **kw)
        if not getattr(self, '_pt_installed', False):
            rep = ptm.install_on_model(self)
            self._pt_installed = True
            _log(f'model wrappers: {json.dumps(rep, default=str)}')
        if not ptm.usable():
            _fallback('model side unusable: ' + str(ptm.snapshot().get('unsupported', 'no layers')))
            return orig(self, *args, **kw)
        tree = _tree_of(self, psp)
        if tree is None:
            return orig(self, *args, **kw)
        kw_t = dict(kw, hidden_states=tree.to_tree(kw['hidden_states']), packed_seq_params=tree.psp)
        check = CHECK and _STATE['check_calls'] < CHECK
        out = orig(self, **kw_t)
        rest = ()
        if isinstance(out, tuple):
            out, rest = out[0], tuple(out[1:])
        y = tree.to_flat(out)
        if check:
            _STATE['check_calls'] += 1
            try:
                _check(self, orig, kw, psp, tree, y, grad_pass=torch.is_grad_enabled())
            except Exception as e:  # noqa: BLE001
                _log(f'check failed: {e!r}'[:400], all_ranks=True)
        return (y,) + rest if rest else y
    forward._prefix_tree = True
    return forward


def _wrap_gpt_forward(orig):
    def forward(self, *a, **k):
        _CTX['gpt'] = self
        return orig(self, *a, **k)
    forward._prefix_tree = True
    return forward


# ---- numerics check ---------------------------------------------------------------------------------------------

@torch.no_grad()
def _resp_logp(h_local, flat_tokens, rows, sp):
    """log p(next token) on the given flat rows from the block output (SP chunk if sp), vocab-parallel."""
    from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
    gpt = _CTX['gpt']
    if sp:
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
        h_local = gather_from_sequence_parallel_region(h_local, tensor_parallel_output_grad=False,
                                                       group=mpu.get_tensor_model_parallel_group())
    W = gpt.shared_embedding_or_output_weight() if gpt.share_embeddings_and_output_weights else gpt.output_layer.weight
    h = h_local[:, 0]
    tgt = flat_tokens.reshape(-1)
    out = torch.empty(len(rows), dtype=torch.float32, device=h.device)
    for a in range(0, len(rows), 2048):
        r = rows[a:a + 2048]
        logits = (h.index_select(0, r) @ W.t()).float()
        out[a:a + 2048] = -vocab_parallel_cross_entropy(logits[:, None, :], tgt.index_select(0, r + 1)[:, None])[:, 0]
    return out


@torch.no_grad()
def _check(block, orig, kw, psp, tree, y_tree, grad_pass=False):
    src = psp._pt_src
    n = len(src['tokens'])
    lens = [int(t.numel()) for t in src['tokens']]
    cu = list(getattr(psp, 'cu_seqlens_q_cpu', None) or psp.cu_seqlens_q.tolist())
    x = kw['hidden_states']
    dev = x.device
    y_flat = orig(block, **kw)
    if isinstance(y_flat, tuple):
        y_flat = y_flat[0]
    # re-cut: same samples, reversed order in the pack (the noise floor of a different packing)
    order = list(range(n))[::-1]
    rcu = [0]
    for i in order:
        rcu.append(rcu[-1] + lens[i])
    pad = cu[-1] - cu[n]
    if pad:
        rcu.append(rcu[-1] + pad)
    r2f = np.concatenate([cu[i] + np.arange(lens[i]) for i in order] + ([cu[n] + np.arange(pad)] if pad else []))
    f2r = np.empty_like(r2f)
    f2r[r2f] = np.arange(len(r2f))
    rpsp = ptm._make_psp(rcu, n + (1 if pad else 0), max(rcu[i + 1] - rcu[i] for i in range(len(rcu) - 1)), dev)
    tp_rank = tree.plan.tp_rank
    ch = lambda m: torch.as_tensor(tree._chunk(m, len(m), tp_rank), device=dev)
    y_r = orig(block, **dict(kw, hidden_states=_gather_select(x, ch(r2f), tree.sp), packed_seq_params=rpsp))
    if isinstance(y_r, tuple):
        y_r = y_r[0]
    y_recut = _gather_select(y_r, ch(f2r), tree.sp)
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
    rec = {'t': time.time(), 'rank': _rank(), 'call': _STATE['check_calls'], 'n': n, 'grad_pass': grad_pass,
           'flat_tokens': tree.flat_total, 'tree_tokens': tree.tree_total,
           'hidden_rel_tree': rel(y_tree, y_flat), 'hidden_rel_recut': rel(y_recut, y_flat)}
    ft, lm = src.get('flat_tokens'), src.get('loss_mask')
    if _CTX['gpt'] is not None and ft is not None:
        rows, owner = [], []
        for i in range(n):
            p = src['plens'][i]
            r = np.arange(cu[i] + max(p, 1) - 1, cu[i] + lens[i] - 1)
            rows.append(r)
            owner.append(np.full(len(r), i))
        rows_np, owner = np.concatenate(rows), np.concatenate(owner)
        rows_t = torch.as_tensor(rows_np, device=dev)
        lp = {k: _resp_logp(v, ft, rows_t, tree.sp) for k, v in (('flat', y_flat), ('tree', y_tree), ('recut', y_recut))}
        mask = lm.reshape(-1).index_select(0, rows_t).bool().cpu().numpy() if lm is not None else np.ones(len(rows_np), bool)
        dt = (lp['tree'] - lp['flat']).abs().cpu().numpy()
        dr = (lp['recut'] - lp['flat']).abs().cpu().numpy()
        sib0 = np.isin(owner, [i for i in range(n) if tree.sib[i] == 0])
        sib1 = np.isin(owner, [i for i in range(n) if tree.sib[i] >= 1])
        solo = np.isin(owner, [i for i in range(n) if tree.sib[i] < 0])
        for nm, sel in (('sib0', sib0), ('sib1plus', sib1), ('singleton', solo), ('all', np.ones_like(sib0))):
            for msk, sfx in ((sel, ''), (sel & mask, '_trained')):
                if msk.any():
                    rec[f'{nm}{sfx}'] = {'rows': int(msk.sum()), 'tree_mean': float(dt[msk].mean()),
                                         'tree_max': float(dt[msk].max()), 'recut_mean': float(dr[msk].mean()),
                                         'recut_max': float(dr[msk].max())}
    d = os.environ.get('ASRT_LOG_DIR')
    if d:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f'prefix_tree_check-rank{_rank()}.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')
    _log('check ' + json.dumps(rec), all_ranks=True)


# ---- install ----------------------------------------------------------------------------------------------------

def install():
    """Patch slime's get_batch and Megatron's TransformerBlock.forward (training actor).  Idempotent."""
    if _STATE['installed']:
        return {'already': True}
    import importlib
    done = []
    for name in ('slime.backends.megatron_utils.model', 'slime.backends.megatron_utils.data'):
        m = importlib.import_module(name)
        f = getattr(m, 'get_batch', None)
        if f is not None and not getattr(f, '_prefix_tree', False):
            m.get_batch = _wrap_get_batch(f)
            done.append(name + '.get_batch')
    from megatron.core.transformer.transformer_block import TransformerBlock
    if not getattr(TransformerBlock.forward, '_prefix_tree', False):
        TransformerBlock.forward = _wrap_block_forward(TransformerBlock.forward)
        done.append('TransformerBlock.forward')
    if CHECK:
        from megatron.core.models.gpt.gpt_model import GPTModel
        if not getattr(GPTModel.forward, '_prefix_tree', False):
            GPTModel.forward = _wrap_gpt_forward(GPTModel.forward)
            done.append('GPTModel.forward (check)')
    _STATE['installed'] = True
    return {'component': 'prefix_tree.slime', 'enabled': enabled(), 'patched': done, 'check': CHECK,
            'min_trunk': MIN_TRUNK}


def megatron_init(args=None, *rest, **kwargs):
    """slime ``--custom-megatron-init-path`` hook (or an entry of ASRT_CHAIN_MEGATRON_INIT)."""
    if not enabled():
        return {'component': 'prefix_tree.slime', 'enabled': False}
    rep = install()
    _log(json.dumps(rep))
    return rep


def snapshot():
    s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _STATE.items()}
    if s['flat_tokens']:
        s['tree_over_flat'] = round(s['tree_tokens'] / s['flat_tokens'], 4)
    m = ptm.snapshot()
    s['model'] = {k: m[k] for k in ('installed_layers', 'attn_calls', 'attn_core_mode_calls', 'core_calls',
                                    'gdn_calls', 'gdn_state_calls', 'plans') if k in m}
    if 'unsupported' in m:
        s['model']['unsupported'] = m['unsupported']
    return s
