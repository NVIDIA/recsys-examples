# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-tree training, model side (Megatron backend).

The micro-batch is packed as a tree (``plan.TreeLayout``).  Token-wise modules (embedding, norms, MLP / MoE, residuals,
output layer, loss) run on the tree rows as they are.  The sequence mixer of every transformer layer
(``layer.self_attention``: softmax attention incl. GQA, or a gated-delta-net module) is wrapped per instance:

  'full' (default)  tree -> virtual 'full' pack (one document per sample = trunk + its branch) -> the original
                    forward -> tree (canonical copy of every row).  The attention module, its RoPE (raw table applied
                    per THD document, so positions restart per document exactly like the flat path) and its kernel
                    run unchanged on documents that are exactly the original samples.  Exact for any per-document
                    mixer (causal attention, GQA, sliding window, GDN, short conv).
  'core'            (PREFIX_TREE_ATTN=core, softmax attention only, models WITHOUT RoPE or with per-token rotary
                    angles at the tree positions): QKV / output projections on the tree rows, only
                    ``self_attention.core_attention`` sees the virtual pack.  With a raw RoPE table the module would
                    apply RoPE per tree segment (branch positions restarting at 0), so layers that receive a rotary
                    table always use 'full'.
  GDN 'state'       (PREFIX_TREE_GDN=state, opt-in): GDN modules run on the virtual 'halo' pack (trunk once + a short
                    look-back per branch for the causal conv) and the module-level ``chunk_gated_delta_rule`` is
                    wrapped so the trunk documents run once and every branch starts from its trunk's final recurrent
                    state (halo tokens get beta = g = 0, i.e. they leave the state unchanged).  Needs a
                    ``chunk_gated_delta_rule`` with FLA's signature (initial_state / output_final_state / cu_seqlens)
                    looked up from the module globals at call time; otherwise the GDN layers stay in 'full'.

expand / contract work on SP / CP shards with differentiable all-gathers (backward: reduce-scatter) and index_select
(backward: index_add), so the n sibling copies of a trunk row sum their gradients.

The wrappers are no-ops unless ``packed_seq_params.prefix_tree`` carries a :class:`RankPlan`, so an installed model
trains ordinary packs exactly as before.  Sequence mixers the wrappers do not know (Mamba, MLA, MTP blocks) disable
the component (``usable()`` is False and the adapters keep the flat path).

Numerics: on the flat path the n siblings each recompute the shared prefix; on the tree it is computed once (the
canonical copy = sibling 0's document).  With CP = 1 the prefix rows of every sibling see the same kernel on the same
document layout, so tree and flat agree bitwise in the forward.  With CP >= 2 the per-document zigzag split depends on
the document length, so in the flat path the shared prefix of sibling k is computed with a different split (and bf16
rounding) than sibling 0's; the tree uses sibling 0's rounding for all siblings, so siblings 1+ differ from the flat
path at bf16 noise level while sibling 0 is identical.  Gradients agree up to the same rounding (the sibling sum of a
trunk row happens per layer on the tree, at the end on the flat path).
"""
import os

import torch

from .plan import local_positions, rank_maps

VIRTUAL_PADDED_CU = {'on': False}   # verl packs carry cu_seqlens_q_padded; the virtual documents are fully valid
FAST_ROWS = {'on': os.environ.get('PREFIX_TREE_FAST_ROWS', '0') == '1'}
ATTN_MODE = {'mode': os.environ.get('PREFIX_TREE_ATTN', 'full')}     # 'full' | 'core'
GDN_MODE = {'mode': os.environ.get('PREFIX_TREE_GDN', 'full')}       # 'full' | 'state'
_STATE = {'installed_layers': 0, 'attn_calls': 0, 'attn_core_mode_calls': 0, 'core_calls': 0, 'gdn_calls': 0,
          'gdn_state_calls': 0, 'plans': 0, 'gdn_receptive': 0}
_CURRENT = {}

# ---- row maps with an optional deterministic backward ----------------------------------------------------------
# FAST_ROWS (PREFIX_TREE_FAST_ROWS=1): backward of the row maps without the atomic index_add over all rows: every source
# row takes the gradient of its first copy by a gather (unique rows), only the extra copies (trunk rows of siblings
# 1+) are added, one copy level at a time (unique rows per level -> deterministic order).  Forward unchanged.
_ROWS_CACHE = {}


def _rows_plan(idx, n_src):
    """(first_dst, first_src, all_present, [(dst_k, src_k) for copy levels k >= 1]) for y = x[idx]."""
    key = (idx.data_ptr(), idx.numel(), n_src)
    hit = _ROWS_CACHE.get(key)
    if hit is not None and hit[0] is idx:
        return hit[1]
    L = idx.numel()
    order = torch.argsort(idx, stable=True)
    sidx = idx.index_select(0, order)
    newg = torch.ones(L, dtype=torch.bool, device=idx.device)
    if L > 1:
        newg[1:] = sidx[1:] != sidx[:-1]
    pos = torch.arange(L, device=idx.device)
    start = torch.where(newg, pos, torch.zeros_like(pos)).cummax(0).values
    lvl = pos - start
    nlev = int(lvl.max().item()) + 1 if L else 0
    first = lvl == 0
    first_dst, first_src = order[first], sidx[first]
    all_present = int(first_src.numel()) == n_src
    extra = [(order[lvl == k], sidx[lvl == k]) for k in range(1, nlev)]
    plan = (first_dst.contiguous(), first_src, all_present, extra)
    if len(_ROWS_CACHE) > 256:
        _ROWS_CACHE.clear()
    _ROWS_CACHE[key] = (idx, plan)
    return plan


class _Rows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx):
        ctx.n_src, ctx.idx = x.shape[0], idx
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx, g):
        first_dst, first_src, all_present, extra = _rows_plan(ctx.idx, ctx.n_src)
        g = g.contiguous()
        if all_present:   # first_src == 0..n-1 in order (sorted)
            gx = g.index_select(0, first_dst)
        else:
            gx = g.new_zeros((ctx.n_src,) + tuple(g.shape[1:]))
            gx.index_copy_(0, first_src, g.index_select(0, first_dst))
        for dst, src in extra:
            gx.index_add_(0, src, g.index_select(0, dst))
        return gx, None


def rows(x, idx):
    """x.index_select(0, idx), with the FAST_ROWS backward when on."""
    if FAST_ROWS['on'] and x.requires_grad and torch.is_grad_enabled():
        return _Rows.apply(x, idx)
    return x.index_select(0, idx)


# ---- Megatron glue (lazy) ----------------------------------------------------------------------------------------

def _mpu():
    from megatron.core import parallel_state as mpu
    return mpu


class _PSP:
    """Stand-in for megatron.core.packed_seq_params.PackedSeqParams when Megatron is not importable (CPU tests)."""

    def __init__(self, **kw):
        self.cu_seqlens_q_padded = self.cu_seqlens_kv_padded = None
        self.__dict__.update(kw)


def _make_psp(cu, num_samples, max_seqlen, device):
    try:
        from megatron.core.packed_seq_params import PackedSeqParams
    except ImportError:
        PackedSeqParams = _PSP
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    p = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu_t, cu_seqlens_kv=cu_t, max_seqlen_q=int(max_seqlen),
                        max_seqlen_kv=int(max_seqlen))
    if VIRTUAL_PADDED_CU['on']:
        p.cu_seqlens_q_padded = cu_t
        p.cu_seqlens_kv_padded = cu_t
    p.cu_seqlens_q_cpu = list(cu)
    p.num_samples = int(num_samples)
    return p


class RankPlan:
    """Per-rank device-side view of one micro-batch's TreeLayout (attached as ``packed_seq_params.prefix_tree``).

    tree_position_ids: position id of every tree row (global, before the CP split), e.g. plan.tree_position_ids.
    sp: the model runs with sequence parallelism (hidden states between layers are TP-sharded along the sequence).
    per_token_rotary: the model hands the layers per-token rotary angles (one row per CP-local token) instead of a
        raw [max_seqlen] table.  Off for Megatron GPTModel ('rope' / 'yarn' ...: raw table); the adapter that knows
        its model sets it.  Never inferred from tensor shapes (a raw table can have the same length by chance)."""

    def __init__(self, layout, tree_position_ids, device, sp=None, per_token_rotary=False):
        self.layout = layout
        self.cp, self.tp = layout.cp, layout.tp
        mpu = _mpu() if (self.cp > 1 or self.tp > 1) else None
        self.cp_rank = mpu.get_context_parallel_rank() if self.cp > 1 else 0
        self.tp_rank = mpu.get_tensor_model_parallel_rank() if self.tp > 1 else 0
        self.sp = bool(sp) and self.tp > 1
        self.per_token_rotary = bool(per_token_rotary)
        self.device = device
        self.maps, self.vpsp, self.vpos = {}, {}, {}
        tree_pos = torch.as_tensor(tree_position_ids).reshape(-1).cpu().numpy()
        for kind, v in layout.virtual.items():
            m = rank_maps(layout, kind, self.cp_rank, self.tp_rank, self.sp)
            mm = {k: torch.as_tensor(m[k], device=device) for k in ('expand', 'contract', 'expand_cp', 'contract_cp')}
            for k in ('expand_zero', 'contract_zero', 'expand_cp_zero', 'contract_cp_zero'):
                mm[k] = None if m[k] is None else torch.as_tensor(m[k], device=device)
            mm['virtual_local_len'] = m['virtual_local_len']
            self.maps[kind] = mm
            self.vpsp[kind] = _make_psp(v['cu'], v['num_samples'], v['max_seqlen'], device)
            vl = local_positions(v['cu'], self.cp, self.cp_rank)
            self.vpos[kind] = torch.as_tensor(tree_pos[v['v_src'][vl]], device=device)[None, :]
        tl = local_positions(layout.cu, self.cp, self.cp_rank)
        self.tree_local_len = len(tl)
        self.tree_pos_local = torch.as_tensor(tree_pos[tl], device=device)[None, :]   # [1, local] position ids
        self._rot = {}
        self._gdn_split = None
        _STATE['plans'] += 1

    # ---- differentiable SP / CP gathers ----------------------------------------------------------------------
    def _gather(self, x, sp):
        if not ((sp and self.sp) or self.cp > 1):
            return x
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
        mpu = _mpu()
        if sp and self.sp:
            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True,
                                                     group=mpu.get_tensor_model_parallel_group())
        if self.cp > 1:
            x = gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=True,
                                                     group=mpu.get_context_parallel_group())
        return x

    def _gather_cp(self, x):
        return self._gather(x, False)

    @staticmethod
    def _zero(t, mask):
        """Zero the pad rows that were read from a real row of the other pack (no value, no gradient)."""
        if mask is None:
            return t
        return t.masked_fill(mask.view(-1, *([1] * (t.dim() - 1))), 0)

    def expand(self, x, kind):
        """tree local (SP chunk if sp) [s, b, h] -> virtual local (SP chunk if sp)."""
        m = self.maps[kind]
        return self._zero(rows(self._gather(x, True), m['expand']), m['expand_zero'])

    def contract(self, y, kind):
        """virtual local (SP chunk if sp) -> tree local (SP chunk if sp)."""
        m = self.maps[kind]
        return self._zero(rows(self._gather(y, True), m['contract']), m['contract_zero'])

    def expand_cp_grad(self, x, kind):
        """CP-local [s, ...] tensor that is not SP-split (q / k / v after the QKV projection) -> virtual CP-local."""
        m = self.maps[kind]
        return self._zero(rows(self._gather_cp(x), m['expand_cp']), m['expand_cp_zero'])

    def contract_cp_grad(self, y, kind):
        m = self.maps[kind]
        return self._zero(rows(self._gather_cp(y), m['contract_cp']), m['contract_cp_zero'])

    def expand_cp(self, x, kind, dim=0):
        """CP-local tensor that is NOT SP-split and needs no gradient (per-token rotary angles, ids)."""
        with torch.no_grad():
            if self.cp > 1:
                import torch.distributed as dist
                parts = [torch.empty_like(x) for _ in range(self.cp)]
                dist.all_gather(parts, x.contiguous(), group=_mpu().get_context_parallel_group())
                x = torch.cat(parts, dim=dim)
            return x.index_select(dim, self.maps[kind]['expand_cp'])

    # ---- kwargs of the virtual call ----------------------------------------------------------------------------
    def rotary(self, rot, kind):
        if rot is None or not self.per_token_rotary:
            return rot     # raw table: applied with the virtual cu_seqlens -> per-document positions, nothing to remap
        if isinstance(rot, (tuple, list)):
            return type(rot)(self.rotary(r, kind) for r in rot)
        if not torch.is_tensor(rot):
            return rot
        if rot.shape[0] != self.tree_local_len:
            raise RuntimeError(f'prefix_tree: per-token rotary has {rot.shape[0]} rows, expected the CP-local tree '
                               f'length {self.tree_local_len}')
        key = (id(rot), kind)
        hit = self._rot.get(key)
        if hit is None or hit[0] is not rot:
            hit = (rot, self.expand_cp(rot, kind, 0))
            self._rot[key] = hit
        return hit[1]

    def position_ids(self, pos, kind):
        if pos is None:
            return None
        if pos.shape[-1] == self.tree_local_len:
            if pos.dim() == 3:   # mrope [3, 1, T]
                return self.vpos[kind].expand(pos.shape[0], -1, -1)
            return self.vpos[kind]
        return pos

    def virtual_kwargs(self, kw, kind):
        kw = dict(kw)
        kw['packed_seq_params'] = self.vpsp[kind]
        if 'rotary_pos_emb' in kw:
            kw['rotary_pos_emb'] = self.rotary(kw['rotary_pos_emb'], kind)
        if kw.get('position_ids') is not None:
            kw['position_ids'] = self.position_ids(kw['position_ids'], kind)
        return kw


def plan_of(psp):
    return getattr(psp, 'prefix_tree', None) if psp is not None else None


def usable():
    """False until install_on_model wrapped at least one layer and found no unsupported sequence mixer."""
    return _STATE['installed_layers'] > 0 and not _STATE.get('disabled')


# ---- GDN state fork ------------------------------------------------------------------------------------------------

def _halo_state_plan(plan):
    """Host-side split of the 'halo' virtual pack for the GDN core (cached on the RankPlan).

    A: trunk documents (run once, final state kept); B: every other document (halo + branch, trailing pad) run from its
    group's trunk final state (zeros if none); halo rows get beta = 0 and g = 0 so they leave the state unchanged
    (their role for the short conv is already done before the core)."""
    hp = plan._gdn_split
    if hp is not None:
        return hp
    import numpy as np
    lay = plan.layout
    v = lay.virtual['halo']
    cu = v['cu']
    docs = []   # (kind, start, len, trunk doc index or -1, halo)
    d = 0
    for g in lay.groups:
        tdoc = -1
        if g['T']:
            docs.append(('trunk', cu[d], cu[d + 1] - cu[d], -1, 0))
            tdoc = d
            d += 1
        h = min(lay.halo, g['T'])
        for _ in g['branches']:
            docs.append(('branch', cu[d], cu[d + 1] - cu[d], tdoc, h))
            d += 1
    while d < len(cu) - 1:   # trailing pad document
        docs.append(('pad', cu[d], cu[d + 1] - cu[d], -1, 0))
        d += 1
    trunk_ids = [i for i, x in enumerate(docs) if x[0] == 'trunk']
    other_ids = [i for i, x in enumerate(docs) if x[0] != 'trunk']
    tpos = {i: k for k, i in enumerate(trunk_ids)}
    span = lambda i: np.arange(docs[i][1], docs[i][1] + docs[i][2])
    A = np.concatenate([span(i) for i in trunk_ids]) if trunk_ids else np.zeros(0, np.int64)
    B = np.concatenate([span(i) for i in other_ids])
    cuA = np.concatenate([[0], np.cumsum([docs[i][2] for i in trunk_ids])]) if trunk_ids else None
    cuB = np.concatenate([[0], np.cumsum([docs[i][2] for i in other_ids])])
    init = np.array([tpos[docs[i][3]] if docs[i][3] >= 0 else len(trunk_ids) for i in other_ids])
    keep = np.ones(len(B), dtype=np.float32)
    off = 0
    for i in other_ids:
        keep[off:off + docs[i][4]] = 0.0
        off += docs[i][2]
    inv = np.empty(len(A) + len(B), dtype=np.int64)
    inv[np.concatenate([A, B]).astype(np.int64)] = np.arange(len(A) + len(B))
    dev = plan.device
    T = lambda x, dt=torch.long: torch.as_tensor(x, dtype=dt, device=dev)
    hp = {'A': T(A), 'B': T(B), 'cuA': None if cuA is None else T(cuA, torch.int32), 'cuB': T(cuB, torch.int32),
          'init': T(init), 'keep': T(keep, torch.float32), 'inv': T(inv), 'total': int(v['total']),
          'nA': len(trunk_ids), 'nB': len(other_ids)}
    plan._gdn_split = hp
    return hp


def gdn_state_core(orig):
    """Wrap a module-level ``chunk_gated_delta_rule`` (FLA signature, [1, L, H, D] varlen with cu_seqlens): inside a
    'state'-mode GDN call on a prefix-tree pack, run the trunk documents once and the branch documents from their
    trunk's final state; otherwise call ``orig`` unchanged."""
    def core(q, k, v, g=None, beta=None, scale=None, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kw):
        plan = _CURRENT.get('gdn')
        if plan is None:
            return orig(q, k, v, g=g, beta=beta, scale=scale, initial_state=initial_state,
                        output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        cu_seqlens=cu_seqlens, **kw)
        hp = _halo_state_plan(plan)
        if (cu_seqlens is None or initial_state is not None or output_final_state or kw or q.shape[0] != 1
                or q.shape[1] != hp['total']):
            # inside a state-mode call the core MUST fork, else branches would start from a zero state
            raise RuntimeError('prefix_tree GDN state mode: unexpected core call (needs the whole virtual halo pack '
                               f'as one [1, {hp["total"]}, ...] varlen call; got q {tuple(q.shape)}, '
                               f'cu_seqlens={cu_seqlens is not None}, extra={sorted(kw)}); use PREFIX_TREE_GDN=full')
        _STATE['gdn_state_calls'] += 1
        sel = lambda t, i: t.index_select(1, i)
        H, K, V = v.shape[2], q.shape[3], v.shape[3]
        if hp['nA']:
            oA, S = orig(sel(q, hp['A']), sel(k, hp['A']), sel(v, hp['A']), g=sel(g, hp['A']), beta=sel(beta, hp['A']),
                         scale=scale, initial_state=None, output_final_state=True,
                         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=hp['cuA'])
            S0 = torch.cat([S, S.new_zeros((1,) + tuple(S.shape[1:]))]).index_select(0, hp['init'])
        else:
            oA = None
            S0 = q.new_zeros((hp['nB'], H, K, V), dtype=torch.float32)
        keep = hp['keep'].view(1, -1, 1)
        gB = sel(g, hp['B']) * keep.to(g.dtype)
        bB = sel(beta, hp['B']) * keep.to(beta.dtype)
        oB, _ = orig(sel(q, hp['B']), sel(k, hp['B']), sel(v, hp['B']), g=gB, beta=bB, scale=scale,
                     initial_state=S0, output_final_state=False, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                     cu_seqlens=hp['cuB'])
        o = oB if oA is None else torch.cat([oA, oB], 1)
        return o.index_select(1, hp['inv']), None
    core._prefix_tree_core = True
    return core


GDN_CORE_MODULES = ('megatron.core.ssm.gated_delta_net', 'megatron.core.ssm.gated_delta_net.common',
                    'megatron.core.ssm.gated_delta_net.gdn')


def install_gdn_core(module_names=GDN_CORE_MODULES):
    """Wrap every module-global ``chunk_gated_delta_rule`` binding (the GDN forward looks it up at call time)."""
    import importlib
    done = []
    for name in module_names:
        try:
            m = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
        f = getattr(m, 'chunk_gated_delta_rule', None)
        if f is not None and not getattr(f, '_prefix_tree_core', False):
            m.chunk_gated_delta_rule = gdn_state_core(f)
            done.append(name)
        elif f is not None:
            done.append(name)
    return done


# ---- per-instance wrappers -----------------------------------------------------------------------------------

def _orig_forward(mod, saved):
    """The module's forward at call time (honours class-level patches installed later), or a pre-existing
    instance-level forward."""
    if saved is not None:
        return saved
    return type(mod).forward.__get__(mod, type(mod))


def _is_gdn(mod):
    return hasattr(mod, 'A_log') and hasattr(mod, 'conv1d')


def _wrap_attention(mod):
    saved = mod.__dict__.get('forward')
    gdn = _is_gdn(mod)

    def forward(*args, **kw):
        plan = plan_of(kw.get('packed_seq_params'))
        orig = _orig_forward(mod, saved)
        if plan is None:
            return orig(*args, **kw)
        if args:
            hidden, args = args[0], args[1:]
        else:
            kw = dict(kw)
            hidden = kw.pop('hidden_states')
        if (not gdn and ATTN_MODE['mode'] == 'core' and getattr(mod, '_prefix_tree_core_wrapped', False)
                and (kw.get('rotary_pos_emb') is None or plan.per_token_rotary)):
            # QKV / RoPE / output projection on the tree rows; only core_attention is expanded (_wrap_core_attention)
            _STATE['attn_core_mode_calls'] += 1
            return orig(hidden, *args, **kw)
        kind = 'halo' if (gdn and GDN_MODE['mode'] == 'state') else 'full'
        _STATE['gdn_calls' if gdn else 'attn_calls'] += 1
        if kind == 'halo' and plan.layout.halo < _STATE['gdn_receptive'] and any(
                g['T'] > plan.layout.halo for g in plan.layout.groups):
            raise RuntimeError(f'prefix_tree: layout halo {plan.layout.halo} < GDN conv look-back '
                               f'{_STATE["gdn_receptive"]}')
        vx = plan.expand(hidden, kind)
        prev = _CURRENT.get('gdn')
        if kind == 'halo':
            _CURRENT['gdn'] = plan
        try:
            out = orig(vx, *args, **plan.virtual_kwargs(kw, kind))
        finally:
            if kind == 'halo':
                _CURRENT['gdn'] = prev
        if isinstance(out, tuple):
            return (plan.contract(out[0], kind),) + tuple(out[1:])
        return plan.contract(out, kind)

    mod.forward = forward


def _wrap_core_attention(attn):
    core = getattr(attn, 'core_attention', None)
    if core is None:
        return False
    saved = core.__dict__.get('forward')

    def forward(query, key, value, attention_mask, *args, **kw):
        orig = _orig_forward(core, saved)
        plan = plan_of(kw.get('packed_seq_params'))
        if plan is None:   # ordinary pack, or the 'full' path (virtual psp carries no plan)
            return orig(query, key, value, attention_mask, *args, **kw)
        _STATE['core_calls'] += 1
        kw = dict(kw)
        kw['packed_seq_params'] = plan.vpsp['full']
        q, k, v = (plan.expand_cp_grad(t, 'full') for t in (query, key, value))
        out = orig(q, k, v, attention_mask, *args, **kw)
        return plan.contract_cp_grad(out, 'full')

    core.forward = forward
    attn._prefix_tree_core_wrapped = True
    return True


_UNSUPPORTED = ('MultiLatentAttention', 'Mamba', 'MultiTokenPrediction')


def _is_layer(m):
    """A transformer layer whose ``self_attention`` is the sequence mixer (Megatron TransformerLayer and the like)."""
    sa = getattr(m, 'self_attention', None)
    return sa is not None and isinstance(sa, torch.nn.Module) and type(m).__name__.endswith('Layer')


def install_on_model(models, gdn_core_modules=GDN_CORE_MODULES):
    """Wrap every transformer layer of the given model chunk(s) (bare or DDP / Float16Module wrapped).  Idempotent.
    gdn_core_modules: modules whose global ``chunk_gated_delta_rule`` the GDN state mode wraps."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    n = 0
    receptive = 0
    for model in models:
        for m in model.modules():
            if not _is_layer(m) or getattr(m, '_prefix_tree_installed', False):
                continue
            sa = m.self_attention
            _wrap_attention(sa)
            if _is_gdn(sa):
                k = getattr(sa, 'conv_kernel_dim', None) or getattr(getattr(sa, 'conv1d', None), 'kernel_size', (1,))[0]
                receptive = max(receptive, int(k) - 1)
            elif ATTN_MODE['mode'] == 'core':
                _wrap_core_attention(sa)
            m._prefix_tree_installed = True
            n += 1
    _STATE['installed_layers'] += n
    _STATE['gdn_receptive'] = max(_STATE['gdn_receptive'], receptive)
    for model in models:   # sequence mixers we do not know how to treat -> refuse (a tree pack would be wrong)
        for name, m in model.named_modules():
            cls = type(m).__name__
            if any(k in cls for k in _UNSUPPORTED) and not getattr(m, '_prefix_tree_ok', False):
                _STATE['unsupported'] = f'{name}: {cls}'
    if n == 0 or 'unsupported' in _STATE:
        _STATE['disabled'] = True
    core = []
    if GDN_MODE['mode'] == 'state':
        core = install_gdn_core(gdn_core_modules)
        if not core:
            GDN_MODE['mode'], GDN_MODE['reason'] = 'full', 'no module-level chunk_gated_delta_rule found'
    return {'component': 'prefix_tree', 'layers_wrapped': n, 'gdn_mode': dict(GDN_MODE), 'attn_mode': dict(ATTN_MODE),
            'gdn_core_patched': core, 'usable': usable(), 'unsupported': _STATE.get('unsupported')}


def required_halo(align):
    """Halo length (multiple of align) that covers the GDN short-conv look-back of the installed layers."""
    h = max(align, _STATE['gdn_receptive'])
    return -(-h // align) * align


def snapshot():
    return dict(_STATE)
