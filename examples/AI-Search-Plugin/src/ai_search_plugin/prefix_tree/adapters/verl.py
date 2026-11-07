# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-tree training inside verl's Megatron engine (training workers).

Switch: ASRT_PREFIX_TREE=1 and ``install()`` in the training worker before the first forward (done by
``ai_search_plugin.prefix_tree.verl_plugin`` when verl's Megatron engine module is imported).  The model-side
wrappers (``model.install_on_model``) are installed lazily on the first micro-batch that has siblings.

Per micro-batch (old-log-prob pass, reference pass and PPO update alike), with no change to verl's data, loss or
optimizer:
  * finds samples of the micro-batch that share a token prefix (GRPO siblings share the prompt; detected from the
    tokens, so no uid plumbing), longest common prefix per group (``plan.group_consecutive``);
  * feeds the model a tree pack instead: per group one trunk segment (T tokens, T <= lcp-1, a multiple of the THD
    alignment) + one branch segment per sample; singletons stay whole segments;
  * position ids of branch tokens continue from T (verl's per-segment positions are replaced); the rolled label of
    the trunk's last row is the shared next token, so every row's log-prob equals the flat path's;
  * maps the per-segment outputs (log-probs, entropy) back to per-sample nested tensors (trunk rows repeated for each
    sibling), so verl's post-processing, loss and metrics see exactly what they saw before.

Micro-batching: verl's dynamic batching (``rearrange_micro_batches``) is told to keep siblings together and to count
tree tokens toward the same token budget (``plan.sibling_units``).  Rollout routing replay (R3, MoE): the replay
routes are packed in the same tree order (sibling 0's routes on the trunk).

Scope: verl's Megatron engine, THD packing, zigzag CP layout (any CP size), TP with sequence parallel, standard
RoPE (raw table; GPTModel) or no RoPE.  Not handled (flat path, counted): value model, MTP training, non-THD data,
micro-batches without siblings.  Not supported: per-token rotary from custom model code (set
``ASRT_PREFIX_TREE_PER_TOKEN_ROTARY=1`` only if the model passes per-token angles), contiguous CP layout,
quantization-aligned THD packing (the pack would differ from the tree layout: raises).
Written against verl's v1 trainer / Megatron engine interfaces (``verl.models.mcore`` forward-fn getters,
``preprocess_thd_engine``, ``verl.workers.engine.utils.rearrange_micro_batches``); every patch point is optional.
"""
import os

import torch

from .. import model as ptm
from ..plan import TreeLayout, group_consecutive, sibling_units

_STATE = {'calls': 0, 'tree_calls': 0, 'flat_tokens': 0, 'tree_tokens': 0, 'groups': 0, 'singletons': 0,
          'mb_patch': 0, 'nontree_calls': 0, 'nontree_samples': 0, 'nontree_tokens': 0, 'nontree_reasons': {},
          'positions_fixed': 0}
_PENDING = {}
_MODELS = set()
MIN_FRAC = float(os.environ.get('ASRT_PREFIX_TREE_MIN_FRAC', '0.25'))
_LOG_EVERY = int(os.environ.get('PREFIX_TREE_LOG_EVERY', '0'))


def enabled():
    return os.environ.get('ASRT_PREFIX_TREE', '0') not in ('0', '', 'false', 'False')


def _count_nontree(reason, input_ids=None):
    _STATE['nontree_calls'] += 1
    r = _STATE['nontree_reasons']
    r[reason] = r.get(reason, 0) + 1
    if input_ids is not None and getattr(input_ids, 'is_nested', False):
        lens = input_ids.offsets().diff().tolist()
        _STATE['nontree_samples'] += len(lens)
        _STATE['nontree_tokens'] += int(sum(lens))


def _ensure_model(model):
    """Install the model-side wrappers on this model once; True when the tree path can be used."""
    if id(model) not in _MODELS:
        rep = ptm.install_on_model(model)
        _MODELS.add(id(model))
        _log(f'model wrappers: {rep}')
    return ptm.usable()


def plan_groups(seqs, align, min_trunk=None):
    return group_consecutive(seqs, align, min_trunk, MIN_FRAC)


def _align():
    from megatron.core import parallel_state as mpu
    tp = mpu.get_tensor_model_parallel_world_size()
    cp = mpu.get_context_parallel_world_size()
    return (2 * cp * tp if cp > 1 else tp), cp, tp


def _round_up(x, m):
    return -(-x // m) * m


def tree_lengths(seqs, align):
    """(flat tokens, tree tokens) after the per-segment padding."""
    flat = sum(_round_up(len(s), align) for s in seqs)
    tree = 0
    for T, idx in plan_groups(seqs, align):
        tree += T + sum(_round_up(len(seqs[i]) - T, align) for i in idx)
    return flat, tree


def build_tree(input_ids, prompt_lens=None, align=None, cp=None, tp=None, halo=None):
    """Per-sample nested input_ids -> tree segments (or None when no siblings share a prefix).

    prompt_lens (optional): trunk capped at prompt-1 so every response token stays in a branch.  align / cp / tp
    default to the Megatron parallel state; halo to what the installed GDN layers need."""
    if align is None:
        align, cp, tp = _align()
    offs = input_ids.offsets().tolist()
    vals = input_ids.values()
    seqs_dev = [vals[a:b] for a, b in zip(offs[:-1], offs[1:])]
    seqs = [s.cpu() for s in seqs_dev]
    groups = plan_groups(seqs, align)
    if prompt_lens is not None:
        capped = []
        for T, idx in groups:
            if T:
                T = min(T, (min(prompt_lens[i] for i in idx) - 1) // align * align)
                if T < 4 * align:
                    T = 0
            if T or len(idx) == 1:
                capped.append((T, idx))
            else:
                capped += [(0, [i]) for i in idx]
        groups = capped
    if all(T == 0 for T, _ in groups):
        _count_nontree('no_shared_group' if len(seqs) > 1 else 'single_sample', input_ids)
        return None
    segs, lab, spec, where = [], [], [], {}
    for T, idx in groups:
        if T:
            trunk = seqs_dev[idx[0]][:T]
            segs.append(trunk)
            # rolled label of the trunk's last row = the shared token right after the trunk
            lab.append(torch.cat([seqs_dev[idx[0]][T:T + 1], trunk[1:]]))
            tseg = len(segs) - 1
            _STATE['groups'] += 1
        else:
            tseg = None
            _STATE['singletons'] += 1
        blens = []
        for i in idx:
            b = seqs_dev[i][T:]
            segs.append(b)
            lab.append(b)
            where[i] = (tseg, len(segs) - 1, T)
            blens.append(_round_up(len(b), align))
        spec.append((T, blens))
    if halo is None:
        halo = ptm.required_halo(align)
    layout = TreeLayout(spec, cp, tp, pad_multiple=1, halo=_round_up(halo, align), align=align)
    pos = []
    for T, blens in spec:
        if T:
            pos.append(torch.arange(T))
        for b in blens:
            pos.append(torch.arange(T, T + b))
    _STATE['tree_calls'] += 1
    _STATE['flat_tokens'] += sum(_round_up(len(s), align) for s in seqs)
    _STATE['tree_tokens'] += layout.total
    n = _STATE['tree_calls']
    if n in (1, 10, 100) or n % 500 == 0 or (_LOG_EVERY and n % _LOG_EVERY == 0):
        _log(snapshot())
    return {'seg_ids': torch.nested.as_nested_tensor(segs, layout=torch.jagged),
            'seg_lab': torch.nested.as_nested_tensor(lab, layout=torch.jagged),
            'groups': groups, 'where': where, 'n': len(seqs), 'layout': layout, 'tree_pos': torch.cat(pos)}


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0


def _log(msg):
    if _rank() == 0:
        print(f'[asrt prefix_tree.verl] {msg}', flush=True)


def map_back(out, tree):
    """Per-segment nested outputs -> per-sample nested outputs (trunk rows repeated for every sibling)."""
    if not isinstance(out, dict):   # non-last PP stage: hidden states only
        return out
    res = {}
    for key, v in out.items():
        if not (torch.is_tensor(v) and getattr(v, 'is_nested', False)):
            res[key] = v
            continue
        vo = v.offsets().tolist()
        vv = v.values()
        seg = [vv[a:b] for a, b in zip(vo[:-1], vo[1:])]
        per = []
        for i in range(tree['n']):
            tseg, bseg, T = tree['where'][i]
            per.append(seg[bseg] if tseg is None else torch.cat([seg[tseg][:T], seg[bseg]]))
        res[key] = torch.nested.as_nested_tensor(per, layout=torch.jagged)
    return res


def segments_of(nested, groups):
    """Per-sample nested tensor -> per-segment nested tensor in tree order (trunk rows from the first sibling)."""
    offs = nested.offsets().tolist()
    vals = nested.values()
    seqs = [vals[a:b] for a, b in zip(offs[:-1], offs[1:])]
    out = []
    for T, idx in groups:
        if T:
            out.append(seqs[idx[0]][:T])
        for i in idx:
            out.append(seqs[i][T:])
    return torch.nested.as_nested_tensor(out, layout=torch.jagged)


def reorder_nested(x, order):
    """Jagged nested tensor -> the same per-sample pieces in 'order' (None = an empty piece for a trunk segment)."""
    offs = x.offsets().tolist()
    vals = x.values()
    pieces, lens = [], []
    for i in order:
        if i is None:
            lens.append(0)
        else:
            pieces.append(vals[offs[i]:offs[i + 1]])
            lens.append(offs[i + 1] - offs[i])
    v = torch.cat(pieces) if pieces else vals[:0]
    o = torch.tensor([0] + lens, dtype=x.offsets().dtype, device=x.offsets().device).cumsum(0)
    return torch.nested.nested_tensor_from_jagged(v, offsets=o)


def _segment_order(groups):
    order = []
    for T, idx in groups:
        if T:
            order.append(None)
        order += idx
    return order


def _run_tree(fn, model, tree, *args, **kw):
    _PENDING['spec'] = (tree['layout'], tree['tree_pos'])
    _PENDING.pop('attached', None)
    try:
        out = fn(model, *args, **kw)
        if _PENDING.get('attached') is not _PENDING['spec']:
            raise RuntimeError('prefix_tree: the tree pack never reached preprocess_thd_engine (verl changed its '
                               'packing path); refusing to continue with a pack the model wrappers do not know')
        return out
    finally:
        _PENDING.pop('spec', None)
        _PENDING.pop('attached', None)


def _wrap_fused_fn(fn):
    """verl use_fused_kernels=True path: fn(model, input_ids, labels, ...)."""
    def tree_fn(model, input_ids, labels, *args, **kw):
        _STATE['calls'] += 1
        if not enabled() or not getattr(input_ids, 'is_nested', False):
            _count_nontree('disabled_or_not_nested', input_ids)
            return fn(model, input_ids, labels, *args, **kw)
        if not _ensure_model(model):
            _count_nontree('model_unusable', input_ids)
            return fn(model, input_ids, labels, *args, **kw)
        tree = build_tree(input_ids)
        if tree is None:
            return fn(model, input_ids, labels, *args, **kw)
        out = _run_tree(fn, model, tree, tree['seg_ids'], tree['seg_lab'], *args, **kw)
        return map_back(out, tree)

    tree_fn._prefix_tree = True
    return tree_fn


def _wrap_engine_fn(fn):
    """verl use_fused_kernels=False path: fn(model, input_ids, multi_modal_inputs, logits_processor=...,
    logits_processor_args={'label', 'temperature', 'loss_mask', 'response_attention_mask'}, ...)."""
    def tree_fn(model, input_ids, multi_modal_inputs, *args, **kw):
        _STATE['calls'] += 1
        lpa = kw.get('logits_processor_args')
        if (not enabled() or not getattr(input_ids, 'is_nested', False) or kw.get('mtp_enable_train')
                or kw.get('data_format', 'thd') != 'thd' or kw.get('value_model') or multi_modal_inputs):
            _count_nontree('unsupported_path', input_ids)
            return fn(model, input_ids, multi_modal_inputs, *args, **kw)
        if not _ensure_model(model):
            _count_nontree('model_unusable', input_ids)
            return fn(model, input_ids, multi_modal_inputs, *args, **kw)
        lm, ram = (lpa or {}).get('loss_mask'), (lpa or {}).get('response_attention_mask')
        padded_masks = (torch.is_tensor(lm) and torch.is_tensor(ram) and not getattr(lm, 'is_nested', False)
                        and not getattr(ram, 'is_nested', False))
        plens = None
        lens = input_ids.offsets().diff().tolist()
        if padded_masks:
            rl = ram.to(torch.int64).sum(-1).tolist()
            plens = [L - r for L, r in zip(lens, rl)]
        elif torch.is_tensor(lm) and getattr(lm, 'is_nested', False):
            # nested response-only masks: response length = the mask piece's length; cap the trunk at the prompt so
            # siblings sharing their first response tokens still keep their whole response in the branch
            rl = lm.offsets().diff().tolist()
            if len(rl) == len(lens):
                plens = [L - r for L, r in zip(lens, rl)]
        tree = build_tree(input_ids, plens)
        if tree is None:
            return fn(model, input_ids, multi_modal_inputs, *args, **kw)
        if lpa:
            new = {}
            for k, v in lpa.items():
                if k == 'label':
                    new[k] = tree['seg_lab']
                elif k == 'temperature' and torch.is_tensor(v) and getattr(v, 'is_nested', False):
                    new[k] = segments_of(v, tree['groups'])
                elif k in ('loss_mask', 'response_attention_mask'):
                    continue
                else:
                    new[k] = v
            order = _segment_order(tree['groups'])
            if padded_masks:
                # per-segment response-only masks (trunk: none), so row planners that build "prompt zeros + response"
                # per sequence see the branch's own response suffix
                z = lm.new_zeros(lm.shape[1:])
                new['loss_mask'] = torch.stack([z if i is None else lm[i] for i in order])
                zr = ram.new_zeros(ram.shape[1:])
                new['response_attention_mask'] = torch.stack([zr if i is None else ram[i] for i in order])
            elif torch.is_tensor(lm) and getattr(lm, 'is_nested', False) and (ram is None or torch.is_tensor(ram)):
                # nested response-only masks, reordered to the tree segments (trunk: empty response), so a
                # response-rows output layer still runs on response rows only
                new['loss_mask'] = reorder_nested(lm, order)
                if torch.is_tensor(ram):
                    if getattr(ram, 'is_nested', False):
                        new['response_attention_mask'] = reorder_nested(ram, order)
                    else:
                        zr = ram.new_zeros(ram.shape[1:])
                        new['response_attention_mask'] = torch.stack([zr if i is None else ram[i] for i in order])
            kw = dict(kw, logits_processor_args=new)
        out = _run_tree(fn, model, tree, tree['seg_ids'], multi_modal_inputs, *args, **kw)
        return map_back(out, tree)

    tree_fn._prefix_tree = True
    return tree_fn


def _cfg_sp():
    from megatron.core import parallel_state as mpu
    return mpu.get_tensor_model_parallel_world_size() > 1   # verl's Megatron engine: sequence_parallel = tp > 1


def _wrap_preprocess(fn):
    def preprocess_thd_engine(input_ids, *args, **kw):
        out = fn(input_ids, *args, **kw)
        spec = _PENDING.get('spec')
        if spec is None or kw.get('need_roll', False) or _PENDING.get('attached') is spec:
            return out
        layout, tree_pos = spec
        if kw.get('cp_layout', 'zigzag') != 'zigzag' and layout.cp > 1:
            raise RuntimeError('prefix_tree: only the zigzag CP layout is supported')
        psp = out[1]
        padded = getattr(psp, 'cu_seqlens_q_padded', None)
        if padded is not None and padded.tolist() != layout.cu:
            raise RuntimeError('prefix_tree: verl packing differs from the tree layout (alignment / bucket padding); '
                               f'pack cu {padded.tolist()[:8]}... vs tree cu {layout.cu[:8]}...')
        dev = input_ids.values().device if getattr(input_ids, 'is_nested', False) else input_ids.device
        psp.prefix_tree = ptm.RankPlan(layout, tree_pos, dev, sp=_cfg_sp(),
                                       per_token_rotary=os.environ.get('ASRT_PREFIX_TREE_PER_TOKEN_ROTARY', '0') == '1')
        _PENDING['attached'] = spec   # first packing per forward = the model input (later ones: label / temperature)
        # GPTModel sizes the rotary table from max_seqlen: it must cover the virtual documents (trunk + branch), not
        # just the longest tree segment
        for f in ('max_seqlen_q', 'max_seqlen_kv'):
            cur = getattr(psp, f, None)
            if cur is not None:
                setattr(psp, f, max(int(cur.item() if torch.is_tensor(cur) else cur), int(layout.max_seqlen)))
        pos = out[2] if len(out) > 2 else None
        if torch.is_tensor(pos) and pos.shape[-1] == psp.prefix_tree.tree_local_len:
            # verl's per-segment positions restart at 0 in every branch; the tree's continue from the trunk
            tp_ = psp.prefix_tree.tree_pos_local.to(pos.dtype)
            pos = tp_.reshape(pos.shape) if pos.dim() == tp_.dim() else tp_.reshape(-1).expand_as(pos)
            out = (out[0], psp, pos) + tuple(out[3:])
            _STATE['positions_fixed'] += 1
        return out
    preprocess_thd_engine._prefix_tree = True
    return preprocess_thd_engine


def _wrap_rearrange(fn):
    """Micro-batching in tree mode: siblings are packed into units that fit max_token_len (trunk + branches; a group
    too large for one micro-batch is split and its trunk repeated); verl's own balancer (budget check, DP-synchronised
    micro-batch count, workload sort) then places whole units, counting their tree tokens."""
    def rearrange_micro_batches(batch, *args, **kw):
        if not (enabled() and 'input_ids' in batch.keys() and getattr(batch['input_ids'], 'is_nested', False)):
            return fn(batch, *args, **kw)
        real = batch['input_ids']
        offs = real.offsets().tolist()
        vals = real.values().cpu()
        seqs = [vals[a:b] for a, b in zip(offs[:-1], offs[1:])]
        align, _, _ = _align()
        budget = kw.get('max_token_len', args[0] if args else None)
        units = sibling_units(seqs, align, budget or 1 << 62, MIN_FRAC)
        if all(len(u[1]) == 1 for u in units):
            return fn(batch, *args, **kw)
        _STATE['mb_patch'] += 1
        import itertools
        from tensordict import TensorDict
        from verl.utils import tensordict_utils as tu
        sizes = [u[0] for u in units]
        dev = real.values().device
        fake_ids = torch.nested.nested_tensor_from_jagged(
            torch.zeros(sum(sizes), dtype=real.values().dtype, device=dev),
            offsets=torch.tensor([0] + list(itertools.accumulate(sizes)), device=real.offsets().device))
        fake = TensorDict({'input_ids': fake_ids}, batch_size=[len(units)])
        kw = dict(kw)
        kw['force_group_size'] = 1
        _, unit_parts = fn(fake, *args, **kw)
        parts = [[i for u in p for i in units[u][1]] for p in unit_parts]
        _STATE['units'] = _STATE.get('units', 0) + len(units)
        return [tu.index_select_tensor_dict(batch, p) for p in parts], parts
    rearrange_micro_batches._prefix_tree = True
    return rearrange_micro_batches


def _wrap_align_r3(fn):
    """R3: remember this micro-batch's grouping so the replay routes are packed in the same tree order."""
    def align_r3_router_replay_data(layers_topk_idx, input_ids, *a, **k):
        out = fn(layers_topk_idx, input_ids, *a, **k)
        _PENDING.pop('r3_groups', None)
        if enabled() and getattr(input_ids, 'is_nested', False):
            align, _, _ = _align()
            offs = input_ids.offsets().tolist()
            vals = input_ids.values().cpu()
            groups = plan_groups([vals[x:y] for x, y in zip(offs[:-1], offs[1:])], align)
            if any(T for T, _ in groups):
                _PENDING['r3_groups'] = groups
        return out
    align_r3_router_replay_data._prefix_tree = True
    return align_r3_router_replay_data


def _wrap_set_replay(fn):
    def set_router_replay_data(layers_topk_idx, attention_mask, *a, **k):
        groups = _PENDING.pop('r3_groups', None)
        if groups is not None and getattr(layers_topk_idx, 'is_nested', False):
            _STATE['r3_tree'] = _STATE.get('r3_tree', 0) + 1
            layers_topk_idx = segments_of(layers_topk_idx, groups)
            rm = k.get('replay_mask')
            if rm is not None and getattr(rm, 'is_nested', False):
                k['replay_mask'] = segments_of(rm, groups)
        return fn(layers_topk_idx, attention_mask, *a, **k)
    set_router_replay_data._prefix_tree = True
    return set_router_replay_data


def install():
    """Patch verl in a training worker (before the first forward).  Idempotent; returns the patched points."""
    import importlib
    done, missing = [], []
    vm = importlib.import_module('verl.models.mcore')
    for getter, wrap in (('get_mcore_forward_fused_model_engine_fn', _wrap_fused_fn),
                         ('get_mcore_engine_forward_fn', _wrap_engine_fn)):
        get = getattr(vm, getter, None)
        if get is None:
            missing.append(getter)
            continue
        if getattr(get, '_prefix_tree_getter', False):
            continue

        def getter_fn(*a, _get=get, _wrap=wrap, **k):
            return _wrap(_get(*a, **k))
        getter_fn._prefix_tree_getter = True
        setattr(vm, getter, getter_fn)
        done.append(getter)
    for modname in ('verl.models.mcore.util', 'verl.models.mcore.model_forward', 'verl.models.mcore.model_forward_fused'):
        try:
            m = importlib.import_module(modname)
        except ImportError:
            continue
        f = getattr(m, 'preprocess_thd_engine', None)
        if f is not None and not getattr(f, '_prefix_tree', False):
            m.preprocess_thd_engine = _wrap_preprocess(f)
            done.append(modname + '.preprocess_thd_engine')
    for modname, name, wrap in (('verl.workers.engine.utils', 'rearrange_micro_batches', _wrap_rearrange),
                                ('verl.workers.engine.megatron.transformer_impl', 'align_r3_router_replay_data',
                                 _wrap_align_r3),
                                ('verl.workers.engine.megatron.transformer_impl', 'set_router_replay_data',
                                 _wrap_set_replay)):
        try:
            m = importlib.import_module(modname)
        except ImportError:
            missing.append(modname)
            continue
        f = getattr(m, name, None)
        if f is None:
            missing.append(f'{modname}.{name}')
        elif not getattr(f, '_prefix_tree', False):
            setattr(m, name, wrap(f))
            done.append(f'{modname}.{name}')
    ptm.VIRTUAL_PADDED_CU['on'] = True
    return {'component': 'prefix_tree.verl', 'patched': done, 'missing': missing}


def snapshot():
    s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _STATE.items()}
    s['model'] = ptm.snapshot()
    if s['flat_tokens']:
        s['tree_over_flat'] = round(s['tree_tokens'] / s['flat_tokens'], 4)
    return s
