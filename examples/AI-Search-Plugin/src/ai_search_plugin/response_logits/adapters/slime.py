"""slime (Megatron backend) adapter: output layer only at response rows.

Stock slime: forward_step (megatron_utils/model.py, train_one_step and forward_only) calls the model with
labels=None and gets full logits [1, T, V/TP]; loss.get_log_probs_and_entropy then computes vocab-parallel
log-prob/entropy over all T rows and slices the response rows per sample (cp1 / zigzag CP / allgather CP).

With the adapter:
  1. get_batch (as imported by megatron_utils/model.py) additionally builds the row plan of the micro-batch:
     slime's own get_responses() run on an arange "logit" tensor gives, for every sample, the local logit rows
     and target tokens it would read (handles THD packing, zigzag / allgather CP and the shift by one). The plan
     rides on batch["unconcat_tokens"] (a list subclass, so every loss helper receives it) and is registered with
     response_logits.request_rows() for the forward that follows.
  2. GPTModel._postprocess (patched by response_logits.install) computes the output layer only on those rows
     -> logits [1, n, V/TP] (TP vocab-parallel kept, SP gathered once, see response_logits docstring).
  3. get_log_probs_and_entropy (loss.py and actor.py references) sees the plan and the compact logits, runs
     slime's own calculate_log_probs_and_entropy on [n, V/TP] and splits the result per sample: same list of
     per-sample tensors, same order, same CP layout (allgather CP is redistributed exactly as stock).
Anything else (critic value head, MTP training, custom loss, score centering, logprob server, a forward whose
input_ids do not match the plan) falls back to the stock full-logits path.

Enable: --custom-megatron-init-path ai_search_plugin.response_logits.adapters.slime.megatron_init
        (or through ai_search_plugin.frameworks.slime.megatron_init, which chains every enabled component)
Tested against slime 0.4.0 as shipped in slimerl/slime:latest-cu130 (Megatron-Core 0.19.2 / 0.20).
"""
import functools
import inspect
import os
import sys
from types import SimpleNamespace

import torch

from ai_search_plugin import response_logits as _core

from . import rows_mode

_STATE = {'installed': False, 'reason': None, 'orig_get_batch': None, 'orig_logp': None, 'mode': None,
          'chunk_ok': True, 'chunk_off_reason': None}
_STATS = {'planned': 0, 'compact_logp': 0, 'fallback_logp': 0, 'chunked_logp': 0}
_UNSUPPORTED_ARGS = (  # features whose code reads the full logits tensor itself
    ('use_score_centering', True), ('enable_mtp_training', True), ('loss_type', 'custom_loss'))


class _TokenList(list):
    """batch["unconcat_tokens"] carrying the row plan (identical list for every other consumer)."""
    rl_plan = None


class _Plan:
    __slots__ = ('req', 'targets', 'counts', 'sel', 'resp_counts', 'n_resp')


def _make_plan(batch, allgather_cp):
    from slime.backends.megatron_utils import loss as L
    tokens = batch['tokens']
    T = tokens.size(1)
    probe = torch.arange(T, device=tokens.device, dtype=torch.float64).view(1, T, 1).float()
    rows, targets, counts = [], [], []
    for r, t in L.get_responses(probe, args=SimpleNamespace(allgather_cp=allgather_cp, rollout_temperature=1.0),
                                unconcat_tokens=batch['unconcat_tokens'], total_lengths=batch['total_lengths'],
                                response_lengths=batch['response_lengths'], apply_temperature=False):
        rows.append(r.reshape(-1))
        targets.append(t.reshape(-1).to(tokens.device))
        counts.append(int(r.numel()))
    rows = torch.cat(rows).round().long() if rows else tokens.new_zeros(0)
    targets = torch.cat(targets).long() if targets else tokens.new_zeros(0)
    plan = _Plan()
    plan.sel = None
    plan.counts = counts
    if _STATE['mode'] == 'loss':
        keep = batch['full_loss_masks'].reshape(-1).index_select(0, rows) > 0
        sel = keep.nonzero().squeeze(1)
        plan.sel, plan.resp_counts, plan.n_resp = sel, counts, int(rows.numel())
        rows, targets = rows.index_select(0, sel), targets.index_select(0, sel)
    plan.targets = targets
    plan.req = _core.RowRequest(rows, T, input_ids=tokens, allow_chunk=_STATE['chunk_ok'])
    return plan


def _get_batch(*args, **kwargs):
    batch = _STATE['orig_get_batch'](*args, **kwargs)
    try:
        need = ('tokens', 'unconcat_tokens', 'total_lengths', 'response_lengths', 'full_loss_masks')
        if not all(batch.get(k) is not None for k in need) or batch['tokens'].dim() != 2 or batch['tokens'].size(0) != 1:
            return batch
        allgather_cp = kwargs.get('allgather_cp', args[3] if len(args) > 3 else False)
        plan = _make_plan(batch, bool(allgather_cp))
    except Exception as e:  # never break training: stock path
        _core.clear_rows()
        print(f'[response_logits.slime] plan failed, stock path: {e!r}', file=sys.stderr)
        return batch
    toks = _TokenList(batch['unconcat_tokens'])
    toks.rl_plan = plan
    batch['unconcat_tokens'] = toks
    _core.request_rows(plan.req)
    _STATS['planned'] += 1
    return batch


def _call_calc(L, logits, targets, tp_group, with_entropy, with_entropy_grad, chunk_size, keep):
    fn = L.calculate_log_probs_and_entropy
    params = inspect.signature(fn).parameters
    kw = {'with_entropy': with_entropy}
    if 'chunk_size' in params:
        kw['chunk_size'] = chunk_size
    if 'with_entropy_grad' in params:
        kw['with_entropy_grad'] = with_entropy_grad
    if keep is not None:
        if 'log_prob_keep_mask' not in params:
            raise RuntimeError('rollout top-p replay needs log_prob_keep_mask support')
        kw['log_prob_keep_mask'] = keep
    return fn(logits, targets, tp_group, **kw)


def _compact_log_probs_and_entropy(plan, logits, *, args, unconcat_tokens, total_lengths, response_lengths,
                                   with_entropy=False, non_loss_data=True, top_p_token_ids=None,
                                   top_p_token_offsets=None):
    from megatron.core import mpu
    from slime.backends.megatron_utils import loss as L
    assert non_loss_data
    assert logits.dtype == torch.float32, f'{logits.dtype}'
    if plan.req.head is not None:
        return _chunked_log_probs_and_entropy(plan, logits, args=args, total_lengths=total_lengths,
                                              response_lengths=response_lengths, with_entropy=with_entropy,
                                              top_p=top_p_token_ids is not None and top_p_token_offsets is not None)
    logits = logits.squeeze(0)
    temperature = getattr(args, 'rollout_temperature', 1.0)
    if temperature != 1.0:
        logits = logits / temperature
    logits = logits.contiguous()
    rows = plan.req.rows
    n = int(rows.numel())
    targets = plan.targets if n else plan.targets.new_zeros(1)
    keep = None
    if top_p_token_ids is not None and top_p_token_offsets is not None:
        full = L._build_topp_keep_mask(plan.req.seq_len, logits.size(-1), logits.device, top_p_token_ids,
                                       top_p_token_offsets, total_lengths, response_lengths, args.allgather_cp)
        keep = full.index_select(0, rows if n else rows.new_zeros(1))
    with_entropy_grad = with_entropy and getattr(args, 'entropy_coef', 0.0) != 0
    lp, ent = _call_calc(L, logits, targets, mpu.get_tensor_model_parallel_group(), with_entropy,
                         with_entropy_grad, getattr(args, 'log_probs_chunk_size', -1), keep)
    lp = lp.reshape(-1)[:n]
    ent = ent.reshape(-1)[:n] if (with_entropy and ent is not None) else None
    _STATS['compact_logp'] += 1
    return _split_result(plan, lp, ent, args, total_lengths, response_lengths, with_entropy, logits.device)


def _chunked_log_probs_and_entropy(plan, hidden, *, args, total_lengths, response_lengths, with_entropy, top_p):
    """The model deferred its output layer (response_logits.chunked): `hidden` is the selected rows [1, n, h]."""
    from ai_search_plugin.response_logits import chunked
    if top_p:
        raise RuntimeError('response_logits chunked path cannot replay rollout top-p (install() disables it when '
                           'rollout_top_p != 1; set RESPONSE_LOGITS_CHUNK=off)')
    head = plan.req.head
    x = hidden.squeeze(0).to(head.weight.dtype)
    rows = plan.req.rows
    n = int(rows.numel())
    targets = plan.targets if n else plan.targets.new_zeros(1)
    with_entropy_grad = with_entropy and getattr(args, 'entropy_coef', 0.0) != 0
    spec = head.spec(temperature=getattr(args, 'rollout_temperature', 1.0), with_entropy=with_entropy,
                     entropy_grad=with_entropy_grad)
    lp, ent = chunked.lm_head_logprob_entropy(x, head.weight, targets, spec)
    lp = lp[:n]
    ent = ent[:n] if with_entropy else None
    _STATS['chunked_logp'] += 1
    return _split_result(plan, lp, ent, args, total_lengths, response_lengths, with_entropy, hidden.device)


def _split_result(plan, lp, ent, args, total_lengths, response_lengths, with_entropy, device):
    from slime.backends.megatron_utils import loss as L
    counts = plan.counts
    if plan.sel is not None:  # loss mode: zeros at response rows outside the loss mask
        lp = lp.new_zeros(plan.n_resp).index_put((plan.sel,), lp)
        if ent is not None:
            ent = ent.new_zeros(plan.n_resp).index_put((plan.sel,), ent)
    res = {'log_probs': list(lp.split(counts))}
    if with_entropy:
        res['entropy'] = list(ent.split(counts))
    if args.allgather_cp:
        L._allgather_cp_redistribute(res, logits_local_len=plan.req.seq_len, total_lengths=total_lengths,
                                     response_lengths=response_lengths)
    return torch.empty((0,), device=device), res


def _make_logp(orig):
    @functools.wraps(orig)
    def get_log_probs_and_entropy(logits, *args_, **kwargs):
        plan = getattr(kwargs.get('unconcat_tokens'), 'rl_plan', None)
        if (plan is None or args_ or not plan.req.compact or logits.dim() != 3
                or logits.size(1) != plan.req.out_rows):
            if plan is not None:
                _STATS['fallback_logp'] += 1
            return orig(logits, *args_, **kwargs)
        return _compact_log_probs_and_entropy(plan, logits, **kwargs)
    get_log_probs_and_entropy._response_logits = True
    return get_log_probs_and_entropy


def _replace_everywhere(orig, new):
    n = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == 'slime' or name.startswith('slime.')):
            continue
        for attr, val in list(vars(mod).items()):
            if val is orig:
                setattr(mod, attr, new)
                n += 1
    return n


def unsupported_reason(args):
    for key, bad in _UNSUPPORTED_ARGS:
        if args is not None and getattr(args, key, None) == bad:
            return f'{key}={bad!r}'
    return None


def install(args=None):
    """Patch slime + GPTModel. `args`: slime args (from --custom-megatron-init-path); used only to refuse
    configurations whose loss reads the full logits itself. Returns a status dict."""
    if _STATE['installed']:
        return dict(_STATE, stats=dict(_STATS))
    reason = unsupported_reason(args)
    if reason is not None:
        _STATE['reason'] = f'not installed: {reason} reads full logits'
        print(f'[response_logits.slime] {_STATE["reason"]}', file=sys.stderr)
        return dict(_STATE)
    import slime.backends.megatron_utils.loss as L
    import slime.backends.megatron_utils.model as M
    patched = _core.install()
    _STATE['mode'] = rows_mode()
    orig_logp = L.get_log_probs_and_entropy
    if getattr(orig_logp, '_response_logits', False):
        _STATE['installed'] = True
        return dict(_STATE)
    _STATE['orig_logp'] = orig_logp
    _STATE['orig_get_batch'] = M.get_batch
    n_logp = _replace_everywhere(orig_logp, _make_logp(orig_logp))
    M.get_batch = functools.wraps(M.get_batch)(_get_batch)  # only the forward_step callers in model.py
    if args is not None and getattr(args, 'rollout_top_p', 1.0) not in (None, 1.0):
        _STATE.update(chunk_ok=False, chunk_off_reason='rollout_top_p replay needs full-vocab keep masks')
    from ai_search_plugin.response_logits import chunked
    _STATE.update(installed=True, reason=None, patched=patched, logp_refs=n_logp,
                  chunk=os.environ.get('RESPONSE_LOGITS_CHUNK', 'auto'),
                  chunk_mb=os.environ.get('RESPONSE_LOGITS_CHUNK_MB', '2048'), chunk_impl=chunked.impl())
    return {k: v for k, v in _STATE.items() if not k.startswith('orig')}


def uninstall():
    if not _STATE['installed'] or _STATE['orig_logp'] is None:
        return
    import slime.backends.megatron_utils.model as M
    current = sys.modules['slime.backends.megatron_utils.loss'].get_log_probs_and_entropy
    _replace_everywhere(current, _STATE['orig_logp'])
    M.get_batch = _STATE['orig_get_batch']
    _STATE.update(installed=False, orig_logp=None, orig_get_batch=None)


def megatron_init(args=None, *a, **k):
    """slime --custom-megatron-init-path hook."""
    return install(args)


def stats():
    return dict(_STATS)
