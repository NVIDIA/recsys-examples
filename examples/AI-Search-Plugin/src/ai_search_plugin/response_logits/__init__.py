"""Response-only output layer for RL policy training (Megatron GPTModel / mcore-bridge GPTModel).

In GRPO/PPO only response tokens carry loss (prompt tokens are ~85-90% of a 40k-token agentic sample), yet the
stock post-process computes vocab logits (hidden x vocab GEMM + vocab-parallel cross entropy over fp32
logits) for every token. This component computes logits and per-token cross entropy only at loss-mask positions
and scatters zeros elsewhere, so the per-token loss tensor keeps its [b, s] shape and any loss function that
multiplies by the mask is unchanged.

Usage (any Megatron-based trainer: slime / verl-megatron / AReaL-megatron / ROLL):
    from ai_search_plugin import response_logits
    response_logits.install()                      # patches GPTModel._postprocess (mcore-bridge and megatron.core)
    with response_logits.loss_mask(mask):          # mask: [b, s_local] (same layout as labels), >0 = loss token
        out = model(input_ids=..., labels=labels, ...)

TP/SP: hidden is all-gathered over TP (as the output layer does anyway), rows are selected identically on all TP
ranks (same CP chunk), and the column-parallel output layer runs with sequence_parallel temporarily off, i.e.
copy-to-TP-region (backward all-reduce) instead of SP all-gather (backward reduce-scatter); the gather uses
tensor_parallel_output_grad=False (backward = split) so gradients are not double counted.
A rank whose local chunk has no loss token still runs the layer on one dummy row with zero weight, so every
parameter receives a (zero) gradient and collectives stay matched.

Frameworks that take full logits from the model (slime, verl-megatron) use the adapters in
`response_logits.adapters` instead: the adapter registers the rows it needs with `request_rows()` right before
the model forward, the patched post-process returns logits only for those rows ([1, n, V/TP], labels=None path),
and the adapter's log-prob / entropy hook maps the compact result back to the framework's own layout.
"""
import contextlib
import contextvars
import inspect

import torch

_MASK = contextvars.ContextVar('response_logits_mask', default=None)
_STATS = {'calls': 0, 'selected_tokens': 0, 'total_tokens': 0, 'fallback_calls': 0,
          'row_calls': 0, 'row_selected': 0, 'row_total': 0, 'row_rejected': 0, 'row_chunked': 0,
          'chunk_ineligible': 0}
_PENDING = [None]  # RowRequest registered by a framework adapter for the next post-process call


class RowRequest:
    """Rows of the (TP-gathered) post-process hidden states [s, 1, h] whose logits an adapter needs.

    rows: LongTensor [n] of local row indices (n may be 0); seq_len: expected local sequence length s
    (after the SP gather); input_ids: optional tensor the forward must receive (identity check, so a stale
    request is never applied to a different micro-batch). After the forward, `compact` tells whether the model
    really returned compact logits ([1, max(n, 1), V/TP]; one dummy row when n == 0).
    """
    __slots__ = ('rows', 'seq_len', 'input_ids', 'compact', 'out_rows', 'allow_chunk', 'head')

    def __init__(self, rows, seq_len, input_ids=None, allow_chunk=False):
        self.rows = rows
        self.seq_len = int(seq_len)
        self.input_ids = input_ids
        self.compact = False
        self.out_rows = 0
        # allow_chunk: the adapter can consume a deferred output layer (response_logits.chunked). When the
        # post-process takes that route, `head` is set and the model returns the selected hidden rows
        # [1, max(n, 1), h] instead of logits; the adapter then runs chunked.lm_head_logprob_entropy(head...).
        self.allow_chunk = bool(allow_chunk)
        self.head = None


class DeferredHead:
    """What the adapter needs to run the output layer itself (chunked): weight shard + TP layout + wgrad mode."""
    __slots__ = ('weight', 'tp_group', 'vstart', 'chunk', 'grad_accum_fusion', 'allreduce_dgrad')

    def __init__(self, weight, tp_group, vstart, chunk, grad_accum_fusion, allreduce_dgrad):
        self.weight, self.tp_group, self.vstart, self.chunk = weight, tp_group, int(vstart), int(chunk)
        self.grad_accum_fusion, self.allreduce_dgrad = bool(grad_accum_fusion), bool(allreduce_dgrad)

    def spec(self, **kw):
        from . import chunked
        return chunked.Spec(tp_group=self.tp_group, vstart=self.vstart, chunk=self.chunk,
                            grad_accum_fusion=self.grad_accum_fusion, allreduce_dgrad=self.allreduce_dgrad, **kw)


def request_rows(req):
    """Register `req` for the next GPTModel post-process call (consumed by the first post-process stage call)."""
    _PENDING[0] = req
    return req


def clear_rows(req=None):
    if req is None or _PENDING[0] is req:
        _PENDING[0] = None



@contextlib.contextmanager
def loss_mask(mask):
    token = _MASK.set(mask)
    try:
        yield
    finally:
        _MASK.reset(token)


def stats():
    return dict(_STATS)


def _selected_postprocess(self, hidden_states, labels, mask):
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
    layer = self.output_layer
    tp_group = getattr(self, 'pg_collection', None)
    tp_group = getattr(tp_group, 'tp', None)
    sp = bool(getattr(layer, 'sequence_parallel', False))
    if sp:
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=tp_group)
    s, b, h = hidden_states.shape
    assert labels.shape == (b, s) and mask.shape == (b, s), (labels.shape, mask.shape, hidden_states.shape)
    flat = (mask.reshape(b, s).transpose(0, 1).reshape(-1) > 0)  # [s*b] in [s, b] order
    idx = flat.nonzero().squeeze(1)
    dummy = idx.numel() == 0
    if dummy:
        idx = torch.zeros(1, dtype=torch.long, device=flat.device)
        _STATS['fallback_calls'] += 1
    sel = hidden_states.reshape(s * b, h).index_select(0, idx).unsqueeze(1)  # [n, 1, h]
    output_weight = self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else None
    sel_labels = labels.transpose(0, 1).reshape(-1).index_select(0, idx).unsqueeze(0)  # [1, n]
    head = _deferred_head(self, layer, int(idx.numel()), tp_group, lambda name, default=None: default)
    if head is not None:  # chunked CE: per-token loss = -log p(label), never more than one [chunk, V/TP] block
        from . import chunked
        lp, _ = chunked.lm_head_logprob_entropy(sel.squeeze(1), head.weight, sel_labels.reshape(-1),
                                                head.spec(with_entropy=False, entropy_grad=False))
        sel_loss = (-lp).unsqueeze(0)
        _STATS['row_chunked'] += 1
    else:
        try:
            layer.sequence_parallel = False
            fwd = getattr(self, '_forward_output_layer', None)  # mcore-bridge; megatron.core calls output_layer
            logits = (fwd(sel, weight=output_weight, runtime_gather_output=None) if fwd is not None
                      else layer(sel, weight=output_weight, runtime_gather_output=None)[0])
        finally:
            layer.sequence_parallel = sp
        sel_loss = self.compute_language_model_loss(sel_labels, logits)  # [1, n]
    if dummy:
        sel_loss = sel_loss * 0.0
    out = sel_loss.new_zeros(s * b)
    out = out.index_put((idx,), sel_loss.reshape(-1), accumulate=False)
    _STATS['calls'] += 1
    _STATS['selected_tokens'] += 0 if dummy else int(idx.numel())
    _STATS['total_tokens'] += s * b
    return out.reshape(s, b).transpose(0, 1).contiguous()


def _bound_getter(sig, self, hidden_states, args, kwargs):
    try:
        ba = sig.bind(self, hidden_states, *args, **kwargs)
    except TypeError:
        return None
    extra = ba.arguments.get('kwargs') or {}

    def get(name, default=None):
        if name in ba.arguments:
            return ba.arguments[name]
        return extra.get(name, default)
    return get


def _rows_postprocess(self, original, hidden_states, args, kwargs, req, get):
    """Run the stock post-process on the requested rows only (labels=None path -> logits [1, n, V/TP])."""
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
    layer = self.output_layer
    tp_group = getattr(getattr(self, 'pg_collection', None), 'tp', None)
    sp = bool(getattr(layer, 'sequence_parallel', False))
    if sp:
        # backward = split (no reduce): every TP rank selects the same rows, and the output layer without SP
        # all-reduces the input grad over TP, so each rank already holds the full grad of its own slice.
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=False, group=tp_group)
    s, b, h = hidden_states.shape
    rows = req.rows
    n = int(rows.numel())
    idx = rows if n else torch.zeros(1, dtype=torch.long, device=hidden_states.device)
    sel = hidden_states.reshape(s * b, h).index_select(0, idx).unsqueeze(1)  # [n, 1, h]
    head = _deferred_head(self, layer, int(idx.numel()), tp_group, get) if req.allow_chunk else None
    if head is not None:
        # chunked route: the adapter runs output layer + log-prob/entropy block by block (chunked.py); the
        # model returns the selected rows. Backward of the TP all-reduce of dX happens inside that op.
        req.head = head
        out = sel.transpose(0, 1).contiguous()  # [1, n, h] (Float16Module casts it to fp32, exact round trip)
        _STATS['row_chunked'] += 1
    else:
        try:
            layer.sequence_parallel = False
            out = original(self, sel, *args, **kwargs)
        finally:
            layer.sequence_parallel = sp
    req.compact = True
    req.out_rows = int(idx.numel())
    _STATS['row_calls'] += 1
    _STATS['row_selected'] += n
    _STATS['row_total'] += s * b
    return out


_SAID = set()


def _say_once(key, msg):
    if key not in _SAID:
        _SAID.add(key)
        import sys
        print(msg, file=sys.stderr, flush=True)


def _deferred_head(self, layer, n_rows, tp_group, get):
    """DeferredHead when the output layer is a plain Megatron ColumnParallelLinear we can reproduce exactly
    (vocab-parallel output, no bias, no MuP scaling, no deferred-embedding wgrad / GTP / FSDP weights) and the
    chunk policy wants chunking for n_rows rows; else None (stock compact logits)."""
    from . import chunked
    if chunked.mode()[0] == 'off':
        return None
    w = self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else layer.weight
    cfg = self.config
    fol = getattr(type(self), '_forward_output_layer', None)  # mcore-bridge hook: a subclass may change logits
    ok = (fol is None or getattr(fol, '__module__', '') == 'mcore_bridge.model.gpt_model') and (w is not None and w.dim() == 2 and type(layer).__name__ == 'ColumnParallelLinear'
          and getattr(layer, 'bias', None) is None and not getattr(layer, 'gather_output', False)
          and not get('runtime_gather_output') and not getattr(cfg, 'use_mup', False)
          and not getattr(cfg, 'defer_embedding_wgrad_compute', False)
          and getattr(layer, 'gtp_remat_size', 1) in (None, 1) and not hasattr(w, '__fsdp_param__')
          and getattr(layer, 'output_dtype', None) in (None, w.dtype)
          and not getattr(layer, 'explicit_expert_comm', False)
          and w.dtype in (torch.bfloat16, torch.float16) and w.is_cuda)
    if not ok:
        _STATS['chunk_ineligible'] += 1
        _say_once('inelig', f'[response_logits] chunked lm-head not applicable to {type(self).__name__}.'
                            f'{type(layer).__name__} (bias/MuP/deferred wgrad/custom output layer): compact logits')
        return None
    rows = chunked.decide(n_rows, w.size(0))
    if not rows:
        return None
    _say_once('on', f'[response_logits] chunked lm-head on: {rows} rows/block, V/TP={w.size(0)}, '
                    f'impl={chunked.impl()}, first micro-batch {n_rows} rows (RESPONSE_LOGITS_CHUNK='
                    f'{chunked.mode()[0]}, budget {chunked.budget_bytes() >> 20} MiB)')
    if tp_group is None:
        tp_group = getattr(layer, 'tp_group', None)
    if tp_group is None:
        from megatron.core import parallel_state
        tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_rank = torch.distributed.get_rank(group=tp_group) if torch.distributed.is_initialized() else 0
    tp_size = torch.distributed.get_world_size(group=tp_group) if torch.distributed.is_initialized() else 1
    return DeferredHead(w, tp_group, tp_rank * w.size(0), rows,
                        grad_accum_fusion=getattr(layer, 'gradient_accumulation_fusion', False),
                        allreduce_dgrad=tp_size > 1 and not getattr(layer, 'disable_grad_reduce', False))


def _row_request_ok(self, hidden_states, req, get):
    if get('labels') is not None or get('inference_context') is not None or get('inference_params') is not None:
        return False
    if get('mtp_in_postprocess') or get('output_processor') is not None:
        return False
    if getattr(self.config, 'task_type', 'causal_lm') != 'causal_lm':
        return False
    layer = getattr(self, 'output_layer', None)
    w = getattr(layer, 'weight', None)
    if w is None and getattr(self, 'share_embeddings_and_output_weights', False):
        w = self.shared_embedding_or_output_weight()
    if w is None or w.shape[0] <= 1:  # value head (critic) or unknown layer: keep full output
        return False
    if hidden_states.dim() != 3 or hidden_states.shape[1] != 1:
        return False
    s = hidden_states.shape[0]
    if getattr(layer, 'sequence_parallel', False):
        from megatron.core import parallel_state
        tp = getattr(getattr(self, 'pg_collection', None), 'tp', None)
        s *= tp.size() if tp is not None else parallel_state.get_tensor_model_parallel_world_size()
    if s != req.seq_len:
        return False
    if req.input_ids is not None:
        ids = get('input_ids')
        if ids is None or ids.data_ptr() != req.input_ids.data_ptr() or ids.shape != req.input_ids.shape:
            return False
    return True


def _wrap(cls):
    if getattr(cls._postprocess, '_response_logits', False):
        return
    original = cls._postprocess
    sig = inspect.signature(original)

    def _postprocess(self, hidden_states, *args, **kwargs):
        get = None
        if self.post_process and _PENDING[0] is not None:
            req, _PENDING[0] = _PENDING[0], None
            get = _bound_getter(sig, self, hidden_states, args, kwargs)
            if get is not None and _row_request_ok(self, hidden_states, req, get):
                return _rows_postprocess(self, original, hidden_states, args, kwargs, req, get)
            _STATS['row_rejected'] += 1
        mask = _MASK.get()
        if mask is None or not self.post_process or not self.training or getattr(self, 'mtp_process', False):
            return original(self, hidden_states, *args, **kwargs)
        get = get or _bound_getter(sig, self, hidden_states, args, kwargs)
        labels = get('labels') if get is not None else None
        if (labels is None or getattr(self.config, 'task_type', 'causal_lm') != 'causal_lm'
                or get('inference_context') is not None or get('inference_params') is not None):
            return original(self, hidden_states, *args, **kwargs)
        return _selected_postprocess(self, hidden_states, labels, mask)

    _postprocess._response_logits = True
    _postprocess.__wrapped__ = original
    cls._postprocess = _postprocess


def install():
    """Patch every importable GPTModel class. Returns the list of patched classes."""
    patched = []
    for mod, name in (('mcore_bridge.model.gpt_model', 'GPTModel'), ('megatron.core.models.gpt.gpt_model', 'GPTModel')):
        try:
            cls = getattr(__import__(mod, fromlist=[name]), name)
        except Exception:
            continue
        _wrap(cls)
        patched.append(f'{mod}.{name}')
    return patched
