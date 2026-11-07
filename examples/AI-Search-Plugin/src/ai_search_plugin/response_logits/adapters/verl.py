"""verl (Megatron engine) adapter: output layer only at response rows.

Stock verl (v1 engine, workers/engine/megatron/transformer_impl.py MegatronEngineWithLMHead.forward_step):
forward_fn = verl.models.mcore.get_mcore_engine_forward_fn(hf_config) (= model_forward.gptmodel_forward_model_engine)
packs the micro-batch to THD (CP zigzag/contiguous shard, quant/bucket padding), calls the model with labels=None,
then calls logits_processor(logits [1, T, V/TP], label=rolled labels [1, T], temperature [1, T]) =
_lm_head_logits_processor -> vocab_parallel_entropy / vocab_parallel_log_probs_from_logits, and unpacks every
output back to nested [b, seq_len] (the loss later slices the response part with no_padding_2_padding).

With the adapter the forward_fn is wrapped:
  1. Build the full-sequence response mask exactly like verl's MTP path does for loss_mask
     (_build_mtp_loss_mask_nested: prompt zeros + response mask) and pack it with the same preprocess_thd_engine
     call verl uses for labels (need_roll=True, same CP layout / padding) -> the logit rows that are read.
  2. response_logits.request_rows(rows): the patched GPTModel._postprocess runs the output layer on those rows only.
  3. logits_processor is wrapped: it runs verl's own processor on the compact logits with label/temperature
     index-selected to the same rows, then scatters every output ([1, n]) into zeros [1, T]. Shapes, keys and
     the unpacking are the stock ones; values at the selected rows are the stock values, other rows are 0
     (prompt rows are never read by the loss).
Falls back to the stock path for value models, MTP training, BSHD format, top-k distillation, fused kernels
(use_fused_kernels: separate forward that never materialises logits), and whenever the model did not return
compact logits.

Enable (every verl worker process): VERL_USE_EXTERNAL_MODULES=ai_search_plugin.response_logits.adapters.verl,
or through the plugin entry (VERL_USE_EXTERNAL_MODULES=ai_search_plugin.verl_plugin with component response_logits).
Tested against verl main of 2026-10-23 (v1 engine) with Megatron-Core 0.20.
"""
import functools
import inspect
import sys

import torch

from ai_search_plugin import response_logits as _core

from . import rows_mode

_STATE = {'installed': False, 'mode': None}
_STATS = {'planned': 0, 'compact': 0, 'fallback': 0, 'chunked': 0}
_WRAPPED = {}


def _plan_rows(model, input_ids, lpa, local_cp_size, pad_to_length_bucket, cp_layout, allow_chunk=False):
    from verl.models.mcore.model_forward import _build_mtp_loss_mask_nested
    from verl.models.mcore.util import get_thd_quant_align_size, preprocess_thd_engine
    from verl.utils.megatron_utils import unwrap_model
    lens = input_ids.offsets().diff().tolist()
    loss_mask = lpa['loss_mask']
    ram = lpa.get('response_attention_mask')
    if _STATE['mode'] == 'response':
        if loss_mask.is_nested:
            src = torch.nested.nested_tensor_from_jagged(torch.ones_like(loss_mask.values()), offsets=loss_mask.offsets())
        else:
            src = ram
    else:
        src = loss_mask
    full = _build_mtp_loss_mask_nested(src, lens, ram)
    packed = preprocess_thd_engine(full, pre_process=True, need_roll=True,
                                   quant_align_size=get_thd_quant_align_size(unwrap_model(model).config),
                                   local_cp_size=local_cp_size, pad_to_length_bucket=pad_to_length_bucket,
                                   cp_layout=cp_layout)[0]
    assert packed.dim() == 2 and packed.size(0) == 1, packed.shape
    rows = packed[0].nonzero().squeeze(1)
    return _core.RowRequest(rows, packed.size(1), allow_chunk=allow_chunk)


def _chunked_processor(processor, req, hidden, label, temperature):
    """verl's _lm_head_logits_processor semantics (temperature <= 0 -> 1e-8, entropy, log_probs) on the deferred
    output layer: hidden = selected rows [1, n, h]; label / temperature already index-selected [1, n]."""
    from ai_search_plugin.response_logits import chunked
    kws = getattr(processor, 'keywords', None) or {}
    head = req.head
    temp = temperature.reshape(-1).float()
    temp = torch.where(temp <= 0, torch.full_like(temp, 1e-8), temp)
    calc_entropy = bool(kws.get('calculate_entropy', False))
    spec = head.spec(with_entropy=calc_entropy, entropy_grad=calc_entropy)
    lp, ent = chunked.lm_head_logprob_entropy(hidden.reshape(-1, hidden.size(-1)).to(head.weight.dtype),
                                              head.weight, label.reshape(-1), spec, temp_vec=temp)
    out = {}
    if calc_entropy:
        out['entropy'] = ent.unsqueeze(0)
    out['log_probs'] = lp.unsqueeze(0)
    return out


def _chunk_ok(processor):
    kws = getattr(processor, 'keywords', None) or {}
    return not (kws.get('calculate_sum_pi_squared') or kws.get('distillation_use_topk') or kws.get('distillation_only'))


def _wrap_processor(processor, req):
    def logits_processor(logits, label, temperature, **kw):
        if not req.compact or logits.size(1) != req.out_rows:
            _STATS['fallback'] += 1
            return processor(logits, label, temperature, **kw)
        rows = req.rows
        n = int(rows.numel())
        idx = rows if n else rows.new_zeros(1)
        T = label.size(1)
        if req.head is not None:
            out = _chunked_processor(processor, req, logits, label.index_select(1, idx),
                                     temperature.index_select(1, idx))
            _STATS['chunked'] += 1
        else:
            out = processor(logits, label.index_select(1, idx), temperature.index_select(1, idx), **kw)
        res = {}
        for k, v in out.items():
            full = v.new_zeros((v.size(0), T) + tuple(v.shape[2:]))
            if n:
                full = full.index_copy(1, idx, v)
            else:
                full = full + 0 * v.sum()  # keep the output layer in the graph (zero grad)
            res[k] = full
        _STATS['compact'] += 1
        return res
    return logits_processor


def _wrap_forward_fn(fn):
    if getattr(fn, '_response_logits', False):
        return fn
    if fn in _WRAPPED:
        return _WRAPPED[fn]
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def forward_fn(*args, **kwargs):
        try:
            ba = sig.bind(*args, **kwargs)
            ba.apply_defaults()
        except TypeError:
            return fn(*args, **kwargs)
        a = ba.arguments
        processor, lpa = a.get('logits_processor'), a.get('logits_processor_args')
        from verl.utils.megatron_utils import unwrap_model
        model = a['model']
        kws = getattr(processor, 'keywords', None) or {}
        if (processor is None or not lpa or lpa.get('loss_mask') is None or a.get('value_model')
                or a.get('mtp_enable_train') or a.get('data_format', 'thd') != 'thd'
                or kws.get('distillation_use_topk') or not unwrap_model(model).post_process
                or not getattr(a['input_ids'], 'is_nested', False)):
            return fn(*args, **kwargs)
        try:
            req = _plan_rows(model, a['input_ids'], lpa, a.get('local_cp_size'), a.get('pad_to_length_bucket'),
                             a.get('cp_layout', 'zigzag'), allow_chunk=_chunk_ok(processor))
        except Exception as e:  # never break training: stock path
            print(f'[response_logits.verl] plan failed, stock path: {e!r}', file=sys.stderr)
            return fn(*args, **kwargs)
        a['logits_processor'] = _wrap_processor(processor, req)
        _STATS['planned'] += 1
        _core.request_rows(req)
        try:
            return fn(*ba.args, **ba.kwargs)
        finally:
            _core.clear_rows(req)

    forward_fn._response_logits = True
    _WRAPPED[fn] = forward_fn
    return forward_fn


def _wrap_getter(getter):
    if getattr(getter, '_response_logits', False):
        return getter

    @functools.wraps(getter)
    def get_mcore_engine_forward_fn(*args, **kwargs):
        return _wrap_forward_fn(getter(*args, **kwargs))
    get_mcore_engine_forward_fn._response_logits = True
    return get_mcore_engine_forward_fn


def install():
    """Patch verl's engine forward-fn getter + GPTModel. Returns a status dict."""
    if _STATE['installed']:
        return dict(_STATE, stats=dict(_STATS))
    import verl.models.mcore as vm
    import verl.models.mcore.registry as reg
    patched = _core.install()
    _STATE['mode'] = rows_mode()
    wrapped = _wrap_getter(reg.get_mcore_engine_forward_fn)
    reg.get_mcore_engine_forward_fn = wrapped
    vm.get_mcore_engine_forward_fn = wrapped
    _STATE.update(installed=True, patched=patched)
    return dict(_STATE)


def stats():
    return dict(_STATS)


class _PostImportHook:
    """sys.meta_path finder that runs `fn` right after module `name` has been executed (used when the adapter is
    imported from verl/__init__ via VERL_USE_EXTERNAL_MODULES: importing verl.models.mcore from inside verl's own
    __init__ can deadlock, so patching waits until verl imports it itself)."""

    def __init__(self, name, fn):
        self.name, self.fn, self.busy = name, fn, False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name or self.busy:
            return None
        import importlib.util
        self.busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self.busy = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return None
        orig_exec, hook = spec.loader.exec_module, self

        def exec_module(module):
            orig_exec(module)
            if hook in sys.meta_path:
                sys.meta_path.remove(hook)
            hook.fn()
        spec.loader.exec_module = exec_module
        return spec


def install_lazy():
    """install() now if verl.models.mcore is already imported, else right after verl imports it."""
    if 'verl.models.mcore' in sys.modules and not getattr(
            getattr(sys.modules['verl.models.mcore'], '__spec__', None), '_initializing', False):
        return install()
    if not any(isinstance(f, _PostImportHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportHook('verl.models.mcore', install))
    return {'installed': 'deferred until verl.models.mcore is imported'}


if __name__ != '__main__':
    import os
    if os.environ.get('RESPONSE_LOGITS_AUTO', '1') == '1' and any(
            m.strip() == __name__ for m in os.environ.get('VERL_USE_EXTERNAL_MODULES', '').split(',')):
        install_lazy()  # imported by verl/__init__ through VERL_USE_EXTERNAL_MODULES
