"""Framework adapters for response_logits (slime / verl-megatron).

Both frameworks take full [1, T, V/TP] logits from the model and compute log-probs / entropy themselves. The
adapters make the model return logits only for the rows those helpers actually read (response tokens, or
loss-mask tokens with RESPONSE_LOGITS_ROWS=loss) and map the compact result back to the exact shapes the
framework expects, so its loss code is unchanged. Off until `install()`.

    from ai_search_plugin.response_logits.adapters import slime as rl_slime; rl_slime.install()
    from ai_search_plugin.response_logits.adapters import verl as rl_verl;   rl_verl.install()
    # or: ai_search_plugin.response_logits.adapters.install_available()  (whatever of the two is importable)
"""
import os


def rows_mode():
    """'response' (default): every response token, values identical to the stock path.
    'loss': only loss-mask tokens; response tokens with loss_mask == 0 (tool output etc.) get log-prob/entropy 0."""
    mode = os.environ.get('RESPONSE_LOGITS_ROWS', 'response').strip().lower()
    if mode not in ('response', 'loss'):
        raise ValueError(f'RESPONSE_LOGITS_ROWS must be response|loss, got {mode!r}')
    return mode


def install_available(args=None):
    """Install the adapter of every framework present in this process. Returns {name: result}.
    verl is patched lazily (right after verl.models.mcore is imported), so this is safe to call from
    verl/__init__ through VERL_USE_EXTERNAL_MODULES."""
    import importlib.util
    out = {}
    try:
        importlib.import_module('slime.backends.megatron_utils.loss')
        has_slime = True
    except Exception:
        has_slime = False
    if has_slime:
        out['slime'] = importlib.import_module(f'{__name__}.slime').install(args)
    if importlib.util.find_spec('verl') is not None:
        out['verl'] = importlib.import_module(f'{__name__}.verl').install_lazy()
    return out
