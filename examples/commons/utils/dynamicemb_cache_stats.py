"""DynamicEmb cache hit rate debug logging.

Enable by setting the environment variable ``CACHE_DEBUG=1`` **before**
importing this module.  When enabled, a ``register_forward_hook`` is
attached to every ``BatchedDynamicEmbeddingTable`` that has a GPU cache,
logging per-table, per-rank hit/miss statistics after each forward pass.

Usage::

    from commons.utils.dynamicemb_cache_stats import auto_install
    auto_install(model)   # no-op if CACHE_DEBUG != "1"

Output format (one line per table per rank per iteration)::

    [rank0] [CACHE iter=42] table=item unique=12345 hit=11000 miss=1345 hit_rate=89.11%
    [rank0] [CACHE iter=42] table=user_id unique=8000 hit=7500 miss=500 hit_rate=93.75%
"""

import os
from typing import Any, List

import torch
import torch.nn as nn
from commons.utils.dynamicemb_utils import find_dynamicemb_modules


class _CacheDebugHook:
    """Forward hook that reads cache_metrics and prints hit rate stats."""

    def __init__(self, table_names: List[str], cache: Any) -> None:
        self._table_names = table_names
        self._cache = cache
        self._iter = 0

    def __call__(self, module: nn.Module, input: Any, output: Any) -> None:
        metrics = self._cache.cache_metrics
        if metrics is None:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        unique = metrics[0].item()
        hit = metrics[1].item()
        if unique == 0:
            return
        miss = unique - hit
        hit_rate = hit / unique * 100.0
        tables_str = ",".join(self._table_names)
        print(
            f"[rank{rank}] [CACHE iter={self._iter}] "
            f"table={tables_str} "
            f"unique={unique} hit={hit} miss={miss} "
            f"hit_rate={hit_rate:.2f}%",
            flush=True,
        )
        self._iter += 1


def install_cache_debug_hooks(model: nn.Module) -> int:
    """Attach forward hooks to all DynamicEmb modules with caching enabled.

    Args:
        model: The top-level model (may be wrapped by DMP/DDP).

    Returns:
        Number of hooks installed.
    """
    count = 0
    modules = find_dynamicemb_modules(model)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        print(f"[CACHE_DEBUG] Found {len(modules)} DynamicEmb module(s)", flush=True)
    for m in modules:
        has_cache = hasattr(m, "cache") and m.cache is not None
        if rank == 0:
            print(
                f"[CACHE_DEBUG]   tables={m.table_names} has_cache={has_cache}",
                flush=True,
            )
        if not has_cache:
            continue
        m.set_record_cache_metrics(True)
        hook = _CacheDebugHook(m.table_names, m.cache)
        m.register_forward_hook(hook)
        count += 1
    if rank == 0:
        print(f"[CACHE_DEBUG] Installed {count} hook(s)", flush=True)
    return count


def auto_install(model: nn.Module) -> int:
    """Install cache debug hooks if ``CACHE_DEBUG=1`` is set. No-op otherwise."""
    if os.environ.get("CACHE_DEBUG", "0") != "1":
        return 0
    return install_cache_debug_hooks(model)
