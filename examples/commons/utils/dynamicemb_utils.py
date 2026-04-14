"""Utilities for finding DynamicEmb modules inside wrapped models.

TorchRec's ShardedEmbeddingCollection stores ``_lookups`` as a plain Python
list (not ``nn.ModuleList``), so ``nn.Module.modules()`` cannot traverse into
it.  This module provides a recursive search that follows the same attribute
chain as dynamicemb's ``check_emb_collection_modules``.
"""

from typing import Any, List, Set

import torch.nn as nn


def find_dynamicemb_modules(module: Any) -> List[Any]:
    """Recursively find DynamicEmb modules (BatchedDynamicEmbeddingTablesV2) by duck typing.

    Traverses ``_lookups``, ``_emb_modules``, ``_emb_module`` attributes
    (used internally by TorchRec/DynamicEmb) as well as standard
    ``nn.Module.children()`` (for DMP/DDP/Float16Module wrappers).

    A module is considered a DynamicEmb module if it has both
    ``table_names`` and ``set_record_cache_metrics`` attributes.

    Args:
        module: The top-level model (may be wrapped by DMP/DDP).

    Returns:
        List of unique DynamicEmb modules found.
    """
    results: List[Any] = []
    visited: Set[int] = set()
    _find_recursive(module, results, visited)
    return results


def _find_recursive(module: Any, results: List[Any], visited: Set[int]) -> None:
    mid = id(module)
    if mid in visited:
        return
    visited.add(mid)

    if hasattr(module, "table_names") and hasattr(module, "set_record_cache_metrics"):
        results.append(module)
        return

    # Follow TorchRec/DynamicEmb internal attributes (plain list, not nn.ModuleList)
    for attr in ("_lookups", "_emb_modules", "_emb_module"):
        child = getattr(module, attr, None)
        if child is None:
            continue
        if isinstance(child, (list, nn.ModuleList)):
            for item in child:
                _find_recursive(item, results, visited)
        else:
            _find_recursive(child, results, visited)

    # Also recurse into nn.Module children (for DMP/DDP/Float16Module wrappers)
    if isinstance(module, nn.Module):
        for child in module.children():
            _find_recursive(child, results, visited)
