# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example: register `HSTUBatch` as a serializable pytree node for torch.export/AOTInductor.

Why this exists:
- torch.export package serialization needs a stable `serialized_type_name` for custom pytree nodes.
- Without it, AOTInductor packaging can fail with:
  "No registered serialization name for <class '...HSTUBatch'> found"

How to use:
1) Import and call `register_hstu_export_pytrees()` once before `torch.export.export(...)`.
2) Then export/package as usual.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from commons.datasets.hstu_batch import HSTUBatch
from torch.utils import _pytree as pytree
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def _flatten_hstu_batch(batch: HSTUBatch) -> Tuple[List[Any], Dict[str, Any]]:
    """Flatten `HSTUBatch` into (children, context).

    children: runtime values that may contain tensors / tensor-like objects.
    context : JSON/pickle-friendly metadata needed to reconstruct the object.
    """
    # NOTE:
    # - labels are intentionally excluded for export in this project.
    # - num_candidates can be a Tensor or None.
    children: List[Any] = [batch.features, batch.num_candidates]

    context: Dict[str, Any] = {
        # BaseBatch fields
        "batch_size": int(batch.batch_size),
        "feature_to_max_seqlen": dict(batch.feature_to_max_seqlen),
        "contextual_feature_names": list(batch.contextual_feature_names),
        "actual_batch_size": (
            int(batch.actual_batch_size)
            if batch.actual_batch_size is not None
            else None
        ),
        # HSTUBatch fields
        "item_feature_name": batch.item_feature_name,
        "action_feature_name": batch.action_feature_name,
        "max_num_candidates": int(batch.max_num_candidates),
    }
    return children, context


def _flatten_hstu_batch_with_keys(
    batch: HSTUBatch,
) -> Tuple[List[Tuple[pytree.KeyEntry, Any]], Dict[str, Any]]:
    children, context = _flatten_hstu_batch(batch)
    keyed_children: List[Tuple[pytree.KeyEntry, Any]] = [
        (pytree.GetAttrKey("features"), children[0]),
        (pytree.GetAttrKey("num_candidates"), children[1]),
    ]
    return keyed_children, context


def _flatten_jagged_tensor(jt: JaggedTensor) -> Tuple[List[Any], Dict[str, Any]]:
    """Flatten `JaggedTensor` into (children, context)."""
    # NOTE:
    # - weights are intentionally excluded for export in this project.
    # - use a canonical representation (values + lengths only), so flattening
    #   is stable and not affected by lazy offsets materialization.
    children: List[Any] = [
        jt.values(),
        jt.lengths(),
    ]
    context: Dict[str, Any] = {}
    return children, context


def _flatten_jagged_tensor_with_keys(
    jt: JaggedTensor,
) -> Tuple[List[Tuple[pytree.KeyEntry, Any]], Dict[str, Any]]:
    children, context = _flatten_jagged_tensor(jt)
    keyed_children: List[Tuple[pytree.KeyEntry, Any]] = [
        (pytree.GetAttrKey("values"), children[0]),
        (pytree.GetAttrKey("lengths"), children[1]),
    ]
    return keyed_children, context


def _unflatten_jagged_tensor(
    children: List[Any], context: Dict[str, Any]
) -> JaggedTensor:
    """Reconstruct `JaggedTensor` from context + children."""
    del context
    values, lengths = children
    return JaggedTensor(
        values=values,
        lengths=lengths,
    )


def _flatten_keyed_jagged_tensor(
    kjt: KeyedJaggedTensor,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Flatten `KeyedJaggedTensor` into (children, context)."""
    children: List[Any] = [
        kjt.values(),
        kjt.lengths(),
    ]
    context: Dict[str, Any] = {
        "keys": list(kjt.keys()),
    }
    return children, context


def _flatten_keyed_jagged_tensor_with_keys(
    kjt: KeyedJaggedTensor,
) -> Tuple[List[Tuple[pytree.KeyEntry, Any]], Dict[str, Any]]:
    children, context = _flatten_keyed_jagged_tensor(kjt)
    keyed_children: List[Tuple[pytree.KeyEntry, Any]] = [
        (pytree.GetAttrKey("values"), children[0]),
        (pytree.GetAttrKey("lengths"), children[1]),
    ]
    return keyed_children, context


def _unflatten_keyed_jagged_tensor(
    children: List[Any], context: Dict[str, Any]
) -> KeyedJaggedTensor:
    """Reconstruct `KeyedJaggedTensor` from context + children."""
    values, lengths = children
    return KeyedJaggedTensor.from_lengths_sync(
        keys=context["keys"],
        values=values,
        lengths=lengths,
    )


def _unflatten_hstu_batch(children: List[Any], context: Dict[str, Any]) -> HSTUBatch:
    """Reconstruct `HSTUBatch` from context + children."""
    features, num_candidates = children

    return HSTUBatch(
        features=features,
        batch_size=context["batch_size"],
        feature_to_max_seqlen=context["feature_to_max_seqlen"],
        contextual_feature_names=context["contextual_feature_names"],
        # labels=labels,
        actual_batch_size=context["actual_batch_size"],
        item_feature_name=context["item_feature_name"],
        action_feature_name=context["action_feature_name"],
        max_num_candidates=context["max_num_candidates"],
        num_candidates=num_candidates,
    )


def _to_dumpable_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Convert context to a dumpable object for export package serialization.

    This can be identity because `context` is already made of dumpable python types.
    """
    return context


def _from_dumpable_context(dumpable_context: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of `_to_dumpable_context`."""
    return dumpable_context


_REGISTERED = False


def register_hstu_export_pytrees() -> None:
    """Register HSTU export pytrees (HSTUBatch + jagged tensor types).

    Call this exactly once before torch.export.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        pytree.register_pytree_node(
            JaggedTensor,
            _flatten_jagged_tensor,
            _unflatten_jagged_tensor,
            flatten_with_keys_fn=_flatten_jagged_tensor_with_keys,
            serialized_type_name="torchrec.sparse.jagged_tensor.JaggedTensor",
            to_dumpable_context=_to_dumpable_context,
            from_dumpable_context=_from_dumpable_context,
        )
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise

    try:
        pytree.register_pytree_node(
            KeyedJaggedTensor,
            _flatten_keyed_jagged_tensor,
            _unflatten_keyed_jagged_tensor,
            flatten_with_keys_fn=_flatten_keyed_jagged_tensor_with_keys,
            serialized_type_name="torchrec.sparse.jagged_tensor.KeyedJaggedTensor",
            to_dumpable_context=_to_dumpable_context,
            from_dumpable_context=_from_dumpable_context,
        )
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise

    try:
        pytree.register_pytree_node(
            HSTUBatch,
            _flatten_hstu_batch,
            _unflatten_hstu_batch,
            flatten_with_keys_fn=_flatten_hstu_batch_with_keys,
            serialized_type_name="commons.datasets.hstu_batch.HSTUBatch",
            to_dumpable_context=_to_dumpable_context,
            from_dumpable_context=_from_dumpable_context,
        )
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise

    _REGISTERED = True


# Example snippet (call site):
#
# from register_hstubatch_pytree_example import register_hstu_export_pytrees
# register_hstu_export_pytrees()
# exported_program = torch.export.export(model, args=(batch, embeddings), dynamic_shapes=...)
# torch._inductor.aoti_compile_and_package(exported_program, package_path="dense_module.pt2")
