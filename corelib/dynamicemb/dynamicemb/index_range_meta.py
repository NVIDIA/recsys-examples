# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Optional

import torch


def _validate_1d(name: str, t: torch.Tensor) -> None:
    if t.dim() != 1:
        raise RuntimeError(
            f"INFERENCE_EMB index-range operators expect 1D {name}, got dim={t.dim()}"
        )


def _get_table_range_fake(
    offsets: torch.Tensor,
    feature_offsets: torch.Tensor,
):
    _validate_1d("offsets", offsets)
    _validate_1d("feature_offsets", feature_offsets)
    return feature_offsets.new_empty(feature_offsets.shape)


def _expand_table_ids_fake(
    offsets: torch.Tensor,
    table_offsets_in_feature: Optional[torch.Tensor] = None,
    num_tables: int = 0,
    local_batch_size: int = 1,
    num_elements: int = 0,
):
    del num_tables

    _validate_1d("offsets", offsets)

    if table_offsets_in_feature is not None:
        _validate_1d("table_offsets_in_feature", table_offsets_in_feature)

    if local_batch_size <= 0:
        raise RuntimeError(
            "INFERENCE_EMB::expand_table_ids expects local_batch_size > 0"
        )

    if num_elements < 0:
        raise RuntimeError(
            "INFERENCE_EMB::expand_table_ids expects num_elements >= 0"
        )

    return offsets.new_empty((num_elements,), dtype=torch.int64)


def register_index_range_fake() -> bool:
    """Register fake/meta kernels for INFERENCE_EMB index-range operators."""

    try:
        torch.library.register_fake("INFERENCE_EMB::get_table_range")(
            _get_table_range_fake
        )
        torch.library.register_fake("INFERENCE_EMB::expand_table_ids")(
            _expand_table_ids_fake
        )
        return True
    except Exception as e:
        warnings.warn(
            (
                "Failed to register fake kernels for INFERENCE_EMB index-range operators. "
                "Load inference_emb_ops.so before importing dynamicemb, or call "
                "dynamicemb.index_range_meta.register_index_range_fake() after load. "
                f"Original error: {e}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return False


REGISTERED = register_index_range_fake()
