# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Optional

import torch


def _validate_1d(name: str, t: torch.Tensor) -> None:
    if t.dim() != 1:
        raise RuntimeError(
            f"INFERENCE_EMB::table_lookup expects 1D {name}, got dim={t.dim()}"
        )


def _table_lookup_fake(
    table_storage: torch.Tensor,
    table_bucket_offsets: torch.Tensor,
    bucket_capacity: int,
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    score_input: Optional[torch.Tensor],
    policy_type: int,
    ovf_storage: Optional[torch.Tensor] = None,
    ovf_bucket_capacity: int = 0,
    ovf_output_offsets: Optional[torch.Tensor] = None,
):
    del table_storage, table_bucket_offsets, bucket_capacity, policy_type, ovf_bucket_capacity

    _validate_1d("keys", keys)
    _validate_1d("table_ids", table_ids)

    if keys.numel() != table_ids.numel():
        raise RuntimeError(
            "INFERENCE_EMB::table_lookup expects keys and table_ids to have same length"
        )

    if score_input is not None and score_input.numel() != keys.numel():
        raise RuntimeError(
            "INFERENCE_EMB::table_lookup expects score_input length == keys length"
        )

    if ovf_storage is not None and ovf_output_offsets is None:
        raise RuntimeError(
            "INFERENCE_EMB::table_lookup with ovf_storage requires ovf_output_offsets"
        )

    n = keys.numel()
    score_out = keys.new_empty((n,), dtype=torch.int64)
    founds = keys.new_empty((n,), dtype=torch.bool)
    indices = keys.new_empty((n,), dtype=torch.int64)
    return score_out, founds, indices


def register_lookup_fake() -> bool:
    """Register fake/meta kernel for INFERENCE_EMB::table_lookup.

    Returns:
        bool: True when registration succeeds, False otherwise.
    """

    try:
        torch.library.register_fake("INFERENCE_EMB::table_lookup")(_table_lookup_fake)
        return True
    except Exception as e:
        warnings.warn(
            (
                "Failed to register fake kernel for INFERENCE_EMB::table_lookup. "
                "Load inference_emb_ops.so before importing dynamicemb, or call "
                "dynamicemb.lookup_meta.register_lookup_fake() after load. "
                f"Original error: {e}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return False


REGISTERED = register_lookup_fake()
