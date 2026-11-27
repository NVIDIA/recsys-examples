# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests of the C++ views of tables of different embedding dimensions
# (src/state/mixed_dim_view.h): what they hold and what their constructors reject.
# That kernels read the right fields is covered where kernels take the views.
#
# Run (from ``corelib/dynamicemb``)::
#   python -m pytest test/unit_tests/test_mixed_dim_view.py -q

from __future__ import annotations

import pytest
import torch
from dynamicemb_extensions import FlatTablesView, PaddedBufferView

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def i64(x, device="cuda") -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int64, device=device)


def make_flat(**overrides) -> FlatTablesView:
    args = dict(
        ptrs=i64([1000, 2000]),
        value_dims=i64([12, 8]),
        emb_dims=i64([4, 8]),
        max_emb_dim=8,
        max_value_dim=12,
        all_dims_vec4=True,
        dtype=torch.float32,
    )
    args.update(overrides)
    return FlatTablesView(**args)


def make_padded(**overrides) -> PaddedBufferView:
    args = dict(
        emb_dims=i64([4, 8]), max_emb_dim=8, max_value_dim=12, all_dims_vec4=True
    )
    args.update(overrides)
    return PaddedBufferView(**args)


def test_padded_buffer_view_fields():
    emb_dims = i64([4, 8])
    v = PaddedBufferView(emb_dims, 8, 12, True)
    assert v.num_tables == 2
    assert v.max_emb_dim == 8 and v.max_value_dim == 12 and v.all_dims_vec4
    # By reference: the view's tensor is the caller's, not a copy.
    assert v.emb_dims.data_ptr() == emb_dims.data_ptr()


def test_flat_tables_view_fields():
    ptrs, value_dims, emb_dims = i64([1000, 2000]), i64([12, 8]), i64([4, 8])
    v = FlatTablesView(ptrs, value_dims, emb_dims, 8, 12, False, torch.float16)
    assert v.num_tables == 2
    assert v.max_emb_dim == 8 and v.max_value_dim == 12 and not v.all_dims_vec4
    assert v.dtype == torch.float16
    assert v.ptrs.data_ptr() == ptrs.data_ptr()
    assert v.value_dims.data_ptr() == value_dims.data_ptr()
    assert v.emb_dims.data_ptr() == emb_dims.data_ptr()


def test_flat_tables_view_follows_in_place_refresh():
    # The Python owner refreshes ptrs in place when a buffer grows; a view made
    # before that reads the new values through the same tensor.
    ptrs = i64([1000, 2000])
    v = make_flat(ptrs=ptrs)
    ptrs.copy_(i64([3000, 4000]))
    assert v.ptrs.tolist() == [3000, 4000]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn]
)
def test_flat_tables_view_accepts_storage_dtypes(dtype):
    assert make_flat(dtype=dtype).dtype == dtype


def test_flat_tables_view_rejects_unsupported_dtype():
    with pytest.raises(Exception, match="Unsupported"):
        make_flat(dtype=torch.float64)


@pytest.mark.parametrize("name", ["ptrs", "value_dims", "emb_dims"])
def test_flat_tables_view_rejects_bad_tensors(name):
    with pytest.raises(Exception, match=name):
        make_flat(**{name: i64([1, 2], device="cpu")})  # not CUDA
    with pytest.raises(Exception, match=name):
        make_flat(**{name: torch.tensor([1, 2], dtype=torch.int32, device="cuda")})
    with pytest.raises(Exception, match=name):
        make_flat(**{name: i64([[1, 2]])})  # not 1-D
    with pytest.raises(Exception, match=name):
        make_flat(**{name: i64([1, 2, 3, 4])[::2]})  # not contiguous


def test_flat_tables_view_rejects_length_mismatch():
    with pytest.raises(Exception, match="one entry per table"):
        make_flat(value_dims=i64([12, 8, 8]))
    with pytest.raises(Exception, match="must not be empty"):
        make_flat(ptrs=i64([]), value_dims=i64([]), emb_dims=i64([]))


def test_padded_buffer_view_rejects_bad_emb_dims():
    with pytest.raises(Exception, match="emb_dims"):
        make_padded(emb_dims=i64([4, 8], device="cpu"))
    with pytest.raises(Exception, match="emb_dims"):
        make_padded(emb_dims=torch.tensor([4, 8], dtype=torch.int32, device="cuda"))
    with pytest.raises(Exception, match="must not be empty"):
        make_padded(emb_dims=i64([]))


@pytest.mark.parametrize("make", [make_flat, make_padded])
def test_views_reject_inconsistent_widths(make):
    with pytest.raises(Exception, match="max_emb_dim"):
        make(max_emb_dim=0)
    with pytest.raises(Exception, match="must cover max_emb_dim"):
        make(max_emb_dim=8, max_value_dim=4)
