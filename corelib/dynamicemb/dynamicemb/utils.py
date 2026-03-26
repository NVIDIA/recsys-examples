# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
from typing import List, Optional, Set, Type, Union, cast

import torch

# from torchrec.distributed import ModuleShardingPlan
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)

TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}


def tabulate(
    table: List[List[Union[str, int]]],
    headers: Optional[List[str]] = None,
    sub_headers: bool = False,
) -> str:
    """
    Format a table as a string.
    Parameters:
        table (list of lists or list of tuples): The data to be formatted as a table.
        headers (list of strings, optional): The column headers for the table. If not provided, the first row of the table will be used as the headers.
    Returns:
        str: A string representation of the table.
    """
    if headers is None:
        headers = table[0]
        table = table[1:]
    headers = cast(List[str], headers)
    rows = []
    # Determine the maximum width of each column
    col_widths = [max([len(str(item)) for item in column]) for column in zip(*table)]
    col_widths = [max(i, len(j)) for i, j in zip(col_widths, headers)]
    # Format each row of the table
    for row in table:
        row_str = " | ".join(
            [str(item).ljust(width) for item, width in zip(row, col_widths)]
        )
        rows.append(row_str)
    # Add the header row and the separator line
    rows.insert(
        0,
        " | ".join(
            [header.center(width) for header, width in zip(headers, col_widths)]
        ),
    )

    rows.insert(1, " | ".join(["-" * width for width in col_widths]))
    if sub_headers:
        rows.insert(3, " | ".join(["-" * width for width in col_widths]))
    return "\n".join(rows)


def assert_tensors_equal(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Assert that two tensors are equal in shape and values.

    Parameters:
    tensor1 (torch.Tensor): The first tensor to compare.
    tensor2 (torch.Tensor): The second tensor to compare.
    atol (float): Absolute tolerance. Default is 1e-8.
    rtol (float): Relative tolerance. Default is 1e-5.
    dtype (Optional[torch.dtype]): Expected data type of the tensors. If None, no check is performed.
    device (Optional[torch.device]): Expected device of the tensors. If None, no check is performed.
    """
    assert isinstance(
        tensor1, torch.Tensor
    ), f"tensor1 must be a torch.Tensor, but got {type(tensor1)}"
    assert isinstance(
        tensor2, torch.Tensor
    ), f"tensor2 must be a torch.Tensor, but got {type(tensor2)}"

    if dtype is not None:
        assert (
            tensor1.dtype == dtype
        ), f"tensor1 dtype is {tensor1.dtype}, expected {dtype}"
        assert (
            tensor2.dtype == dtype
        ), f"tensor2 dtype is {tensor2.dtype}, expected {dtype}"

    if device is not None:
        assert (
            tensor1.device == device
        ), f"tensor1 device is {tensor1.device}, expected {device}"
        assert (
            tensor2.device == device
        ), f"tensor2 device is {tensor2.device}, expected {device}"

    assert (
        tensor1.shape == tensor2.shape
    ), f"Shapes are different: {tensor1.shape} vs {tensor2.shape}"

    if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
        differences = (tensor1 != tensor2).nonzero(as_tuple=True)
        for idx in zip(*differences):
            print(
                f"Difference at index {idx}: tensor1={tensor1[idx].item()}, tensor2={tensor2[idx].item()}"
            )
        raise AssertionError("Values are different")
    print("Assert success: Shapes and values are equal.")
