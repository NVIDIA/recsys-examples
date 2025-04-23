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

import pytest
import torch
from dynamicemb.unique_op import UniqueOp


def generate_random_integers(length, device, low=0, high=100, dtype=torch.int64):
    return torch.randint(low, high, (length,), device=device, dtype=dtype)


def compare_results(
    custom_unique_keys,
    custom_inversed_indices,
    original_keys,
    result_counter,
    pytorch_unique_keys,
    offset=0,
):
    # Convert tensors to CPU for comparison
    custom_unique_keys = custom_unique_keys.to("cpu")[: result_counter[0]]
    custom_inversed_indices = custom_inversed_indices.to("cpu") - offset
    original_keys = original_keys.to("cpu")
    pytorch_unique_keys = pytorch_unique_keys.to("cpu")

    # Compare the number of unique keys
    assert (
        custom_unique_keys.shape[0] == pytorch_unique_keys.shape[0]
    ), f"Unique keys count do not match. Custom: {custom_unique_keys.shape[0]}, PyTorch: {pytorch_unique_keys.shape[0]}"

    # Sort the unique keys for comparison
    custom_unique_keys_sorted = torch.sort(custom_unique_keys).values
    pytorch_unique_keys_sorted = torch.sort(pytorch_unique_keys).values

    # Compare the unique keys values
    assert torch.equal(
        custom_unique_keys_sorted, pytorch_unique_keys_sorted
    ), f"Unique keys values do not match after sorting.\nCustom unique keys: {custom_unique_keys_sorted}\nPyTorch unique keys: {pytorch_unique_keys_sorted}"

    # Reconstruct original keys using unique keys and inversed indices
    reconstructed_keys = custom_unique_keys[custom_inversed_indices]

    assert torch.equal(
        reconstructed_keys, original_keys
    ), f"Inverse indices do not correctly reconstruct the original keys.\nReconstructed keys: {reconstructed_keys}\nOriginal keys: {original_keys}"


@pytest.fixture
def setup_device():
    assert torch.cuda.is_available()
    device_id = 0
    return torch.device(f"cuda:{device_id}")


@pytest.fixture
def setup_unique_op(setup_device):
    device = setup_device
    key_type = torch.int64
    val_type = torch.int64

    keys = torch.tensor(
        [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], dtype=key_type, device=device
    )
    capacity = keys.shape[0] * 2

    reserve_keys = torch.zeros(capacity, dtype=key_type, device=device)
    reserve_vals = torch.zeros(capacity, dtype=val_type, device=device)
    counter = torch.tensor(1, dtype=val_type, device=device)

    unique_op = UniqueOp(reserve_keys, reserve_vals, counter, capacity)
    return unique_op, keys, device, key_type, val_type


def test_first_unique(setup_unique_op):
    unique_op, keys, device, key_type, val_type = setup_unique_op

    capacity = keys.shape[0] * 2
    unique_keys = torch.zeros(capacity, dtype=key_type, device=device)
    inversed_indices = torch.zeros(keys.shape[0], dtype=val_type, device=device)
    result_counter = torch.zeros(1, dtype=val_type, device=device)

    unique_op.unique(keys, keys.shape[0], inversed_indices, unique_keys, result_counter)

    pytorch_unique_keys, pytorch_inverse_indices = torch.unique(
        keys, return_inverse=True
    )
    torch.cuda.synchronize()

    compare_results(
        unique_keys, inversed_indices, keys, result_counter, pytorch_unique_keys
    )
    current_capacity = unique_op.get_capacity()
    print(f"first capacity: {current_capacity}")


def test_second_unique(setup_unique_op):
    unique_op, keys, device, key_type, val_type = setup_unique_op

    new_length = 512
    new_reserve_keys = torch.zeros(new_length * 2, dtype=key_type, device=device)
    new_reserve_vals = torch.zeros(new_length * 2, dtype=val_type, device=device)
    unique_op.reset_capacity(new_reserve_keys, new_reserve_vals, new_length * 2)

    low = 0
    high = 132
    new_keys = generate_random_integers(new_length, device, low, high, dtype=key_type)
    new_unique_keys = torch.zeros(new_length, dtype=key_type, device=device)
    new_inversed_indices = torch.zeros(new_length, dtype=val_type, device=device)
    new_result_counter = torch.zeros(1, dtype=val_type, device=device)

    unique_op.unique(
        new_keys,
        new_keys.shape[0],
        new_inversed_indices,
        new_unique_keys,
        new_result_counter,
    )
    new_pytorch_unique_keys, new_pytorch_inverse_indices = torch.unique(
        new_keys, return_inverse=True
    )
    torch.cuda.synchronize()

    compare_results(
        new_unique_keys[: new_result_counter.item()],
        new_inversed_indices,
        new_keys,
        new_result_counter,
        new_pytorch_unique_keys,
    )
    current_capacity = unique_op.get_capacity()
    print(f"second capacity: {current_capacity}")


def test_third_unique(setup_unique_op):
    unique_op, keys, device, key_type, val_type = setup_unique_op

    random_length = torch.randint(5000, 10000, (1,)).item()
    random_offset = torch.randint(0, 100, (1,)).item()

    third_reserve_keys = torch.zeros(random_length * 2, dtype=key_type, device=device)
    third_reserve_vals = torch.zeros(random_length * 2, dtype=val_type, device=device)
    unique_op.reset_capacity(third_reserve_keys, third_reserve_vals, random_length * 2)

    third_keys = generate_random_integers(
        random_length, device, low=0, high=132, dtype=key_type
    )
    third_unique_keys = torch.zeros(random_length, dtype=key_type, device=device)
    third_inversed_indices = torch.zeros(random_length, dtype=val_type, device=device)
    third_result_counter = torch.zeros(1, dtype=val_type, device=device)

    offset_tensor = torch.tensor([random_offset], dtype=val_type, device=device)
    unique_op.unique(
        third_keys,
        third_keys.shape[0],
        third_inversed_indices,
        third_unique_keys,
        third_result_counter,
        offset=offset_tensor,
    )
    third_pytorch_unique_keys, third_pytorch_inverse_indices = torch.unique(
        third_keys, return_inverse=True
    )
    torch.cuda.synchronize()

    compare_results(
        third_unique_keys[: third_result_counter.item()],
        third_inversed_indices,
        third_keys,
        third_result_counter,
        third_pytorch_unique_keys,
        offset=random_offset,
    )
    current_capacity = unique_op.get_capacity()
    print(f"third capacity: {current_capacity}")


if __name__ == "__main__":
    pytest.main()
