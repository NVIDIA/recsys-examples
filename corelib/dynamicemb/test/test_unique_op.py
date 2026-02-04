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
from dynamicemb_extensions import unique_cuda


def generate_random_integers(length, device, low=0, high=100, dtype=torch.int64):
    return torch.randint(low, high, (length,), device=device, dtype=dtype)


def compare_results(
    custom_unique_keys,
    custom_inversed_indices,
    original_keys,
    num_unique,
    pytorch_unique_keys,
):
    # Convert tensors to CPU for comparison
    count = num_unique.item()
    custom_unique_keys = custom_unique_keys.to("cpu")[:count]
    custom_inversed_indices = custom_inversed_indices.to("cpu")
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


def test_basic_unique(setup_device):
    """Test basic unique operation."""
    device = setup_device
    key_type = torch.int64

    keys = torch.tensor(
        [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], dtype=key_type, device=device
    )

    unique_keys, inversed_indices, num_unique = unique_cuda(keys, None, None)

    pytorch_unique_keys, pytorch_inverse_indices = torch.unique(
        keys, return_inverse=True
    )
    torch.cuda.synchronize()

    compare_results(
        unique_keys, inversed_indices, keys, num_unique, pytorch_unique_keys
    )
    print(f"Basic test: found {num_unique.item()} unique keys")


def test_random_unique(setup_device):
    """Test unique with random inputs."""
    device = setup_device
    key_type = torch.int64

    length = 512
    low = 0
    high = 132
    keys = generate_random_integers(length, device, low, high, dtype=key_type)

    unique_keys, inversed_indices, num_unique = unique_cuda(keys, None, None)

    pytorch_unique_keys, pytorch_inverse_indices = torch.unique(
        keys, return_inverse=True
    )
    torch.cuda.synchronize()

    compare_results(
        unique_keys, inversed_indices, keys, num_unique, pytorch_unique_keys
    )
    print(f"Random test: found {num_unique.item()} unique keys from {length} inputs")


def test_empty_input(setup_device):
    """Test unique with empty input."""
    device = setup_device
    key_type = torch.int64

    keys = torch.tensor([], dtype=key_type, device=device)

    unique_keys, inversed_indices, num_unique = unique_cuda(keys, None, None)

    assert num_unique.item() == 0, "Empty input should return 0 unique keys"
    assert unique_keys.numel() == 0, "Empty input should return empty unique keys"
    assert inversed_indices.numel() == 0, "Empty input should return empty indices"
    print("Empty input test passed")


def test_all_same_keys(setup_device):
    """Test unique with all same keys."""
    device = setup_device
    key_type = torch.int64

    keys = torch.full((1000,), 42, dtype=key_type, device=device)

    unique_keys, inversed_indices, num_unique = unique_cuda(keys, None, None)

    assert num_unique.item() == 1, "All same keys should return 1 unique key"
    assert unique_keys[0].item() == 42, "Unique key should be 42"
    assert torch.all(inversed_indices == 0), "All indices should map to 0"
    print("All same keys test passed")


def test_all_unique_keys(setup_device):
    """Test unique with all unique keys (no duplicates)."""
    device = setup_device
    key_type = torch.int64

    keys = torch.arange(1000, dtype=key_type, device=device)

    unique_keys, inversed_indices, num_unique = unique_cuda(keys, None, None)

    assert num_unique.item() == 1000, "All unique keys should return same count"
    print("All unique keys test passed")


def test_uint64_dtype(setup_device):
    """Test unique with uint64 dtype."""
    device = setup_device
    key_type = torch.uint64

    keys = torch.tensor(
        [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], dtype=key_type, device=device
    )

    unique_keys, inversed_indices, num_unique = unique_cuda(keys, None, None)

    pytorch_unique_keys, pytorch_inverse_indices = torch.unique(
        keys, return_inverse=True
    )
    torch.cuda.synchronize()

    # Compare count
    assert num_unique.item() == pytorch_unique_keys.shape[0]
    print(f"uint64 test: found {num_unique.item()} unique keys")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
