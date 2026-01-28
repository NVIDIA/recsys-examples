# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dynamicemb.extendable_buffer import VmmDeviceBuffer


@pytest.mark.parametrize("init_capacity", [1, 1023, 1024, 1 << 30])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_vmm_buffer(init_capacity, dtype):
    device = torch.device("cuda", torch.cuda.current_device())

    buffer = VmmDeviceBuffer(init_capacity, dtype, device)

    apply_capacity = init_capacity
    real_capaicty = buffer.capacity()
    pointer = buffer.tensor().data_ptr()
    print(f"Base address: {hex(pointer)}")

    try:
        while True:
            buffer.extend(apply_capacity * 2)

            apply_capacity = apply_capacity * 2
            real_capaicty = buffer.capacity()

            assert (
                pointer == buffer.tensor().data_ptr()
            ), "Base address should keep the same after extending."

            buffer.tensor().fill_(-1)

            # synchronize is necessary to avoid IMA, we will add device synchronization to extend.
            # stream synchronization and device synchronization in C++ not worked.
            torch.cuda.synchronize()

            print(
                f"Extend successfully with dtype={dtype}, apply_capacity={apply_capacity},  real_capaicty={real_capaicty}"
            )

    except RuntimeError as e:
        print(
            f"RuntimeError: {e} after extending buffer with dtype={dtype}, apply_capacity={apply_capacity},  real_capaicty={real_capaicty}"
        )
