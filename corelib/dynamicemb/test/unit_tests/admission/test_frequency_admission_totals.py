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

"""What the frequency admitter hands back in ``frequencies``.

``frequencies`` comes in as this batch's increment and goes back as each key's
running total, which a table that evicts by LFU takes as the admitted key's score.
"""

import pytest
import torch
from dynamicemb import FrequencyAdmissionStrategy, KVCounter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the counter lives on the GPU"
)


def _admitter(threshold):
    strategy = FrequencyAdmissionStrategy(
        threshold=threshold, counter=KVCounter(capacity=1024)
    )
    return FrequencyAdmissionStrategy.create_admitter(
        [strategy], torch.device("cuda")
    )


def _tensor(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, device="cuda")


def test_the_frequencies_given_come_back_as_running_totals():
    admitter = _admitter(threshold=3)
    keys, table_ids = _tensor([1, 2, 3, 4]), _tensor([0, 0, 0, 0])

    frequencies = _tensor([1, 2, 3, 5])
    admitted = admitter.admit(keys, table_ids, frequencies)
    # Nothing was waiting yet: the totals are the increments.
    assert admitted.tolist() == [False, False, True, True]
    assert frequencies.tolist() == [1, 2, 3, 5]

    # Keys 3 and 4 were admitted, so they stopped waiting; 1 and 2 kept their count.
    frequencies = _tensor([1, 1, 1, 1])
    admitted = admitter.admit(keys, table_ids, frequencies)
    assert frequencies.tolist() == [2, 3, 1, 1]
    assert admitted.tolist() == [False, True, False, False]


def test_without_frequencies_each_key_counts_once_and_nothing_is_handed_back():
    admitter = _admitter(threshold=2)
    keys, table_ids = _tensor([1, 2]), _tensor([0, 0])
    assert admitter.admit(keys, table_ids).tolist() == [False, False]
    assert admitter.admit(keys, table_ids).tolist() == [True, True]
