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

"""Which tables FrequencyAdmissionStrategy(carry_count=True) may serve.

Only a table that evicts by LFU keeps an occurrence count as its score, so only
there does a carried count mean anything. Every other table refuses the option
when its options are built. Nothing here needs a device.
"""

import pytest
from dynamicemb import DynamicEmbScoreStrategy, FrequencyAdmissionStrategy, KVCounter
from dynamicemb.dynamicemb_config import DynamicEmbTableOptions


def _carrying_strategy() -> FrequencyAdmissionStrategy:
    return FrequencyAdmissionStrategy(
        threshold=4, counter=KVCounter(capacity=1024), carry_count=True
    )


@pytest.mark.parametrize(
    "score_strategy",
    [
        DynamicEmbScoreStrategy.LFU,
        (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU),
    ],
)
def test_an_lfu_table_takes_the_carried_count(score_strategy):
    options = DynamicEmbTableOptions(
        score_strategy=score_strategy, admit_strategy=_carrying_strategy()
    )
    assert options.admit_strategy.carry_count


@pytest.mark.parametrize(
    "score_strategy",
    [
        DynamicEmbScoreStrategy.TIMESTAMP,
        DynamicEmbScoreStrategy.STEP,
        DynamicEmbScoreStrategy.CUSTOMIZED,
    ],
)
def test_a_table_without_an_lfu_score_refuses_it(score_strategy):
    with pytest.raises(ValueError, match="carry_count"):
        DynamicEmbTableOptions(
            score_strategy=score_strategy, admit_strategy=_carrying_strategy()
        )


def test_off_is_the_default_and_any_table_takes_it():
    strategy = FrequencyAdmissionStrategy(threshold=4, counter=KVCounter(capacity=1024))
    assert not strategy.carry_count
    DynamicEmbTableOptions(
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP, admit_strategy=strategy
    )
