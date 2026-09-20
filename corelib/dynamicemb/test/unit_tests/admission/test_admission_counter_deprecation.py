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

"""Tests for the deprecated DynamicEmbTableOptions.admission_counter.

A counter used to sit on the table beside the strategy rather than on the
strategy that counts. Configuring it that way still works and still sizes each
table's counter separately; it only warns. These pin that down so the shim can
be deleted one day on purpose rather than by accident, and so the two things
easy to get wrong about it -- whose object gets written to, and which table's
capacity survives when one strategy serves every table -- stay right meanwhile.

Nothing here needs a device: it is all what DynamicEmbTableOptions settles at
construction.
"""

import pytest
from dynamicemb import FrequencyAdmissionStrategy, KVCounter
from dynamicemb.dynamicemb_config import DynamicEmbTableOptions


def test_the_old_spelling_still_reaches_the_strategy():
    counter = KVCounter(capacity=1024)
    with pytest.warns(DeprecationWarning, match="admission_counter"):
        options = DynamicEmbTableOptions(
            admit_strategy=FrequencyAdmissionStrategy(threshold=4),
            admission_counter=counter,
        )
    assert options.admit_strategy.counter is counter
    assert options.admit_strategy.threshold == 4


def test_each_table_keeps_the_capacity_it_was_given():
    """The counter is folded into a copy, once per table.

    One strategy handed to every table is the usual way to configure this, so
    folding in place would leave all of them holding whichever table was
    configured last.
    """
    shared = FrequencyAdmissionStrategy(threshold=4)
    with pytest.warns(DeprecationWarning):
        small = DynamicEmbTableOptions(
            admit_strategy=shared, admission_counter=KVCounter(capacity=1024)
        )
        large = DynamicEmbTableOptions(
            admit_strategy=shared, admission_counter=KVCounter(capacity=4096)
        )

    assert small.admit_strategy.counter.capacity == 1024
    assert large.admit_strategy.counter.capacity == 4096
    # and the object the caller still holds is untouched
    assert shared.counter is None


def test_a_strategy_that_brought_its_own_counter_keeps_it():
    own = KVCounter(capacity=1024)
    strategy = FrequencyAdmissionStrategy(threshold=4, counter=own)
    with pytest.warns(DeprecationWarning):
        options = DynamicEmbTableOptions(
            admit_strategy=strategy, admission_counter=KVCounter(capacity=4096)
        )
    assert options.admit_strategy.counter is own
    # nothing to fold, so nothing to copy either
    assert options.admit_strategy is strategy


def test_leaving_it_unset_says_nothing(recwarn):
    DynamicEmbTableOptions(
        admit_strategy=FrequencyAdmissionStrategy(
            threshold=4, counter=KVCounter(capacity=1024)
        )
    )
    deprecations = [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, [str(w.message) for w in deprecations]
