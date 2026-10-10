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

"""Tests for what ProbabilisticAdmissionStrategy admits.

Admission is consulted only for a key that is missing, so one toss per
appearance is what makes a key's chance of being in the table rise with how
often it turns up. These check the distribution that produces, the two ends
where no arithmetic applies, and that a batch holding a key k times is the same
as k separate appearances.
"""

import math

import pytest
import torch
from dynamicemb import ProbabilisticAdmissionStrategy
from dynamicemb.types import DynamicEmbInitializerArgs, DynamicEmbInitializerMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="admit draws on the device"
)

NUM_KEYS = 200_000

# How many standard deviations an observed rate may stray before a test fails.
# Five is a two-sided 5.7e-7: about one false failure in 1.7 million runs.
# Three would be one in 370, which across a suite run often is more trouble
# than what it is looking for.
SIGMAS = 5.0


def _admitter(probability, initializer_args=None):
    """The admitter one table configured this way would run."""
    strategy = ProbabilisticAdmissionStrategy(probability, initializer_args)
    return ProbabilisticAdmissionStrategy.create_admitter(
        [strategy], torch.device("cuda")
    )


def _inputs(num_keys=NUM_KEYS, device="cuda"):
    keys = torch.arange(num_keys, dtype=torch.int64, device=device)
    table_ids = torch.zeros(num_keys, dtype=torch.int64, device=device)
    return keys, table_ids


def _assert_fraction(admitted: torch.Tensor, expected: float):
    """Check an observed admission rate against the rate it was configured at.

    A key is admitted by drawing a uniform on [0, 1) and testing it against a
    threshold, and a uniform falls below t with probability exactly t. So each
    key is an independent Bernoulli(expected) -- the uniform is only the source
    -- their sum over N keys is Binomial(N, expected), and the fraction
    admitted has variance

        expected * (1 - expected) / N

    the Bernoulli variance divided by N, because dividing a sum by N divides
    its variance by N squared. Hence the square root below, and hence four
    times the keys only halving the window.

    Two limits on this. The normal tail SIGMAS is read against wants N * rate
    and N * (1 - rate) both above ~10, so a rate close to either end needs a
    coarser assertion instead. And it assumes one rate for every key: keys with
    different rates sum to a Poisson binomial, whose variance is the sum of the
    per-key ones rather than N times one of them.
    """
    num_keys = admitted.numel()
    observed = float(admitted.sum()) / num_keys
    # The floor keeps a rate of exactly 0 or 1 -- no randomness, so no variance
    # -- from collapsing the window to nothing. Nothing here reaches it.
    tolerance = SIGMAS * math.sqrt(max(expected * (1.0 - expected), 1e-12) / num_keys)
    assert (
        abs(observed - expected) <= tolerance
    ), f"admitted {observed:.6f}, expected {expected:.6f} +/- {tolerance:.6f}"


def test_admits_at_the_configured_rate():
    probability = 0.3
    keys, table_ids = _inputs()
    admitted = _admitter(probability).admit(keys, table_ids)
    _assert_fraction(admitted, probability)


def test_probability_zero_admits_nothing():
    # torch.rand can return exactly 0.0, so this only holds if the comparison
    # is strict. It is also the case log1p(-p) has no value for, at the far end.
    keys, table_ids = _inputs()
    admitted = _admitter(0.0).admit(keys, table_ids)
    assert not bool(admitted.any())


def test_probability_one_admits_everything():
    keys, table_ids = _inputs()
    admitted = _admitter(1.0).admit(keys, table_ids)
    assert bool(admitted.all())


@pytest.mark.parametrize("occurrences", [1, 20])
def test_repeats_within_a_batch_count_as_separate_tosses(occurrences):
    """A count of one has to come out as the plain probability, and k as
    1 - (1 - p)^k. Compared by rate rather than key for key: the compounded
    threshold reaches p through log1p/expm1 and lands within a float32 ulp of
    it, so the two paths disagree on a draw in that last ulp now and then."""
    probability = 0.1
    keys, table_ids = _inputs()
    frequencies = torch.full_like(keys, occurrences)
    admitted = _admitter(probability).admit(keys, table_ids, frequencies)
    _assert_fraction(admitted, 1.0 - (1.0 - probability) ** occurrences)


def test_compounding_survives_a_probability_too_small_for_float32():
    """1 - p rounds to 1 in float32 below 2^-24, which would admit nothing.

    Written against p = 1e-8 and a count large enough for the true chance to be
    1 - e^-10: an implementation computing (1 - p)^k directly admits no key at
    all here, so this separates the two without needing a large sample.
    """
    probability, occurrences = 1e-8, 10**9
    keys, table_ids = _inputs()
    frequencies = torch.full_like(keys, occurrences)
    admitted = _admitter(probability).admit(keys, table_ids, frequencies)
    # Asserted coarsely on purpose. The true rate is 1 - e^-10 = 0.99995, which
    # leaves about nine rejections out of NUM_KEYS -- too few for the normal
    # approximation _assert_fraction rests on. Telling 0.99995 from 0 needs no
    # precision anyway.
    assert float(admitted.sum()) / admitted.numel() > 0.99


def test_tables_agreeing_only_on_probability_fuse():
    # The initializer's parameters are resolved per table, so they are not part
    # of what has to match; the probability is.
    def strategy(probability, value):
        return ProbabilisticAdmissionStrategy(
            probability,
            DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT, value=value
            ),
        )

    assert strategy(0.3, 1.0).get_grouped_key() == strategy(0.3, 2.0).get_grouped_key()
    assert strategy(0.3, 1.0).get_grouped_key() != strategy(0.4, 1.0).get_grouped_key()


def test_rejects_a_probability_outside_the_unit_interval():
    with pytest.raises(ValueError):
        ProbabilisticAdmissionStrategy(1.5)
    with pytest.raises(ValueError):
        ProbabilisticAdmissionStrategy(-0.1)
