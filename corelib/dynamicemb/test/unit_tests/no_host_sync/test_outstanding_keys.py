# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A cache cannot evict a key that a prefetched batch is still going to use, so the
batches that are prefetched and not forwarded yet must fit in it together
(``OutstandingKeys``). The check must never be a false alarm and must not wait for
the device: a batch that is all duplicates is long in keys but short in unique keys,
and the unique count lives on the device.

The first half drives ``OutstandingKeys`` itself, with the unique counts on the
device as a padded front end hands them over. The second half drives a caching module.
"""

import pytest
import torch
from dynamicemb import (
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_function import OutstandingKeys
from dynamicemb.batched_dynamicemb_tables import (
    BatchedDynamicEmbeddingTablesV2,
    StorageLayout,
)

CAPACITY = 100


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def device_total(count: int, device) -> torch.Tensor:
    """The unique count of a batch as the dedup leaves it: a 0-d int64 on the device."""
    return torch.tensor(count, dtype=torch.int64, device=device)


def prefetch_all(outstanding, batches, device, capacity=CAPACITY):
    """Prefetch ``batches`` -- ``(bound, unique count)`` pairs -- one after the other,
    letting the device catch up between them, as a pipeline with slack does. Returns
    the error that stopped it and the number of the batch that was being prefetched."""
    entries = []
    for number, (bound, unique) in enumerate(batches):
        entries.append(outstanding.add(bound, device_total(unique, device), None))
        try:
            outstanding.check(capacity)
        except RuntimeError as error:
            return error, number
        torch.cuda.synchronize()
    return None, len(batches)


def test_duplicates_are_never_an_overflow(current_device):
    """Three outstanding batches of 60 keys, 20 of them unique, in a cache of 100:
    the keys they carry add up to 180 but the keys they pin to 60."""
    device = torch.device("cuda", current_device)
    outstanding = OutstandingKeys()
    error, _ = prefetch_all(outstanding, [(60, 20)] * 3, device)
    assert error is None
    # Whatever is still being counted lands, and says it fits.
    for _ in range(3):
        outstanding.check(CAPACITY)
        torch.cuda.synchronize()
    assert outstanding.bound == 180


def test_a_real_overflow_raises_within_two_prefetches(current_device):
    """Three outstanding batches of 60 unique keys do not fit in a cache of 100.
    The second one is where it overflows: the error comes with it or one prefetch
    later, and names the exact count, in the words it always had."""
    device = torch.device("cuda", current_device)
    outstanding = OutstandingKeys()
    error, number = prefetch_all(outstanding, [(60, 60)] * 3, device)
    assert error is not None, "an over-subscribed cache was never reported"
    assert number <= 2
    message = str(error)
    assert message.startswith("Outstanding prefetched keys (")
    assert message.endswith(
        "exceed cache capacity (100). "
        "Reduce prefetch pipeline depth or increase cache size."
    )
    count = int(message.split("(")[1].split(")")[0])
    assert count in (120, 180)


def test_a_batch_with_a_host_count_is_checked_at_once(current_device):
    """A front end that knows its unique count on the host (the exact arrays of the
    legacy path, a host-sized storage) is checked as it always was: immediately."""
    outstanding = OutstandingKeys()
    outstanding.add(70, None, 70)
    outstanding.check(CAPACITY)
    outstanding.add(70, None, 70)
    with pytest.raises(RuntimeError) as raised:
        outstanding.check(CAPACITY)
    assert str(raised.value) == (
        "Outstanding prefetched keys (140) exceed cache capacity (100). "
        "Reduce prefetch pipeline depth or increase cache size."
    )


def test_host_and_device_counts_add_up(current_device):
    device = torch.device("cuda", current_device)
    outstanding = OutstandingKeys()
    outstanding.add(60, None, 40)
    outstanding.add(60, device_total(70, device), None)
    outstanding.check(CAPACITY)  # 120 keys may not fit: the exact count is asked for
    torch.cuda.synchronize()
    outstanding.add(1, None, 1)
    with pytest.raises(RuntimeError, match=r"\(110\)"):
        outstanding.check(CAPACITY)  # 40 on the host + 70 from the device


def test_release_gives_the_keys_back(current_device):
    outstanding = OutstandingKeys()
    first = outstanding.add(10, None, 10)
    second = outstanding.add(20, None, 20)
    outstanding.release(first)
    outstanding.release(first)  # a second release, or none at all, changes nothing
    outstanding.release(None)
    assert outstanding.bound == 20
    outstanding.release(second)
    assert outstanding.bound == 0
    outstanding.check(CAPACITY)


def test_reset_forgets_everything(current_device):
    """``reset_cache_states`` empties the cache, so nothing is outstanding any more --
    not the batches, and not a count that is still on its way."""
    device = torch.device("cuda", current_device)
    outstanding = OutstandingKeys()
    stale = outstanding.add(60, device_total(60, device), None)
    outstanding.add(60, device_total(60, device), None)
    outstanding.check(CAPACITY)  # inconclusive: a count is in flight
    outstanding.reset()
    assert outstanding.bound == 0
    torch.cuda.synchronize()
    outstanding.check(CAPACITY)  # the landed count of the old batches is not heard of
    outstanding.release(stale)  # nor is the release of a batch that was reset away
    assert outstanding.bound == 0
    error, _ = prefetch_all(outstanding, [(60, 20)] * 3, device)
    assert error is None


def test_checking_never_waits_for_the_device(current_device):
    """Sync-debug mode raises on anything that waits: the check that cannot decide
    from the host numbers asks the device late instead, and a full ring of asks
    simply stops asking."""
    device = torch.device("cuda", current_device)
    outstanding = OutstandingKeys()
    totals = [device_total(1, device) for _ in range(12)]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for total in totals:
            outstanding.add(60, total, None)
            outstanding.check(CAPACITY)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


# --- a caching module -----------------------------------------------------------

DIM = 16
ROWS = 64 * 1024
TABLE_BYTES = ROWS * DIM * 4  # fp32, SGD keeps no optimizer state


def build_caching_model(current_device: int):
    """One table, with a cache of ``ROWS // 16`` rows in front of the host storage."""
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=ROWS,
        local_hbm_for_values=TABLE_BYTES // 16,
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        caching=True,
    )
    model = BatchedDynamicEmbeddingTablesV2(
        table_options=[options],
        table_names=["t_0"],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.0,
        stochastic_rounding=False,
    )
    assert model._storage_layout == StorageLayout.CACHING
    return model


def one_key_per_bag(keys, device):
    indices = torch.as_tensor(keys, dtype=torch.int64).to(device)
    offsets = torch.arange(indices.numel() + 1, dtype=torch.int64, device=device)
    return indices, offsets


def test_a_module_over_its_cache_raises_and_reset_clears(current_device):
    device = torch.device("cuda", current_device)
    model = build_caching_model(current_device)
    capacity = model.cache._state.capacity
    n = int(0.6 * capacity)
    batches = [one_key_per_bag(range(i * n, (i + 1) * n), device) for i in range(3)]

    model.prefetch(*batches[0])  # one batch fits
    with pytest.raises(RuntimeError, match="Outstanding prefetched keys"):
        # A second does not; a late report comes with the third prefetch at the latest.
        for batch in batches[1:]:
            model.prefetch(*batch)
            torch.cuda.synchronize()

    model.reset_cache_states()
    assert model._prefetch_outstanding_keys.bound == 0
    model._prefetch_states.clear()
    # The cache is empty again: a batch is prefetched, forwarded and released.
    model.prefetch(*batches[0])
    model(*batches[0]).sum().backward()
    torch.cuda.synchronize()
    assert model._prefetch_outstanding_keys.bound == 0


def test_a_module_with_duplicates_never_overflows(current_device):
    """Batches longer than the cache but made of 50 distinct keys, three of them
    outstanding: they pin 50 keys, not 15,000."""
    device = torch.device("cuda", current_device)
    model = build_caching_model(current_device)
    capacity = model.cache._state.capacity
    n = capacity + capacity // 4
    batch = one_key_per_bag([k % 50 for k in range(n)], device)

    for _ in range(3):
        model.prefetch(*batch)
        torch.cuda.synchronize()
    for _ in range(3):
        model(*batch).sum().backward()
    torch.cuda.synchronize()
    assert model._prefetch_outstanding_keys.bound == 0
