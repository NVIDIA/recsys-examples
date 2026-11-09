#!/bin/bash
set -e

# Retain-evicted-keys feature tests (table / storage / module / distributed).

# Table + storage level (single GPU).
pytest test/unit_tests/retain_evicted_keys/test_insert_collect_evicted.py -s

# Lazy retained keys: counts left on the device, folded later (single GPU).
pytest test/unit_tests/retain_evicted_keys/test_lazy_retain.py -s

# Module level (single GPU).
pytest test/unit_tests/retain_evicted_keys/test_pop_evicted_keys.py -s

# Distributed, model level.
torchrun --nproc_per_node=1 -m pytest test/unit_tests/retain_evicted_keys/test_distributed_pop_evicted_keys.py -s
torchrun --nproc_per_node=8 -m pytest test/unit_tests/retain_evicted_keys/test_distributed_pop_evicted_keys.py -s
