#!/bin/bash 
set -e

pytest test/unit_tests/table_operation/test_table_operation.py -s

pytest test/unit_tests/table_operation/test_lru_lfu.py -s

pytest test/unit_tests/table_operation/test_multitable_ref_counter.py -s

pytest test/unit_tests/table_operation/test_async_count.py -s

pytest test/unit_tests/table_operation/test_determinism_mode.py -s

pytest test/unit_tests/table_operation/test_bucketize_segmented.py -s

pytest test/unit_tests/table_operation/test_sorted_insert.py -s

pytest test/unit_tests/table_operation/test_sorted_insert_hw.py -s

torchrun --nproc_per_node=1 -m pytest test/unit_tests/table_operation/test_table_dump_load.py -s
torchrun --nproc_per_node=8 -m pytest test/unit_tests/table_operation/test_table_dump_load.py -s
