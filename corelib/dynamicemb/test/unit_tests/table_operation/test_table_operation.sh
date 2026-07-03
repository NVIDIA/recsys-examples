#!/bin/bash 
set -e

pytest test/unit_tests/table_operation/test_table_operation.py -s

pytest test/unit_tests/table_operation/test_lru_lfu.py -s

torchrun --nproc_per_node=1 -m pytest test/unit_tests/table_operation/test_table_dump_load.py -s
torchrun --nproc_per_node=8 -m pytest test/unit_tests/table_operation/test_table_dump_load.py -s