#!/bin/bash
set -e

MAX_GPUS=${DYNAMICEMB_NUM_GPUS:-2}

pytest test/unit_tests/table_operation/test_table_operation.py -s

torchrun --nproc_per_node=1 -m pytest test/unit_tests/table_operation/test_table_dump_load.py -s
torchrun --nproc_per_node=${MAX_GPUS} -m pytest test/unit_tests/table_operation/test_table_dump_load.py -s