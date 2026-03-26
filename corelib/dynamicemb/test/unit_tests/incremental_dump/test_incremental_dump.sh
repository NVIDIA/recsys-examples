
set -e

MAX_GPUS=${DYNAMICEMB_NUM_GPUS:-2}
MASTER_PORT=${DYNAMICEMB_MASTER_PORT:-29604}

pytest test/unit_tests/incremental_dump/test_dynamicemb_extensions.py -s
pytest test/unit_tests/incremental_dump/test_batched_dynamicemb_tables.py -s
torchrun --nproc_per_node=1 --master-port ${MASTER_PORT} -m pytest test/unit_tests/incremental_dump/test_distributed_dynamicemb.py -s
torchrun --nproc_per_node=${MAX_GPUS} --master-port ${MASTER_PORT} -m pytest test/unit_tests/incremental_dump/test_distributed_dynamicemb.py -s
