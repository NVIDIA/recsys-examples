set -e

MAX_GPUS=${DYNAMICEMB_NUM_GPUS:-2}
MASTER_PORT=${DYNAMICEMB_MASTER_PORT:-29603}

torchrun --nproc_per_node=1 --master-port ${MASTER_PORT} -m pytest -svv test/unit_tests/test_twin_module.py
torchrun --nproc_per_node=${MAX_GPUS} --master-port ${MASTER_PORT} -m pytest -svv test/unit_tests/test_twin_module.py
