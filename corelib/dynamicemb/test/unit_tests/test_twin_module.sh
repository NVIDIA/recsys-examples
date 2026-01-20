set -e

torchrun --nproc_per_node=1 -m pytest -svv test/unit_tests/test_twin_module.py
torchrun --nproc_per_node=8 -m pytest -svv test/unit_tests/test_twin_module.py
