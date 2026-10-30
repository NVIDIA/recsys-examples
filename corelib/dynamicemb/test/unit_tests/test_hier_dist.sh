#!/bin/bash
# test_hier_part1.py and test_hier_dedup.py on one GPU, then test_hier_dist.py on 2, 4, 6 and 8 GPUs.
set -e
if [ "$(python -c 'import dynamicemb_extensions as ext; print(hasattr(ext, "HierOutputDist"))')" = False ]; then
  echo 'this build has no hierarchical dist: skipped'
  exit 0
fi

python ./test/unit_tests/test_hier_part1.py
python ./test/unit_tests/test_hier_dedup.py

NUM_GPUS=$(nvidia-smi -L | wc -l)
for nproc in 2 4 6 8; do
  [ "${nproc}" -le "${NUM_GPUS}" ] || continue
  echo "=== ${nproc} GPU(s) ==="
  torchrun --nnodes 1 --nproc_per_node "${nproc}" \
    ./test/unit_tests/test_hier_dist.py || exit 1
done
