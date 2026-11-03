#!/bin/bash
# test_hier_fallback.py on 8, 6 and 2 GPUs (6: a world size that is not a power of two).
set -e
if [ "$(python -c 'import dynamicemb_extensions as ext; print(hasattr(ext, "HierOutputDist"))')" = False ]; then
    echo 'this build has no hierarchical dist: skipped'
    exit 0
fi

NUM_GPUS=$(nvidia-smi -L | wc -l)
for nproc in 8 6 2; do
    if [ "$nproc" -gt "$NUM_GPUS" ]; then
        continue
    fi
    echo "=== ${nproc} GPU(s) ==="
    torchrun --nnodes 1 --nproc_per_node $nproc \
        ./test/unit_tests/test_hier_fallback.py || exit 1
done
