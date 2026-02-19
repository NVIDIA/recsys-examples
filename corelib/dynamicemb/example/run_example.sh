#!/bin/bash

# torchrun --standalone --nproc_per_node=${NGPU} example.py --train "$@"
# torchrun --standalone --nproc_per_node=${NGPU} example.py --train --caching --prefetch_pipeline "$@"
# torchrun --standalone --nproc_per_node=${NGPU} example.py --incremental_dump "$@"
# torchrun --standalone --nproc_per_node=${NGPU} example.py --dump "$@"
# torchrun --standalone --nproc_per_node=${NGPU} example.py --eval "$@"
# torchrun --standalone --nproc_per_node=${NGPU} example.py --inference "$@"
torchrun --standalone --nproc_per_node=${NGPU} example.py --export "$@"

# torchrun --standalone --nproc_per_node=${NGPU} example.py --export "$@"
# torchrun --standalone --nproc_per_node=${NGPU} example.py --load "$@"