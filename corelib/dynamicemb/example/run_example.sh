#!/bin/bash

# The 2D example supports full checkpoints and uncached HBM tables.
for arg in "$@"; do
    case "$arg" in
        --sharding_group_size|--sharding_group_size=*)
            exec torchrun --standalone --nproc_per_node="${NGPU}" example.py --train --dump --load "$@"
            ;;
    esac
done

torchrun --standalone --nproc_per_node=${NGPU} example.py --train "$@"
torchrun --standalone --nproc_per_node=${NGPU} example.py --train --caching --prefetch_pipeline "$@"
torchrun --standalone --nproc_per_node=${NGPU} example.py --load --dump "$@"
torchrun --standalone --nproc_per_node=${NGPU} example.py --incremental_dump "$@"
torchrun --standalone --nproc_per_node=${NGPU} example.py --incremental_load "$@"
