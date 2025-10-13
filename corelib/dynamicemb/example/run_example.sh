#!/bin/bash

torchrun --standalone --nproc_per_node=${NGPU} example.py --train "$@"
torchrun --standalone --nproc_per_node=${NGPU} example.py --train --caching --prefetch_pipeline "$@"
torchrun --standalone --nproc_per_node=${NGPU} example.py --load --dump "$@"
torchrun --standalone --nproc_per_node=${NGPU} example.py --incremental_dump "$@"