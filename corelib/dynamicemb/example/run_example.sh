#!/bin/bash

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export LOCAL_WORLD_SIZE=1

python example.py