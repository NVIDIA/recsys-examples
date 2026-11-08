#!/bin/bash
# Run a command on one GPU: via srun when SLURM_PARTITION (or -p) is set, otherwise directly on this machine.
# Usage: tests/gpu-run.sh [-p SLURM_PARTITION] [-t TIME(01:00:00)] [-m MEM(64G)] -- cmd args...
# On a Slurm cluster, scripts and data must live on a file system shared with the compute nodes (not /tmp).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/env.sh"
TIME=01:00:00
while getopts "p:t:m:" o; do case $o in p) export SLURM_PARTITION=$OPTARG;; t) TIME=$OPTARG;; m) export MEM=$OPTARG;; esac; done
shift $((OPTIND-1)); [ "$1" = "--" ] && shift
export LD_LIBRARY_PATH=$NVRTC_DIR:$CUDA_HOME/lib64:$LD_LIBRARY_PATH PATH=$JAVA_HOME/bin:$PATH
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
gpu_run $TIME "$@"
