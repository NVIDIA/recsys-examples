#!/bin/bash
# Runs demo.py on one GPU (Spark local mode) with the plugin. Usage: examples/run-demo.sh [-p SLURM_PARTITION]
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$HERE/../scripts/run-spark.sh" "$@" spark-submit \
  --conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension \
  --conf spark.rapids.sql.fusion.mode=SAFE "$HERE/demo.py"
