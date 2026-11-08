#!/bin/bash
# One-click differential test: CPU vs spark-rapids (unfused) vs fused, on one GPU.
#   tests/run-all.sh [-p SLURM_PARTITION] [-c FUZZ_CASES(150)] [-s SEED(1)] [-o OUTDIR(tests/out/run-<time>)] [-n]   (-n: skip fused mode)
# Needs: SPARK_HOME, RAPIDS_JAR (see scripts/env.sh), python3 with pyarrow+numpy (or PYLIB_DIR) and, for fused mode, target/gpu-fusion-plugin-*.jar (built by scripts/build.sh).
# Result: summary table on stdout; failing cases are auto-minimised and saved under OUTDIR/fuzz/failures/*.json; exit code 1 if fused != unfused GPU.
HERE=$(cd $(dirname $0) && pwd); ROOT=$HERE/..
source $ROOT/scripts/env.sh
PART=${SLURM_PARTITION:-}; CASES=150; SEED=1; OUT=$HERE/out/run-$(date +%m%d-%H%M); MODES=cpu,gpu,fused
while getopts "p:c:s:o:n" o; do case $o in p) PART=$OPTARG;; c) CASES=$OPTARG;; s) SEED=$OPTARG;; o) OUT=$OPTARG;; n) MODES=cpu,gpu;; esac; done
mkdir -p $OUT; cd $HERE
DATA=$FUSION_WORK/data/fz1m
[ -d $DATA ] || { echo "[run-all] generating fuzz data"; PYTHONPATH= python3 fz/gen_data.py --rows 1000000 --out $DATA > $OUT/gen.log 2>&1; }
# snapshot the plugin jar so that a rebuild during the run cannot break it
J=$FUSION_JAR
[ -n "$J" ] && cp $J $OUT/fusion.jar && export FUSION_JAR=$OUT/fusion.jar
[ -z "$J" ] && MODES=cpu,gpu
echo "[run-all] partition=$PART modes=$MODES cases=$CASES seed=$SEED out=$OUT jar=${J:-none}"
# all steps in ONE allocation
./gpu-run.sh ${PART:+-p $PART} -t 03:00:00 -- bash -c "
  python3 fz/directed.py --out $OUT/directed --modes $MODES > $OUT/directed.log 2>&1;
  fz/fuzz_loop.sh --data $DATA --out $OUT/fuzz --cases $CASES --seed $SEED --modes $MODES --time-limit 5400 > $OUT/fuzz.log 2>&1;
  [ -f udf/udfs.jar ] || udf/build.sh > $OUT/udf-build.log 2>&1;
  python3 udf/coverage.py --data $DATA --out $OUT/coverage.json --modes $MODES > $OUT/coverage.log 2>&1"
PYTHONPATH=${PYLIB_DIR:-} python3 fz/report.py $OUT; rc=$?
echo "[run-all] logs in $OUT ; exit=$rc"; exit $rc
