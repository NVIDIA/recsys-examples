#!/bin/bash
# One-click plugin test run: build, no-GPU self tests, then on one GPU: smoke battery (nullable data, non-null data, aggregates/grouping),
# Round stress, retry/split-OOM injection. Usage: scripts/test-all.sh [-p SLURM_PARTITION] [-t TIME]. Writes target/test-all.log.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; S=$FUSION_HOME/scripts
TIME=01:30:00
while getopts "p:t:" o; do case $o in p) export SLURM_PARTITION=$OPTARG;; t) TIME=$OPTARG;; esac; done
mkdir -p $FUSION_HOME/target; LOG=$FUSION_HOME/target/test-all.log; : > $LOG
$S/build.sh >> $LOG 2>&1 || { echo "BUILD FAILED (see $LOG)"; exit 1; }
$S/selftest.sh >> $LOG 2>&1 || { echo "SELFTEST FAILED"; tail -5 $LOG; exit 1; }
EXT="--conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension"
run() { # name, java opts, script
  echo "== $1" | tee -a $LOG
  $S/run-spark.sh -t $TIME spark-shell $EXT --conf "spark.driver.extraJavaOptions=$2" -i $3 < /dev/null 2>&1 | grep -aE "SUMMARY|MISMATCH|ROUNDSTRESS|RETRY|ERROR " | grep -v Inbox | cut -c1-220 | tee -a $LOG
}
[ -d $FUSION_WORK/data/perf_t ] || run "generate perf data (GenData)" "-Dperf.rows=${PERF_ROWS:-30000000}" $S/perf/GenData.scala
run "battery nullable data"   "" $S/smoke/Battery.scala
run "battery non-null data"   "-Dbattery.nonull=true" $S/smoke/Battery.scala
run "battery aggregates/grouping" "-Dbattery.exprs=$S/smoke/exprs_agg.txt" $S/smoke/Battery.scala
run "round/bround stress"     "-Drs.n=500000" $S/smoke/RoundStress.scala
run "retry / split-and-retry injection" "" $S/smoke/RetryStress.scala
echo "DONE: look for MISMATCH lines and 'fusedVsUnfusedMismatch=0' / 'mismatches=0' / 'equal=true' above" | tee -a $LOG
