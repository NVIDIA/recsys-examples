#!/bin/bash
# Full performance run on one GPU type: microbench, e2e (parquet), E/N benchmark queries (bench/queries.py), no-IO e2e, operator breakdown.
# Usage: scripts/final-perf.sh <label> [SLURM_PARTITION]   -> $FUSION_WORK/perf-logs/<label>/
# Data: perf/GenData.scala (-> $FUSION_WORK/data/perf_t) and bench/gen_data.py (-> $FUSION_WORK/data/semdata) must have been run.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; S=$FUSION_HOME/scripts; P=$S/perf
LABEL=$1; [ -n "$2" ] && export SLURM_PARTITION=$2
OUT=$FUSION_WORK/perf-logs/$LABEL; mkdir -p $OUT
EXT="--conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension"
echo "jar: $(basename $FUSION_JAR) md5 $(md5sum < $FUSION_JAR | cut -c1-8)" > $OUT/info.txt; date >> $OUT/info.txt
$S/run-java.sh -t 00:15:00 com.nvidia.spark.rapids.fusion.MicroBench 20000000 15 > $OUT/microbench.log 2>&1
$S/run-spark.sh -t 01:00:00 spark-shell $EXT -i $P/E2E.scala < /dev/null 2>&1 | grep -a "E2E" > $OUT/e2e_parquet.log
$S/run-spark.sh -t 01:30:00 spark-shell $EXT --conf "spark.driver.extraJavaOptions=-Dsb.reps=5" -i $P/SemBench.scala < /dev/null 2>&1 | grep -aE "^SB" > $OUT/sem_bench.log
$S/run-spark.sh -t 01:00:00 spark-shell $EXT --conf "spark.driver.extraJavaOptions=-Dnoio.rows=300000000" -i $P/NoIO.scala < /dev/null 2>&1 | grep -a "^NOIO" > $OUT/noio.log
$S/run-spark.sh -t 00:30:00 spark-shell $EXT --conf "spark.driver.extraJavaOptions=-Dsb.queries=E1,E3,E5,E6" -i $P/SemBreakdown.scala < /dev/null 2>&1 | grep -a "SBD" | sed 's/.*SBD/SBD/' > $OUT/breakdown.log
date >> $OUT/info.txt; echo done > $OUT/DONE
