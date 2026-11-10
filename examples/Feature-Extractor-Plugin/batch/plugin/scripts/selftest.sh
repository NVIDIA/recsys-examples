#!/bin/bash
# No-GPU self tests: (1) planner/translator/codegen unit checks, (2) every operator x type kernel compiles with NVRTC for sm_75..sm_120.
# Needs libnvrtc.so.12 (NVRTC_DIR) but no GPU.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export NOSRUN=1 EXTRA_CP=$RAPIDS_SHIM_JAR
"$FUSION_HOME/scripts/run-java.sh" com.nvidia.spark.rapids.fusion.PlannerSelfTest || exit 1
"$FUSION_HOME/scripts/run-java.sh" com.nvidia.spark.rapids.fusion.CodegenSelfTest
