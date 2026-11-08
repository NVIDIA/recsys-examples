#!/bin/bash
# Standalone-cluster smoke test on ONE GPU machine: master + 1 worker with a GPU resource + 2 executors via spark-shell.
# Run it on the GPU machine itself (e.g. inside an allocation). Usage: cluster-test.sh -i script.scala
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; TC=$FUSION_WORK
export PATH=$JAVA_HOME/bin:$PATH
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
export LD_LIBRARY_PATH=$NVRTC_DIR:$LD_LIBRARY_PATH
HOST=$(hostname)
mkdir -p $TC/scratch
export SPARK_LOG_DIR=$TC/scratch/cluster-logs SPARK_WORKER_DIR=$TC/scratch/cluster-work SPARK_PID_DIR=$TC/scratch/cluster-pids SPARK_LOCAL_IP=$HOST
mkdir -p $SPARK_LOG_DIR $SPARK_WORKER_DIR $SPARK_PID_DIR
cat > $TC/scratch/gpu_discover.sh <<'EOS'
#!/bin/bash
echo '{"name": "gpu", "addresses": ["0"]}'
EOS
chmod +x $TC/scratch/gpu_discover.sh
$SPARK_HOME/sbin/start-master.sh -h $HOST -p 7177 --webui-port 8180 > /dev/null
sleep 5
SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 -Dspark.worker.resource.gpu.discoveryScript=$TC/scratch/gpu_discover.sh" \
  $SPARK_HOME/sbin/start-worker.sh spark://$HOST:7177 -c 4 -m 24g --webui-port 8181 > /dev/null
sleep 8
JARS=$RAPIDS_JAR,$FUSION_JAR
$SPARK_HOME/bin/spark-shell --master spark://$HOST:7177 --total-executor-cores 4 --executor-cores 2 --executor-memory 6g --driver-memory 4g \
  --conf spark.executor.resource.gpu.amount=1 --conf spark.task.resource.gpu.amount=0.5 \
  --conf spark.executor.resource.gpu.discoveryScript=$TC/scratch/gpu_discover.sh --conf spark.worker.resource.gpu.discoveryScript=$TC/scratch/gpu_discover.sh \
  --jars $JARS --conf spark.plugins=com.nvidia.spark.SQLPlugin --conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension \
  --conf spark.rapids.memory.gpu.allocFraction=0.3 --conf spark.rapids.memory.gpu.maxAllocFraction=0.3 --conf spark.rapids.memory.pinnedPool.size=512m \
  --conf spark.executorEnv.LD_LIBRARY_PATH=$NVRTC_DIR --conf spark.ui.enabled=false "$@"
RC=$?
$SPARK_HOME/sbin/stop-worker.sh > /dev/null 2>&1; $SPARK_HOME/sbin/stop-master.sh > /dev/null 2>&1
exit $RC
