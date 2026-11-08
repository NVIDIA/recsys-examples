"""Shared helpers for the differential / fuzz tests (PySpark from $SPARK_HOME, see scripts/env.sh).

Modes (all run inside ONE Spark session by toggling runtime confs, so a single GPU allocation is enough):
  cpu    : spark.rapids.sql.enabled=false                       (reference oracle)
  gpu    : rapids on, fusion off                                (spark-rapids unfused baseline)
  ast    : rapids on, projectAstEnabled=true, fusion off        (spark-rapids' own AST path)
  fused  : rapids on, fusion plugin on                          (the plugin under test)
Fusion conf keys come from env FUSION_ON_CONF / FUSION_OFF_CONF ("k=v,k=v"); defaults follow the design doc
(spark.rapids.sql.fusion.mode=SAFE / OFF).
"""
import glob
import os
import sys
import time

ROOT = os.environ.get("FUSION_HOME") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
TC = os.environ.get("TOOLCHAIN_DIR") or os.path.join(ROOT, ".toolchain")
# optional extra python libs dir (pyarrow/numpy) if they are not installed in python3 itself
PYLIB = os.environ.get("PYLIB_DIR", "")


def _kv(s):
    d = {}
    for p in [x for x in (s or "").split(",") if x.strip()]:
        k, v = p.split("=", 1)
        d[k.strip()] = v.strip()
    return d


FUSION_ON = _kv(
    os.environ.get(
        "FUSION_ON_CONF",
        "spark.rapids.sql.fusion.mode=SAFE,spark.rapids.sql.fusion.force=true",
    )
)
FUSION_OFF = _kv(os.environ.get("FUSION_OFF_CONF", "spark.rapids.sql.fusion.mode=OFF"))
# physical-plan node names used to verify that the intended path was really taken
GPU_PROJECT = os.environ.get("GPU_PROJECT_NAME", "GpuProject")
FUSED_PROJECT = os.environ.get("FUSED_PROJECT_NAME", "FusedGpuProject")
GPU_FILTER = "GpuFilter"
FUSED_FILTER = os.environ.get("FUSED_FILTER_NAME", "FusedGpuFilter")


def setup_env():
    """Make pyspark importable (Spark ships its own python dir) + pyarrow/numpy from $PYLIB_DIR (optional)."""
    sh = os.environ.setdefault(
        "SPARK_HOME", os.path.join(TC, "spark-3.5.9-bin-hadoop3")
    )
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")
    for p in [
        os.path.join(sh, "python"),
        glob.glob(os.path.join(sh, "python/lib/py4j-*-src.zip"))[0],
    ] + ([PYLIB] if PYLIB else []):
        if p not in sys.path:
            sys.path.insert(0, p)
    if PYLIB:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [PYLIB, os.environ.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
    for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        os.environ.pop(k, None)


setup_env()
from pyspark.sql import SparkSession  # noqa: E402


def build_session(
    use_gpu=True,
    local_n=4,
    driver_mem="24g",
    extra=None,
    app="fusion-test",
    aqe=False,
    tz="UTC",
):
    b = (
        SparkSession.builder.master(f"local[{local_n}]")
        .appName(app)
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.session.timeZone", tz)
        .config("spark.sql.adaptive.enabled", "true" if aqe else "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config(
            "spark.local.dir",
            os.path.join(
                os.environ.get("FUSION_WORK") or os.path.join(ROOT, "work"),
                "spark-local",
            ),
        )
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .config(
            "spark.hadoop.parquet.enable.dictionary", "false"
        )  # dictionary pages merge 0.0/-0.0 (== keys): would corrupt the bit-exact transport
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
    )
    if use_gpu:
        jars = [j for j in os.environ.get("EXTRA_JARS", "").split(",") if j]
        jars += [
            j
            for j in [
                os.environ.get("RAPIDS_JAR")
                or os.path.join(TC, "rapids-4-spark_2.12-26.08.1-cuda12.jar"),
                os.environ.get("FUSION_JAR") or "",
            ]
            if j
        ]
        fj = os.environ.get("FUSION_JAR")
        if not fj:
            c = sorted(
                glob.glob(os.path.join(ROOT, "target/gpu-fusion-plugin*.jar")),
                key=os.path.getmtime,
            )
            c = [x for x in c if "sources" not in x and "original" not in x]
            if c:
                jars.append(c[-1])
        jl = ",".join(dict.fromkeys(j for j in jars if j))
        b = (
            b.config("spark.jars", jl)
            .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.rapids.sql.enabled", "true")
            .config(
                "spark.rapids.sql.format.parquet.write.enabled", "false"
            )  # results are always written by the CPU writer
            .config(
                "spark.rapids.memory.gpu.allocFraction",
                os.environ.get("RAPIDS_POOL_FRAC", "0.4"),
            )
            .config(
                "spark.rapids.memory.gpu.maxAllocFraction",
                os.environ.get("RAPIDS_POOL_FRAC", "0.4"),
            )
            .config("spark.rapids.memory.pinnedPool.size", "1g")
            .config("spark.rapids.sql.concurrentGpuTasks", "2")
            .config("spark.rapids.sql.explain", "NONE")
        )
        ext = os.environ.get(
            "FUSION_EXTENSIONS",
            "com.nvidia.spark.rapids.fusion.FusionExtension"
            if any("gpu-fusion-plugin" in j for j in jars)
            else "",
        )
        if ext:
            b = b.config("spark.sql.extensions", ext)
    for k, v in (extra or {}).items():
        b = b.config(k, v)
    s = b.getOrCreate()
    s.sparkContext.setLogLevel("OFF")
    return s


def mode_confs(mode):
    if mode == "cpu":
        return {"spark.rapids.sql.enabled": "false", "spark.sql.ansi.enabled": "false"}
    if (
        mode == "cpuansi"
    ):  # self-test of the harness only: ANSI differs from non-ANSI on purpose
        return {"spark.rapids.sql.enabled": "false", "spark.sql.ansi.enabled": "true"}
    base = {
        "spark.rapids.sql.enabled": "true",
        "spark.rapids.sql.projectAstEnabled": "false",
        **FUSION_OFF,
    }
    if mode == "gpu":
        return base
    if mode == "ast":
        return {**base, "spark.rapids.sql.projectAstEnabled": "true"}
    if mode == "fused":
        return {**base, **FUSION_ON}
    raise ValueError(mode)


def set_mode(spark, mode):
    for k, v in mode_confs(mode).items():
        spark.conf.set(k, v)


def plan_str(df):
    return df._jdf.queryExecution().executedPlan().toString()


def plan_flags(df):
    import re

    p = plan_str(df)

    def has(name):
        return re.search(r"(?<![A-Za-z])" + re.escape(name), p) is not None

    return {
        "gpu_project": has(GPU_PROJECT),
        "fused_project": has(FUSED_PROJECT),
        "gpu_filter": has(GPU_FILTER),
        "fused_filter": has(FUSED_FILTER),
    }


# FUSION_ASYNC_WARM=1 (used with FUSION_ON_CONF containing spark.rapids.sql.fusion.asyncCompile=true): a fused query is executed twice,
# the first run (cold: new regions fall back to the unfused operator while NVRTC compiles in the background), then wait for the background
# compiles, then the second run (really fused). Both results must be identical; the second one is returned for the usual 3-way comparison.
ASYNC_WARM = os.environ.get("FUSION_ASYNC_WARM", "") == "1"


def run_with_async_warm(spark, mode, do_run, same):
    t = do_run()
    if mode == "fused" and ASYNC_WARM:
        ok = spark._jvm.com.nvidia.spark.rapids.fusion.NativeBridge.awaitCompiles(60000)
        t2 = do_run()
        if not ok or not same(t, t2):
            raise RuntimeError(
                "ASYNC COLD/WARM MISMATCH or background compile did not finish"
            )
        return t2
    return t


def now():
    return time.strftime("%H:%M:%S")
