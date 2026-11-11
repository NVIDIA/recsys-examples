#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-test for fep-flink-tools (Python 3 standard library only).

Runs qualification and profiling on the bundled synthetic fixtures, checks the
operator-family inventory contract, that no sensitive fixture string reaches the
outputs, and the read-only REST path against a local fixture server.
Usage: python3 tests/self_test.py  (outputs stay in a new temp directory; it is not deleted)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
TOOL = TOOLS_DIR / "fep-flink-tools"
FIXTURES = TOOLS_DIR / "fixtures"
SNAPSHOT = FIXTURES / "rest-snapshot.json"
EXPLAIN = FIXTURES / "explain.txt"
INVENTORY = TOOLS_DIR / "qualification" / "flink-2.2.1-operator-families.json"
SERVER = TOOLS_DIR / "tests" / "rest_fixture_server.py"
JOB_ID = "a" * 32
SECRETS = ("secret", "Acme", "password", "private-jdbc", "example.com")


def fail(message):
    print(f"self-test FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def check(condition, message):
    if not condition:
        fail(message)


def run(args, env=None, expect=0):
    proc = subprocess.run(
        [str(TOOL), *args], capture_output=True, text=True, env=env, timeout=120
    )
    check(
        proc.returncode == expect, f"{args[0]} exited {proc.returncode}: {proc.stderr}"
    )
    return proc.stdout.strip()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_no_secrets(directory):
    for path in Path(directory).iterdir():
        text = path.read_text(encoding="utf-8")
        for secret in SECRETS:
            check(secret not in text, f"{path.name} leaks fixture string {secret!r}")


def test_inventory():
    doc = load(INVENTORY)
    check(doc["schema"] == "fep-flink-operator-families/v1", "inventory schema")
    check(doc["flink_version"] == "2.2.1", "inventory Flink version")
    names = [f["name"] for f in doc["families"]]
    check(len(names) == len(set(names)), "duplicate family names")
    by_name = {f["name"]: f for f in doc["families"]}
    provided = {n for n, f in by_name.items() if f["coverage"] == "library-provided"}
    check(
        provided
        == {
            "WindowAggregate",
            "LocalWindowAggregate",
            "GlobalWindowAggregate",
            "LookupJoin",
        },
        f"library-provided families drifted: {sorted(provided)}",
    )
    for name, family in by_name.items():
        check(family["coverage"] in doc["coverage_values"], f"{name}: bad coverage")
        check(family["sql_physical_nodes"], f"{name}: no sql_physical_nodes")
        check(
            (family["library_api"] is not None) == (name in provided),
            f"{name}: library_api must be set exactly for library-provided families",
        )


def test_offline(root):
    out = run(
        [
            "qualification",
            "--snapshot-file",
            SNAPSHOT,
            "--explain-file",
            EXPLAIN,
            "--output-dir",
            root / "qual",
        ]
    )
    check(out.startswith("FEP_FLINK_QUALIFICATION_OK priority=high"), out)
    qual = load(root / "qual" / "qualification.json")
    check(qual["schema"] == "fep-flink-workload-analysis/v1", "analysis schema")
    check(qual["candidate_compute_share"] == 1, "candidate share")
    families = {r["family"]: r for r in qual["family_breakdown"]}
    check(families["LookupJoin"]["library_api"] == "GpuJoins.lookupInner", "lookup API")
    check(
        families["WindowAggregate"]["coverage"] == "library-provided", "window coverage"
    )
    check("speedup_prediction" not in qual, "speedup prediction must not exist")
    check_no_secrets(root / "qual")

    out = run(
        [
            "profiling",
            "--snapshot-file",
            SNAPSHOT,
            "--explain-file",
            EXPLAIN,
            "--output-dir",
            root / "prof",
        ]
    )
    check(out.startswith("FEP_FLINK_PROFILING_OK status=needs-attention"), out)
    prof = load(root / "prof" / "profiling.json")
    codes = {f["code"] for f in prof["findings"]}
    check({"high_backpressure", "checkpoint_health"} <= codes, f"findings: {codes}")
    check_no_secrets(root / "prof")

    # reuse an existing sanitized workload.json
    run(
        [
            "qualification",
            "--workload-file",
            root / "qual" / "workload.json",
            "--output-dir",
            root / "qual-reuse",
        ]
    )
    check(
        load(root / "qual-reuse" / "qualification.json") == qual,
        "workload reuse differs",
    )

    # removed subcommands must stay removed
    for command in ("compare", "calibrate", "train"):
        run([command], expect=2)


def test_rest(root):
    env = dict(
        os.environ,
        FIXTURE_AUTHORIZATION="Bearer self-test-token",
        TOOL_AUTHORIZATION="Bearer self-test-token",
        NO_PROXY="127.0.0.1",
        no_proxy="127.0.0.1",
    )
    port_file, request_log = root / "port", root / "requests.log"
    server = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-I",
            str(SERVER),
            "--snapshot",
            str(SNAPSHOT),
            "--port-file",
            str(port_file),
            "--request-log",
            str(request_log),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if port_file.exists() and port_file.read_text().strip():
                break
            time.sleep(0.05)
        else:
            fail("REST fixture server did not start")
        port = port_file.read_text().strip()
        out = run(
            [
                "profiling",
                "--rest-url",
                f"http://127.0.0.1:{port}",
                "--job-id",
                JOB_ID,
                "--authorization-env",
                "TOOL_AUTHORIZATION",
                "--sample-count",
                "2",
                "--sample-interval-seconds",
                "0.01",
                "--timeout-seconds",
                "5",
                "--output-dir",
                root / "rest",
            ],
            env=env,
        )
        check(out.startswith("FEP_FLINK_PROFILING_OK"), out)
        work = load(root / "rest" / "workload.json")
        check(work["collection"]["source"] == "flink-rest", "REST source")
        check(len(work["runtime_samples"]) == 2, "REST sample count")
        check(
            "self-test-token" not in (root / "rest" / "workload.json").read_text(),
            "credential retained",
        )
        check_no_secrets(root / "rest")
    finally:
        server.terminate()
        server.wait(timeout=10)
    methods = [
        line.split()[0] for line in request_log.read_text().splitlines() if line.strip()
    ]
    check(
        methods and all(m == "GET" for m in methods),
        f"non-GET request seen: {set(methods)}",
    )


def main():
    root = Path(tempfile.mkdtemp(prefix="fep-flink-tools-test-"))
    test_inventory()
    test_offline(root)
    test_rest(root)
    print(f"outputs kept in {root}")
    print("fep-flink-tools self-test: OK")


if __name__ == "__main__":
    main()
