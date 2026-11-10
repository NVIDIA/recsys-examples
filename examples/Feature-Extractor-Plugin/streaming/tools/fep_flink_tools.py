#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified qualification and profiling CLI for sanitized Flink workloads."""

import argparse
import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path


def load_core():
    path = Path(__file__).with_name("fep_flink_collector.py")
    spec = importlib.util.spec_from_file_location("fep_flink_collector_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the collector core")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_core()


DEFAULT_CAPABILITIES = str(
    Path(__file__).resolve().parent / "qualification/flink-2.2.1-operator-families.json"
)

SANITIZED_STATISTICS = {"value", "min", "max", "avg", "sum"}
LIFETIME_METRICS = {
    "accumulatedBackpressuredTimeMs",
    "accumulatedBusyTimeMs",
    "accumulatedIdleTimeMs",
    "numBytesIn",
    "numBytesOut",
    "numRecordsIn",
    "numRecordsOut",
}
BACKPRESSURE_RATIOS = {
    "backpressureratio",
    "busyratio",
    "hardbackpressureratio",
    "idleratio",
    "softbackpressureratio",
}


def add_workload_options(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--rest-url", help="Flink JobManager or History Server base URL"
    )
    source.add_argument(
        "--snapshot-file",
        help=f"offline raw REST snapshot using schema {CORE.SNAPSHOT_SCHEMA}",
    )
    source.add_argument(
        "--archive-file",
        help="raw job archive written by Flink's FsJobArchivist for History Server",
    )
    source.add_argument(
        "--workload-file",
        help="existing sanitized workload.json produced by this tool",
    )
    parser.add_argument("--job-id", help="32-character hexadecimal Flink job id")
    parser.add_argument(
        "--archive-flink-version",
        help="Flink version that wrote --archive-file; currently must be 2.2.1",
    )
    parser.add_argument(
        "--output-dir", required=True, help="new directory for analysis output"
    )
    parser.add_argument(
        "--capabilities",
        default=DEFAULT_CAPABILITIES,
        help="family-level operator inventory",
    )
    parser.add_argument(
        "--explain-file",
        help="optional Flink EXPLAIN text/JSON; raw text is never retained",
    )
    parser.add_argument("--sample-count", type=CORE.positive_int, default=1)
    parser.add_argument(
        "--sample-interval-seconds",
        type=CORE.positive_float,
        default=CORE.DEFAULT_SAMPLE_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=CORE.positive_float,
        default=CORE.DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--startup-wait-seconds",
        type=CORE.non_negative_float,
        default=CORE.DEFAULT_STARTUP_WAIT_SECONDS,
        help="wait for newly submitted vertices to finish deployment before sampling",
    )
    parser.add_argument(
        "--authorization-env",
        help="environment variable containing the complete Authorization header",
    )
    parser.add_argument(
        "--pseudonym-key-env",
        help="environment variable containing a stable pseudonymization secret",
    )
    parser.add_argument("--ca-file", help="custom CA bundle for an HTTPS REST endpoint")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="fep-flink-tools",
        description=(
            "Qualify a Flink SQL workload for the FEP GPU operator library or profile its sanitized "
            "runtime bottlenecks. Live collection performs HTTP GET requests only."
        ),
    )
    parser.add_argument("--version", action="version", version=CORE.COLLECTOR_VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    qualification = commands.add_parser(
        "qualification",
        help="decide whether a workload merits a controlled CPU/GPU comparison",
    )
    add_workload_options(qualification)
    profiling = commands.add_parser(
        "profiling", help="rank vertices and diagnose runtime bottleneck signals"
    )
    add_workload_options(profiling)
    return parser.parse_args()


def capability_sha256(capability_doc):
    return hashlib.sha256(
        CORE.canonical_json(capability_doc).encode("utf-8")
    ).hexdigest()


def validate_sanitized_metrics(raw_metrics, allowed_names, label):
    if not isinstance(raw_metrics, dict) or any(
        name not in allowed_names for name in raw_metrics
    ):
        CORE.fail(f"--workload-file contains invalid {label} metric names")
    for statistics in raw_metrics.values():
        if (
            not isinstance(statistics, dict)
            or not statistics
            or any(key not in SANITIZED_STATISTICS for key in statistics)
        ):
            CORE.fail(f"--workload-file contains invalid {label} statistics")
        for value in statistics.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or CORE.finite_number(value) is None
                or value < 0
            ):
                CORE.fail(f"--workload-file contains invalid {label} metric values")


def validate_sanitized_backpressure(raw_backpressure):
    if not isinstance(raw_backpressure, dict) or any(
        key not in {"level", "status", "ratios"} for key in raw_backpressure
    ):
        CORE.fail("--workload-file contains invalid backpressure evidence")
    if raw_backpressure.get("level") not in {None, "ok", "low", "high"}:
        CORE.fail("--workload-file contains an invalid backpressure level")
    if raw_backpressure.get("status") not in {None, "ok", "deprecated"}:
        CORE.fail("--workload-file contains an invalid backpressure status")
    ratios = raw_backpressure.get("ratios", {})
    if not isinstance(ratios, dict) or any(
        name not in BACKPRESSURE_RATIOS for name in ratios
    ):
        CORE.fail("--workload-file contains invalid backpressure ratios")
    for statistics in ratios.values():
        if not isinstance(statistics, dict) or set(statistics) != {"min", "max", "avg"}:
            CORE.fail("--workload-file contains invalid backpressure ratio statistics")
        for value in statistics.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or CORE.finite_number(value) is None
                or not 0 <= value <= 1
            ):
                CORE.fail("--workload-file contains invalid backpressure ratio values")


def validate_sanitized_bundle(bundle, capability_doc, family_specs):
    if not isinstance(bundle, dict) or bundle.get("schema") != CORE.BUNDLE_SCHEMA:
        CORE.fail(f"--workload-file must use schema {CORE.BUNDLE_SCHEMA}")
    collection = bundle.get("collection")
    privacy = bundle.get("privacy")
    if (
        not isinstance(collection, dict)
        or collection.get("raw_rest_payloads_retained") is not False
    ):
        CORE.fail("--workload-file does not declare sanitized REST payloads")
    if (
        not isinstance(privacy, dict)
        or not privacy
        or any(value is not False for value in privacy.values())
    ):
        CORE.fail("--workload-file does not satisfy the privacy contract")
    job = bundle.get("job")
    plan = bundle.get("plan")
    samples = bundle.get("runtime_samples")
    if not isinstance(job, dict) or not isinstance(job.get("vertices"), list):
        CORE.fail("--workload-file is missing sanitized job vertices")
    if not isinstance(plan, dict) or not isinstance(samples, list):
        CORE.fail("--workload-file is missing plan or runtime samples")
    if len(job["vertices"]) > CORE.MAX_VERTICES or len(samples) > CORE.MAX_SAMPLES:
        CORE.fail("--workload-file exceeds a safety limit")
    raw_version = bundle.get("collector_version")
    if (
        not isinstance(raw_version, str)
        or CORE.SAFE_VERSION_RE.fullmatch(raw_version) is None
    ):
        CORE.fail("--workload-file has an invalid collector version")
    if collection.get("source") not in {
        "flink-rest",
        "flink-history-archive",
        "offline-rest-snapshot",
    }:
        CORE.fail("--workload-file has an invalid collection source")
    if not isinstance(collection.get("pseudonyms_stable_across_runs"), bool):
        CORE.fail("--workload-file has invalid pseudonym metadata")
    cluster = bundle.get("cluster")
    if not isinstance(cluster, dict):
        CORE.fail("--workload-file has invalid cluster metadata")
    flink_version = cluster.get("flink_version")
    if flink_version is not None and (
        not isinstance(flink_version, str)
        or CORE.SAFE_VERSION_RE.fullmatch(flink_version) is None
    ):
        CORE.fail("--workload-file has an invalid Flink version")
    allowed_states = CORE.SAFE_JOB_STATES.union({"UNKNOWN"})
    if job.get("state") not in allowed_states:
        CORE.fail("--workload-file has an invalid job state")
    capability = bundle.get("capability_inventory")
    expected_capability = {
        "schema": capability_doc["schema"],
        "flink_version": capability_doc["flink_version"],
        "family_count": len(family_specs),
        "sha256": capability_sha256(capability_doc),
    }
    if not isinstance(capability, dict) or any(
        capability.get(name) != value for name, value in expected_capability.items()
    ):
        CORE.fail("--workload-file does not match the current capability inventory")
    allowed_families = {spec["name"] for spec in family_specs}.union({"Unknown"})
    plan_nodes = plan.get("nodes")
    family_counts = plan.get("family_counts")
    if not isinstance(plan_nodes, list) or len(plan_nodes) > CORE.MAX_PLAN_NODES:
        CORE.fail("--workload-file has an invalid sanitized plan")
    if not isinstance(family_counts, dict) or any(
        family not in allowed_families
        or (parsed_count := CORE.non_negative_int(count)) is None
        or parsed_count > CORE.MAX_PLAN_NODES
        for family, count in family_counts.items()
    ):
        CORE.fail("--workload-file has invalid plan family counts")
    seen_ids = set()
    parallelism_by_id = {}
    safe_vertices = []
    for vertex in job["vertices"]:
        if not isinstance(vertex, dict):
            CORE.fail("--workload-file contains an invalid vertex")
        vertex_id = vertex.get("id")
        families = vertex.get("families")
        if (
            not isinstance(vertex_id, str)
            or re.fullmatch(r"vertex_[0-9]{3,8}", vertex_id) is None
            or vertex_id in seen_ids
        ):
            CORE.fail("--workload-file contains an invalid synthetic vertex id")
        if (
            not isinstance(families, list)
            or not families
            or any(
                not isinstance(family, str) or family not in allowed_families
                for family in families
            )
            or len(families) != len(set(families))
        ):
            CORE.fail("--workload-file contains an invalid operator family")
        raw_parallelism = vertex.get("parallelism")
        parallelism = (
            None if raw_parallelism is None else CORE.non_negative_int(raw_parallelism)
        )
        if parallelism is not None and not 1 <= parallelism <= 100_000:
            CORE.fail("--workload-file contains invalid vertex parallelism")
        if raw_parallelism is not None and parallelism is None:
            CORE.fail("--workload-file contains invalid vertex parallelism")
        if vertex.get("state") not in allowed_states:
            CORE.fail("--workload-file contains an invalid vertex state")
        duration = vertex.get("duration_ms")
        if duration is not None and CORE.non_negative_int(duration) is None:
            CORE.fail("--workload-file contains an invalid vertex duration")
        validate_sanitized_metrics(
            vertex.get("lifetime_metrics", {}),
            LIFETIME_METRICS,
            "lifetime",
        )
        safe_vertex = dict(vertex)
        safe_vertex["state"] = CORE.safe_state(vertex["state"])
        safe_vertex["parallelism"] = parallelism
        safe_vertices.append(safe_vertex)
        parallelism_by_id[vertex_id] = parallelism
        seen_ids.add(vertex_id)

    previous_offset = -1
    for sample in samples:
        if not isinstance(sample, dict):
            CORE.fail("--workload-file contains an invalid runtime sample")
        offset = CORE.non_negative_int(sample.get("offset_ms"))
        sample_vertices = sample.get("vertices")
        if (
            offset is None
            or offset < previous_offset
            or not isinstance(sample_vertices, list)
            or len(sample_vertices) > CORE.MAX_VERTICES
        ):
            CORE.fail("--workload-file contains an invalid runtime sample")
        previous_offset = offset
        sample_ids = set()
        for sample_vertex in sample_vertices:
            if not isinstance(sample_vertex, dict):
                CORE.fail("--workload-file contains an invalid runtime vertex")
            vertex_id = sample_vertex.get("id")
            if (
                not isinstance(vertex_id, str)
                or vertex_id not in seen_ids
                or vertex_id in sample_ids
            ):
                CORE.fail("--workload-file contains an invalid runtime vertex id")
            sample_ids.add(vertex_id)
            validate_sanitized_metrics(
                sample_vertex.get("metrics", {}),
                set(CORE.RUNTIME_METRICS),
                "runtime",
            )
            validate_sanitized_backpressure(sample_vertex.get("backpressure", {}))
            subtasks = sample_vertex.get("subtasks", [])
            if (
                not isinstance(subtasks, list)
                or (
                    parallelism_by_id[vertex_id] is not None
                    and len(subtasks) > parallelism_by_id[vertex_id]
                )
                or len(subtasks) > CORE.MAX_FALLBACK_SUBTASKS_TOTAL
            ):
                CORE.fail("--workload-file contains invalid subtask evidence")
            subtask_indexes = set()
            for subtask in subtasks:
                if not isinstance(subtask, dict):
                    CORE.fail("--workload-file contains invalid subtask evidence")
                subtask_index = CORE.non_negative_int(subtask.get("index"))
                if (
                    subtask_index is None
                    or (
                        parallelism_by_id[vertex_id] is not None
                        and subtask_index >= parallelism_by_id[vertex_id]
                    )
                    or subtask_index in subtask_indexes
                ):
                    CORE.fail("--workload-file contains an invalid subtask index")
                subtask_indexes.add(subtask_index)
                validate_sanitized_metrics(
                    subtask.get("metrics", {}),
                    set(CORE.RUNTIME_METRICS),
                    "subtask",
                )
    declared_sample_count = CORE.non_negative_int(collection.get("sample_count"))
    if declared_sample_count != len(samples):
        CORE.fail("--workload-file sample count does not match its runtime samples")
    safe_bundle = dict(bundle)
    safe_bundle["job"] = dict(job)
    safe_bundle["job"]["vertices"] = safe_vertices
    if bundle.get("warnings"):
        safe_bundle["warnings"] = [
            "the input workload bundle reported missing collection evidence"
        ]
    else:
        safe_bundle["warnings"] = []
    return safe_bundle


def load_or_collect(args):
    capability_doc, family_specs = CORE.load_capabilities(args.capabilities)
    if args.workload_file:
        incompatible = {
            "--job-id": args.job_id,
            "--archive-flink-version": args.archive_flink_version,
            "--explain-file": args.explain_file,
            "--authorization-env": args.authorization_env,
            "--pseudonym-key-env": args.pseudonym_key_env,
            "--ca-file": args.ca_file,
        }
        used = [name for name, value in incompatible.items() if value is not None]
        if args.sample_count != 1:
            used.append("--sample-count")
        if args.sample_interval_seconds != CORE.DEFAULT_SAMPLE_INTERVAL_SECONDS:
            used.append("--sample-interval-seconds")
        if args.timeout_seconds != CORE.DEFAULT_TIMEOUT_SECONDS:
            used.append("--timeout-seconds")
        if args.startup_wait_seconds != CORE.DEFAULT_STARTUP_WAIT_SECONDS:
            used.append("--startup-wait-seconds")
        if used:
            CORE.fail(
                "--workload-file cannot be combined with collection options: "
                + ", ".join(sorted(used))
            )
        bundle = CORE.read_stable_json(args.workload_file, "sanitized workload")
        return (
            validate_sanitized_bundle(bundle, capability_doc, family_specs),
            family_specs,
            False,
        )
    bundle, collected_family_specs = CORE.collect_bundle(args)
    if collected_family_specs != family_specs:
        CORE.fail("capability inventory changed during collection")
    expected_hash = capability_sha256(capability_doc)
    if bundle.get("capability_inventory", {}).get("sha256") != expected_hash:
        CORE.fail("capability inventory identity changed during collection")
    return bundle, family_specs, True


def output_files(command, bundle, qualification, collected):
    files = {
        "qualification.json": CORE.json_text(qualification),
        "qualification.md": CORE.markdown_report(bundle, qualification),
    }
    if collected:
        files["workload.json"] = CORE.json_text(bundle)
    if command == "profiling":
        profile = CORE.analyze_profile(bundle, qualification)
        files["profiling.json"] = CORE.json_text(profile)
        files["profiling.md"] = CORE.profile_markdown_report(
            bundle, qualification, profile
        )
        return files, profile
    return files, None


def main():
    os.umask(0o077)
    args = parse_args()
    bundle, family_specs, collected = load_or_collect(args)
    qualification = CORE.analyze_bundle(bundle, family_specs)
    files, profile = output_files(args.command, bundle, qualification, collected)
    output = CORE.write_named_outputs(args.output_dir, files)
    if profile is None:
        print(
            "FEP_FLINK_QUALIFICATION_OK "
            f"priority={qualification['priority']} output_dir={output}"
        )
    else:
        print(
            "FEP_FLINK_PROFILING_OK "
            f"status={profile['status']} priority={qualification['priority']} "
            f"output_dir={output}"
        )


if __name__ == "__main__":
    try:
        main()
    except CORE.CollectorError as error:
        print(f"FEP_FLINK_TOOLS_FAILED: {error}", file=sys.stderr)
        sys.exit(1)
