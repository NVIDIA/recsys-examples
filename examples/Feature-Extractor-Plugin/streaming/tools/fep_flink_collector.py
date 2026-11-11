#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect a shareable, pseudonymized Flink workload summary from read-only REST data."""

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import ssl
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

COLLECTOR_VERSION = "0.1.0"
BUNDLE_SCHEMA = "fep-flink-workload-bundle/v1"
ANALYSIS_SCHEMA = "fep-flink-workload-analysis/v1"
PROFILE_SCHEMA = "fep-flink-workload-profile/v1"
SNAPSHOT_SCHEMA = "flink-rest-snapshot/v1"
CAPABILITY_SCHEMA = "fep-flink-operator-families/v1"
DEFAULT_SAMPLE_INTERVAL_SECONDS = 10.0
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_STARTUP_WAIT_SECONDS = 30.0
UNKNOWN_COMPUTE_SHARE_THRESHOLD = 0.50
HIGH_CANDIDATE_SHARE_THRESHOLD = 0.60
MEDIUM_CANDIDATE_SHARE_THRESHOLD = 0.25
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_SAMPLES = 120
MAX_COLLECTION_SECONDS = 3600
MAX_VERTICES = 2048
MAX_PLAN_NODES = 10_000
MAX_ARCHIVE_ENTRIES = 10_000
MAX_METRIC_QUERY_CHARS = 768
MAX_METRIC_ID_CHARS = 512
MAX_SUBTASK_INDEX = 1_000_000
MAX_FALLBACK_SUBTASKS_PER_VERTEX = 256
MAX_FALLBACK_SUBTASKS_TOTAL = 2048
MAX_CHECKPOINT_INTERVAL_MS = 365 * 24 * 60 * 60 * 1000
JOB_ID_RE = re.compile(r"[0-9a-fA-F]{32}\Z")
ARCHIVE_JOB_PATH_RE = re.compile(r"/jobs/([0-9a-fA-F]{32})(?:/|\Z)")
ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
SAFE_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}\Z")
SAFE_VERSION_PREFIX_RE = re.compile(r"\A([0-9]+(?:\.[0-9]+){1,3})")

RUNTIME_METRICS = (
    "busyTimeMsPerSecond",
    "backPressuredTimeMsPerSecond",
    "softBackPressuredTimeMsPerSecond",
    "hardBackPressuredTimeMsPerSecond",
    "idleTimeMsPerSecond",
    "numRecordsIn",
    "numRecordsOut",
    "numRecordsInPerSecond",
    "numRecordsOutPerSecond",
    "numBytesIn",
    "numBytesOut",
    "pendingRecords",
)

COVERAGE_VALUES = {"library-provided", "not-provided", "cpu-boundary"}
COMPUTE_BOUNDARY_FAMILIES = {"Source", "Sink", "Exchange"}
TIME_PER_SECOND_METRICS = {
    "busyTimeMsPerSecond",
    "backPressuredTimeMsPerSecond",
    "softBackPressuredTimeMsPerSecond",
    "hardBackPressuredTimeMsPerSecond",
    "idleTimeMsPerSecond",
}
ADDITIVE_RUNTIME_METRICS = {
    "numRecordsIn",
    "numRecordsOut",
    "numRecordsInPerSecond",
    "numRecordsOutPerSecond",
    "numBytesIn",
    "numBytesOut",
    "pendingRecords",
}

DETAIL_METRIC_NAMES = {
    "read-records": "numRecordsIn",
    "write-records": "numRecordsOut",
    "read-bytes": "numBytesIn",
    "write-bytes": "numBytesOut",
    "accumulated-backpressured-time": "accumulatedBackpressuredTimeMs",
    "accumulated-busy-time": "accumulatedBusyTimeMs",
    "accumulated-idle-time": "accumulatedIdleTimeMs",
}

SAFE_CHECKPOINT_CONFIG = {
    "interval",
    "timeout",
    "min_pause",
    "max_concurrent",
    "mode",
    "tolerable_failed_checkpoints",
    "unaligned_checkpoints",
    "checkpoints_after_tasks_finish",
    "state_changelog_enabled",
}

SAFE_JOB_STATES = {
    "CREATED",
    "RUNNING",
    "FAILING",
    "FAILED",
    "CANCELLING",
    "CANCELED",
    "FINISHED",
    "RESTARTING",
    "SUSPENDED",
    "RECONCILING",
    "INITIALIZING",
}

SAFE_SHIP_STRATEGIES = {
    "ALL_TO_ALL",
    "BROADCAST",
    "FORWARD",
    "GLOBAL",
    "HASH",
    "PARTITION_HASH",
    "POINTWISE",
    "RANDOM",
    "REBALANCE",
    "RESCALE",
}

SAFE_EXECUTION_MODES = {
    "BATCH",
    "BATCH_FORCED",
    "PIPELINED",
    "PIPELINED_FORCED",
    "STREAMING",
}

SAFE_CHECKPOINT_MODES = {"AT_LEAST_ONCE", "EXACTLY_ONCE"}


class CollectorError(Exception):
    """Expected input, privacy, or collection failure."""


def fail(message):
    raise CollectorError(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if parsed.is_integer() and abs(parsed) <= 9_007_199_254_740_991:
        return int(parsed)
    return parsed


def non_negative_int(value):
    parsed = finite_number(value)
    if not isinstance(parsed, int) or parsed < 0:
        return None
    return parsed


def positive_int(text):
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0 or value > MAX_SAMPLES:
        raise argparse.ArgumentTypeError(f"must be in [1,{MAX_SAMPLES}]")
    return value


def positive_float(text):
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value) or value <= 0 or value > 300:
        raise argparse.ArgumentTypeError("must be in (0,300]")
    return value


def non_negative_float(text):
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value) or value < 0 or value > 300:
        raise argparse.ArgumentTypeError("must be in [0,300]")
    return value


def read_stable_bytes(path, label):
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        fail(f"cannot open {label}: {error.strerror or error.__class__.__name__}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(f"{label} is not a regular file")
        if before.st_size > MAX_INPUT_BYTES:
            fail(f"{label} exceeds the {MAX_INPUT_BYTES}-byte safety limit")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, MAX_INPUT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_INPUT_BYTES:
                fail(f"{label} exceeds the {MAX_INPUT_BYTES}-byte safety limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        fail(f"{label} changed while it was being read")
    return b"".join(chunks)


def read_stable_json(path, label):
    raw = read_stable_bytes(path, label)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} is not valid UTF-8 JSON: {error}")


def read_stable_text_or_json(path, label):
    raw = read_stable_bytes(path, label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"{label} is not UTF-8: {error}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = text
    return parsed


class Pseudonymizer:
    def __init__(self, key, stable):
        self.key = key
        self.stable = stable

    def digest(self, namespace, value):
        material = f"{namespace}\0{value}".encode("utf-8", errors="replace")
        return hmac.new(self.key, material, hashlib.sha256).hexdigest()[:24]


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RestClient:
    def __init__(self, base_url, timeout_seconds, authorization, ca_file):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            fail("--rest-url must be an absolute http:// or https:// URL")
        if parsed.username is not None or parsed.password is not None:
            fail("--rest-url must not contain credentials; use --authorization-env")
        if parsed.query or parsed.fragment:
            fail("--rest-url must not contain a query or fragment")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.authorization = authorization
        handlers = [RejectRedirects()]
        if parsed.scheme == "https":
            context = ssl.create_default_context(cafile=ca_file)
            handlers.append(urllib.request.HTTPSHandler(context=context))
        self.opener = urllib.request.build_opener(*handlers)
        self.warnings = []

    def get(self, path, label, required=False, query=None):
        url = self.base_url + "/" + path.lstrip("/")
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {
            "Accept": "application/json",
            "User-Agent": "fep-flink-workload-collector",
        }
        if self.authorization is not None:
            headers["Authorization"] = self.authorization
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        if int(content_length) > MAX_INPUT_BYTES:
                            fail(f"{label} response exceeds the safety limit")
                    except ValueError:
                        pass
                raw = response.read(MAX_INPUT_BYTES + 1)
        except urllib.error.HTTPError as error:
            message = f"{label} unavailable (HTTP {error.code})"
            if required:
                fail(message)
            self.warnings.append(message)
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            message = f"{label} unavailable (network or TLS error)"
            if required:
                fail(message)
            self.warnings.append(message)
            return None
        if len(raw) > MAX_INPUT_BYTES:
            fail(f"{label} response exceeds the safety limit")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if required:
                fail(f"{label} did not return valid UTF-8 JSON: {error}")
            self.warnings.append(f"{label} returned invalid JSON")
            return None


def load_capabilities(path):
    document = read_stable_json(path, "capability inventory")
    if not isinstance(document, dict) or document.get("schema") != CAPABILITY_SCHEMA:
        fail(f"capability inventory must use schema {CAPABILITY_SCHEMA}")
    families = document.get("families")
    if not isinstance(families, list) or not families:
        fail("capability inventory has no families")
    version = document.get("flink_version")
    if not isinstance(version, str) or SAFE_VERSION_RE.fullmatch(version) is None:
        fail("capability inventory has an invalid Flink version")
    names = set()
    validated = []
    for entry in families:
        if not isinstance(entry, dict):
            fail("capability inventory contains a non-object family")
        name = entry.get("name")
        tokens = entry.get("tokens")
        coverage = entry.get("coverage")
        priority = entry.get("review_priority")
        review_checks = entry.get("review_checks")
        sql_nodes = entry.get("sql_physical_nodes")
        library_api = entry.get("library_api")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name):
            fail("capability inventory contains an invalid family name")
        if name in names:
            fail(f"capability inventory repeats family {name}")
        if (
            not isinstance(tokens, list)
            or not tokens
            or not all(
                isinstance(token, str) and 2 <= len(token) <= 80 for token in tokens
            )
        ):
            fail(f"capability family {name} has invalid tokens")
        if coverage not in COVERAGE_VALUES:
            fail(f"capability family {name} has invalid coverage")
        if not isinstance(sql_nodes, list) or not all(
            isinstance(node, str) and node for node in sql_nodes
        ):
            fail(f"capability family {name} has invalid sql_physical_nodes")
        if (coverage == "library-provided") != isinstance(library_api, str):
            fail(
                f"capability family {name}: library_api is required exactly for library-provided"
            )
        if priority not in {"high", "medium", "low"}:
            fail(f"capability family {name} has invalid review priority")
        if (
            not isinstance(review_checks, list)
            or not review_checks
            or not all(
                isinstance(check, str) and re.fullmatch(r"[a-z][a-z0-9_]{2,63}", check)
                for check in review_checks
            )
            or len(set(review_checks)) != len(review_checks)
        ):
            fail(f"capability family {name} has invalid review checks")
        names.add(name)
        validated.append(
            {
                "name": name,
                "tokens": tuple(tokens),
                "coverage": coverage,
                "review_priority": priority,
                "review_checks": tuple(review_checks),
                "sql_physical_nodes": tuple(sql_nodes),
                "library_api": library_api,
            }
        )
    return document, validated


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings_in(child)


def contains_operator_token(text, token):
    variants = (token,)
    return any(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(variant)}(?![A-Za-z0-9_])",
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for variant in variants
    )


def classify_families(parts, pact, family_specs):
    pact_lower = pact.lower() if isinstance(pact, str) else ""
    text = " ".join(part for part in parts if isinstance(part, str))
    lowered = text.lower()
    if "data source" in pact_lower:
        return ["Source"]
    if "data sink" in pact_lower:
        return ["Sink"]
    matched = [
        spec["name"]
        for spec in family_specs
        if any(contains_operator_token(lowered, token) for token in spec["tokens"])
    ]

    specific_joins = {"IntervalJoin", "WindowJoin", "TemporalJoin", "LookupJoin"}
    if specific_joins.intersection(matched) and "StreamExecJoin" not in text:
        matched = [name for name in matched if name != "Join"]
    if {"LocalWindowAggregate", "GlobalWindowAggregate"}.intersection(matched):
        matched = [name for name in matched if name != "WindowAggregate"]
    if {"LocalGroupAggregate", "GlobalGroupAggregate"}.intersection(matched):
        matched = [name for name in matched if name != "GroupAggregate"]
    if "WindowRank" in matched:
        matched = [name for name in matched if name != "Rank"]
    if "TemporalSort" in matched:
        matched = [name for name in matched if name != "SortLimit"]
    return list(dict.fromkeys(matched)) or ["Unknown"]


PLAN_RISK_PATTERNS = {
    "python_or_udf": re.compile(
        r"(?<![A-Za-z0-9_])(?:Python|PythonFunction|ScalarFunction|TableFunction|UDF)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "decimal_type": re.compile(r"(?<![A-Za-z0-9_])DECIMAL\s*\(", re.IGNORECASE),
    "timestamp_ltz": re.compile(
        r"(?<![A-Za-z0-9_])TIMESTAMP_LTZ(?![A-Za-z0-9_])", re.IGNORECASE
    ),
    "nested_or_binary_type": re.compile(
        r"(?<![A-Za-z0-9_])(?:ARRAY|MAP|MULTISET|VARBINARY|BINARY)\s*[<(]",
        re.IGNORECASE,
    ),
    "unsupported_string_expression": re.compile(
        r"(?<![A-Za-z0-9_])(?:REGEXP|REGEX|UPPER|LOWER|CONCAT)\s*\(",
        re.IGNORECASE,
    ),
    "distinct_aggregate": re.compile(
        r"(?<![A-Za-z0-9_])DISTINCT(?![A-Za-z0-9_])", re.IGNORECASE
    ),
    "processing_time": re.compile(
        r"(?<![A-Za-z0-9_])(?:PROCTIME|PROCESSING[-_ ]?TIME)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "session_window": re.compile(
        r"(?<![A-Za-z0-9_])SESSION(?![A-Za-z0-9_])", re.IGNORECASE
    ),
    "non_inner_join": re.compile(
        r"(?:joinType\s*=\s*\[?\s*(?:LEFT|RIGHT|FULL|SEMI|ANTI)|"
        r"(?<![A-Za-z0-9_])(?:LEFT|RIGHT|FULL|SEMI|ANTI)(?:\s+OUTER)?\s+JOIN)",
        re.IGNORECASE,
    ),
    "async_lookup": re.compile(
        r"(?<![A-Za-z0-9_])ASYNC(?:HRONOUS)?(?![A-Za-z0-9_])", re.IGNORECASE
    ),
    "update_or_retract_changelog": re.compile(
        r"(?<![A-Za-z0-9_])(?:UPDATE_BEFORE|UPDATE_AFTER|RETRACT)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
}


def plan_risk_summary(value):
    text = "\n".join(strings_in(value))
    return [
        {"code": code, "mentions": len(pattern.findall(text))}
        for code, pattern in sorted(PLAN_RISK_PATTERNS.items())
        if pattern.search(text) is not None
    ]


def plan_object(raw_plan):
    if isinstance(raw_plan, dict) and "plan" in raw_plan:
        raw_plan = raw_plan["plan"]
    if isinstance(raw_plan, str):
        try:
            raw_plan = json.loads(raw_plan)
        except json.JSONDecodeError:
            return None
    return raw_plan if isinstance(raw_plan, dict) else None


def sanitize_plan(raw_plan, pseudonymizer, family_specs):
    parsed = plan_object(raw_plan)
    if parsed is None:
        families = Counter(
            classify_families(list(strings_in(raw_plan)), "", family_specs)
        )
        return {
            "plan_fingerprint_hmac": pseudonymizer.digest(
                "plan", canonical_json(raw_plan)
            ),
            "nodes": [],
            "family_counts": dict(sorted(families.items())),
            "shape_risks": plan_risk_summary(raw_plan),
            "graph_available": False,
        }
    nodes = parsed.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
    if len(nodes) > MAX_PLAN_NODES:
        fail(f"job plan exceeds the {MAX_PLAN_NODES}-node safety limit")
    id_map = {}
    for index, node in enumerate(nodes, 1):
        if isinstance(node, dict) and node.get("id") is not None:
            id_map[str(node["id"])] = f"plan_node_{index:03d}"
    sanitized_nodes = []
    counts = Counter()
    for index, node in enumerate(nodes, 1):
        if not isinstance(node, dict):
            continue
        pact = node.get("pact") if isinstance(node.get("pact"), str) else ""
        parts = [
            node.get("type"),
            node.get("contents"),
            node.get("description"),
            node.get("operator"),
        ]
        families = classify_families(parts, pact, family_specs)
        counts.update(families)
        predecessors = node.get("predecessors")
        if isinstance(predecessors, dict):
            predecessors = [predecessors]
        edges = []
        if isinstance(predecessors, list):
            for predecessor in predecessors:
                if not isinstance(predecessor, dict) or predecessor.get("id") is None:
                    continue
                raw_id = str(predecessor["id"])
                source = id_map.get(raw_id)
                if source is None:
                    source = (
                        "external_" + pseudonymizer.digest("plan-node", raw_id)[:12]
                    )
                strategy = predecessor.get("ship_strategy")
                safe_edge = {"source": source}
                if isinstance(strategy, str):
                    normalized_strategy = strategy.upper().replace(" ", "_")
                    if normalized_strategy in SAFE_SHIP_STRATEGIES:
                        safe_edge["ship_strategy"] = normalized_strategy
                edges.append(safe_edge)
        raw_id = node.get("id")
        safe_id = id_map.get(str(raw_id), f"plan_node_{index:03d}")
        sanitized = {
            "id": safe_id,
            "families": families,
            "parallelism": non_negative_int(node.get("parallelism")),
            "predecessors": edges,
        }
        sanitized_nodes.append(sanitized)
    return {
        "plan_fingerprint_hmac": pseudonymizer.digest("plan", canonical_json(parsed)),
        "nodes": sanitized_nodes,
        "family_counts": dict(sorted(counts.items())),
        "shape_risks": plan_risk_summary(parsed),
        "graph_available": bool(sanitized_nodes),
    }


def metric_name(raw_name):
    if not isinstance(raw_name, str):
        return None
    for candidate in RUNTIME_METRICS:
        if raw_name == candidate or raw_name.endswith("." + candidate):
            return candidate
    return None


def runtime_metric_ids(raw_inventory):
    if not isinstance(raw_inventory, list):
        return []
    selected = set()
    for entry in raw_inventory:
        raw_id = entry.get("id") if isinstance(entry, dict) else None
        if (
            not isinstance(raw_id, str)
            or not raw_id
            or len(raw_id) > MAX_METRIC_ID_CHARS
            or "," in raw_id
            or any(ord(character) < 32 for character in raw_id)
        ):
            continue
        canonical = metric_name(raw_id)
        if canonical is None:
            continue
        if raw_id == canonical:
            selected.add(raw_id)
            continue
        prefix = raw_id[: -(len(canonical) + 1)]
        if re.fullmatch(r"[0-9]+", prefix) is not None:
            selected.add(raw_id)
    return sorted(selected)


def metric_id_chunks(metric_ids):
    chunks = []
    current = []
    for metric_id in metric_ids:
        candidate = [*current, metric_id]
        if current and len(",".join(candidate)) > MAX_METRIC_QUERY_CHARS:
            chunks.append(current)
            current = [metric_id]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def collect_vertex_metric_ids(client, prefix):
    inventory = client.get(prefix + "/metrics", "vertex metric inventory")
    return runtime_metric_ids(inventory)


def collect_vertex_metrics(client, prefix, metric_ids):
    if not metric_ids:
        return []

    collected = []
    for chunk in metric_id_chunks(metric_ids):
        requested = set(chunk)
        response = client.get(
            prefix + "/metrics",
            "vertex metrics",
            query={"get": ",".join(chunk)},
        )
        if not isinstance(response, list):
            return []
        returned = {
            entry.get("id")
            for entry in response
            if isinstance(entry, dict) and entry.get("id") in requested
        }
        if returned != requested:
            return []
        collected.extend(
            entry
            for entry in response
            if isinstance(entry, dict) and entry.get("id") in requested
        )
    return collected


def collect_subtask_metrics(client, prefix, parallelism, metric_ids_by_subtask):
    collected = []
    for subtask_index in range(parallelism):
        subtask_prefix = f"{prefix}/subtasks/{subtask_index}"
        metric_ids = metric_ids_by_subtask.get(subtask_index, [])
        if not metric_ids:
            metric_ids = collect_vertex_metric_ids(client, subtask_prefix)
            metric_ids_by_subtask[subtask_index] = metric_ids
        for entry in collect_vertex_metrics(client, subtask_prefix, metric_ids):
            canonical = metric_name(entry.get("id"))
            if canonical is None:
                continue
            safe_entry = {
                key: value
                for key, value in entry.items()
                if key in {"value", "min", "max", "avg", "sum"}
            }
            safe_entry["id"] = f"{subtask_index}.{canonical}"
            collected.append(safe_entry)
    return collected


def sanitize_metrics(raw_metrics):
    values = defaultdict(lambda: defaultdict(list))
    if isinstance(raw_metrics, dict):
        entries = []
        for raw_name, raw_value in raw_metrics.items():
            canonical = DETAIL_METRIC_NAMES.get(raw_name, metric_name(raw_name))
            if canonical is not None:
                entries.append({"id": canonical, "value": raw_value})
    elif isinstance(raw_metrics, list):
        entries = raw_metrics
    else:
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        canonical = metric_name(entry.get("id"))
        if canonical is None and entry.get("id") in DETAIL_METRIC_NAMES.values():
            canonical = entry.get("id")
        if canonical is None:
            continue
        for statistic in ("value", "min", "max", "avg", "sum"):
            parsed = finite_number(entry.get(statistic))
            if parsed is not None:
                values[canonical][statistic].append(parsed)
    result = {}
    for name, statistics in sorted(values.items()):
        aggregate = {}
        for statistic, samples in statistics.items():
            if statistic == "min":
                aggregate[statistic] = min(samples)
            elif statistic == "max":
                aggregate[statistic] = max(samples)
            elif statistic == "sum":
                aggregate[statistic] = sum(samples)
            else:
                aggregate[statistic] = sum(samples) / len(samples)
        result[name] = aggregate
    return result


def metric_subtask_index(raw_name, canonical):
    if not isinstance(raw_name, str) or raw_name == canonical:
        return None
    suffix = "." + canonical
    if not raw_name.endswith(suffix):
        return None
    prefix = raw_name[: -len(suffix)]
    if re.fullmatch(r"[0-9]+", prefix) is None:
        return None
    index = int(prefix)
    return index if index <= MAX_SUBTASK_INDEX else None


def sanitize_subtask_metrics(raw_metrics):
    """Keep numeric subtask splits without retaining task or operator names."""
    if not isinstance(raw_metrics, list):
        return []
    grouped = defaultdict(list)
    for entry in raw_metrics:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id")
        canonical = metric_name(raw_id)
        if canonical is None:
            continue
        index = metric_subtask_index(raw_id, canonical)
        if index is None:
            continue
        safe_entry = {"id": canonical}
        for statistic in ("value", "min", "max", "avg", "sum"):
            if statistic in entry:
                safe_entry[statistic] = entry[statistic]
        grouped[index].append(safe_entry)
    return [
        {"index": index, "metrics": sanitize_metrics(entries)}
        for index, entries in sorted(grouped.items())
    ]


def collect_ratio_values(value, output):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.replace("-", "").replace("_", "").lower()
            if normalized in {
                "backpressureratio",
                "busyratio",
                "idleratio",
                "softbackpressureratio",
                "hardbackpressureratio",
            }:
                parsed = finite_number(child)
                if parsed is not None and 0 <= parsed <= 1:
                    output[normalized].append(parsed)
            else:
                collect_ratio_values(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_ratio_values(child, output)


def sanitize_backpressure(raw):
    if not isinstance(raw, dict):
        return {}
    level = raw.get("backpressure-level", raw.get("backpressureLevel"))
    status = raw.get("status")
    result = {}
    if isinstance(level, str) and level.lower() in {"ok", "low", "high"}:
        result["level"] = level.lower()
    if isinstance(status, str) and status.lower() in {"ok", "deprecated"}:
        result["status"] = status.lower()
    ratios = defaultdict(list)
    collect_ratio_values(raw, ratios)
    if ratios:
        result["ratios"] = {
            name: {
                "min": min(samples),
                "max": max(samples),
                "avg": sum(samples) / len(samples),
            }
            for name, samples in sorted(ratios.items())
        }
    return result


def safe_state(value):
    if isinstance(value, str) and value.upper() in SAFE_JOB_STATES:
        return value.upper()
    return "UNKNOWN"


def safe_task_counts(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, raw in value.items():
        normalized = key.upper().replace("-", "_")
        parsed = non_negative_int(raw)
        if normalized in SAFE_JOB_STATES and parsed is not None:
            result[normalized] = parsed
    return dict(sorted(result.items()))


def sanitize_vertices(raw_job, vertex_details, family_specs):
    raw_vertices = raw_job.get("vertices") if isinstance(raw_job, dict) else None
    if not isinstance(raw_vertices, list):
        raw_vertices = []
    if len(raw_vertices) > MAX_VERTICES:
        fail(f"job exceeds the {MAX_VERTICES}-vertex safety limit")
    id_map = {}
    for index, vertex in enumerate(raw_vertices, 1):
        if isinstance(vertex, dict) and isinstance(vertex.get("id"), str):
            id_map[vertex["id"]] = f"vertex_{index:03d}"
    sanitized = []
    for index, vertex in enumerate(raw_vertices, 1):
        if not isinstance(vertex, dict):
            continue
        raw_id = (
            vertex.get("id")
            if isinstance(vertex.get("id"), str)
            else f"missing-{index}"
        )
        details = vertex_details.get(raw_id)
        if not isinstance(details, dict):
            details = {}
        parts = [vertex.get("name"), details.get("name")]
        families = classify_families(parts, "", family_specs)
        parallelism = non_negative_int(vertex.get("parallelism"))
        if parallelism is None:
            parallelism = non_negative_int(details.get("parallelism"))
        duration = non_negative_int(vertex.get("duration"))
        if duration is None:
            duration = non_negative_int(details.get("duration"))
        raw_metrics = vertex.get("metrics")
        if not isinstance(raw_metrics, dict):
            raw_metrics = details.get("metrics")
        sanitized.append(
            {
                "id": id_map.get(raw_id, f"vertex_{index:03d}"),
                "families": families,
                "state": safe_state(vertex.get("status", details.get("status"))),
                "parallelism": parallelism,
                "duration_ms": duration,
                "task_counts": safe_task_counts(details.get("tasks")),
                "lifetime_metrics": sanitize_metrics(raw_metrics),
            }
        )
    return sanitized, id_map


def safe_number_or_bool(value):
    if isinstance(value, bool):
        return value
    return finite_number(value)


def safe_restart_strategy(value):
    if not isinstance(value, str):
        return None
    normalized = value.lower().replace("_", "-").replace(" ", "-")
    if "exponential" in normalized:
        return "exponential-delay"
    if "failure-rate" in normalized or "failurerate" in normalized:
        return "failure-rate"
    if "fixed-delay" in normalized or "fixeddelay" in normalized:
        return "fixed-delay"
    if normalized in {"none", "no-restart", "off"}:
        return "none"
    if normalized in {"cluster", "cluster-default", "default"}:
        return "cluster-default"
    return "custom-or-unknown"


def sanitize_job_config(raw):
    if not isinstance(raw, dict):
        return {}
    execution = raw.get("execution-config")
    if not isinstance(execution, dict):
        execution = {}
    result = {}
    for key in ("job-parallelism", "object-reuse-mode"):
        value = safe_number_or_bool(execution.get(key))
        if value is not None:
            result[key] = value
    execution_mode = execution.get("execution-mode")
    if (
        isinstance(execution_mode, str)
        and execution_mode.upper() in SAFE_EXECUTION_MODES
    ):
        result["execution-mode"] = execution_mode.upper()
    restart_strategy = safe_restart_strategy(execution.get("restart-strategy"))
    if restart_strategy is not None:
        result["restart-strategy"] = restart_strategy
    return result


def sanitize_checkpoint_config(raw):
    if not isinstance(raw, dict):
        return {}
    result = {}
    for key in sorted(SAFE_CHECKPOINT_CONFIG - {"interval", "mode"}):
        value = safe_number_or_bool(raw.get(key))
        if value is not None:
            result[key] = value
    interval = finite_number(raw.get("interval"))
    if interval is not None and 0 < interval <= MAX_CHECKPOINT_INTERVAL_MS:
        result["enabled"] = True
        result["interval"] = interval
    elif interval is not None:
        # Flink uses Long.MAX_VALUE when periodic checkpointing is disabled.
        result["enabled"] = False
    mode = raw.get("mode")
    if isinstance(mode, str) and mode.upper() in SAFE_CHECKPOINT_MODES:
        result["mode"] = mode.upper()
    externalization = raw.get("externalization")
    if isinstance(externalization, dict):
        result["externalization"] = {
            key: value
            for key in ("enabled", "delete_on_cancellation")
            if isinstance((value := externalization.get(key)), bool)
        }
    return result


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def checkpoint_entry_numbers(entry):
    if not isinstance(entry, dict):
        return None
    duration = finite_number(entry.get("end_to_end_duration", entry.get("duration")))
    size = finite_number(entry.get("checkpointed_size", entry.get("state_size")))
    return duration, size


def sanitize_checkpoints(raw):
    if not isinstance(raw, dict):
        return {}
    result = {}
    counts = raw.get("counts")
    if isinstance(counts, dict):
        safe_counts = {}
        for key in ("total", "in_progress", "completed", "failed", "restored"):
            value = non_negative_int(counts.get(key))
            if value is not None:
                safe_counts[key] = value
        result["counts"] = safe_counts
    history = raw.get("history")
    if not isinstance(history, list):
        history = []
    durations = []
    sizes = []
    for entry in history:
        numbers = checkpoint_entry_numbers(entry)
        if numbers is None:
            continue
        duration, size = numbers
        if duration is not None and duration >= 0:
            durations.append(duration)
        if size is not None and size >= 0:
            sizes.append(size)
    if history:
        result["history_entries"] = len(history)
    if durations:
        result["duration_ms"] = {
            "min": min(durations),
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "max": max(durations),
        }
    if sizes:
        result["checkpointed_size_bytes"] = {
            "min": min(sizes),
            "p50": percentile(sizes, 0.50),
            "p95": percentile(sizes, 0.95),
            "max": max(sizes),
        }
    return result


def sanitize_exceptions(raw):
    if not isinstance(raw, dict):
        return {}
    exceptions = raw.get("all-exceptions", raw.get("allExceptions"))
    count = len(exceptions) if isinstance(exceptions, list) else 0
    root = raw.get("root-exception", raw.get("rootException"))
    if count == 0 and isinstance(root, str) and root:
        count = 1
    result = {"exception_count": count}
    truncated = raw.get("truncated")
    if isinstance(truncated, bool):
        result["truncated"] = truncated
    return result


def sanitize_overview(raw):
    if not isinstance(raw, dict):
        return {}
    result = {}
    version = raw.get("flink-version", raw.get("flinkVersion"))
    version_match = (
        SAFE_VERSION_PREFIX_RE.match(version) if isinstance(version, str) else None
    )
    if version_match is not None:
        result["flink_version"] = version_match.group(1)
    for raw_key, output_key in (
        ("taskmanagers", "taskmanagers"),
        ("slots-total", "slots_total"),
        ("slots-available", "slots_available"),
        ("jobs-running", "jobs_running"),
        ("jobs-finished", "jobs_finished"),
        ("jobs-cancelled", "jobs_cancelled"),
        ("jobs-failed", "jobs_failed"),
    ):
        value = non_negative_int(raw.get(raw_key))
        if value is not None:
            result[output_key] = value
    return result


def valid_vertex_id(value):
    return isinstance(value, str) and JOB_ID_RE.fullmatch(value) is not None


def raw_vertex_ids(raw_job):
    vertices = raw_job.get("vertices") if isinstance(raw_job, dict) else None
    if not isinstance(vertices, list):
        return []
    result = []
    seen = set()
    for vertex in vertices:
        vertex_id = vertex.get("id") if isinstance(vertex, dict) else None
        if valid_vertex_id(vertex_id) and vertex_id not in seen:
            seen.add(vertex_id)
            result.append(vertex_id)
    if len(result) > MAX_VERTICES:
        fail(f"job exceeds the {MAX_VERTICES}-vertex safety limit")
    return result


def job_vertices_ready(raw_job):
    if not isinstance(raw_job, dict):
        return False
    job_state = safe_state(raw_job.get("state", raw_job.get("status")))
    if job_state in {"FAILED", "CANCELED", "FINISHED", "SUSPENDED"}:
        return True
    vertices = raw_job.get("vertices")
    if not isinstance(vertices, list) or not vertices:
        return False
    ready_states = {
        "RUNNING",
        "FAILING",
        "CANCELLING",
        "FAILED",
        "CANCELED",
        "FINISHED",
    }
    return all(
        isinstance(vertex, dict)
        and isinstance(vertex.get("status"), str)
        and vertex["status"].upper() in ready_states
        for vertex in vertices
    )


def collect_rest(args, authorization):
    client = RestClient(
        args.rest_url, args.timeout_seconds, authorization, args.ca_file
    )
    job_id = args.job_id.lower()
    encoded_job = urllib.parse.quote(job_id, safe="")
    raw = {
        "overview": client.get("overview", "cluster overview"),
        "job": client.get(f"jobs/{encoded_job}", "job details", required=True),
        "plan": client.get(f"jobs/{encoded_job}/plan", "job plan", required=True),
        "config": client.get(f"jobs/{encoded_job}/config", "job configuration"),
        "checkpoints": client.get(
            f"jobs/{encoded_job}/checkpoints", "checkpoint statistics"
        ),
        "checkpoint_config": client.get(
            f"jobs/{encoded_job}/checkpoints/config", "checkpoint configuration"
        ),
        "exceptions": client.get(f"jobs/{encoded_job}/exceptions", "job exceptions"),
    }
    deployment_deadline = time.monotonic() + args.startup_wait_seconds
    while not job_vertices_ready(raw["job"]) and time.monotonic() < deployment_deadline:
        remaining = deployment_deadline - time.monotonic()
        time.sleep(min(0.5, max(0.0, remaining)))
        refreshed_job = client.get(f"jobs/{encoded_job}", "job deployment status")
        if isinstance(refreshed_job, dict):
            raw["job"] = refreshed_job
    if not job_vertices_ready(raw["job"]):
        client.warnings.append(
            "job vertices did not finish deployment before runtime sampling began"
        )

    vertex_ids = raw_vertex_ids(raw["job"])
    vertex_details = {}
    for vertex_id in vertex_ids:
        encoded_vertex = urllib.parse.quote(vertex_id, safe="")
        value = client.get(
            f"jobs/{encoded_job}/vertices/{encoded_vertex}", "vertex details"
        )
        if value is not None:
            vertex_details[vertex_id] = value
    raw["vertex_details"] = vertex_details

    metric_ids_by_vertex = {}
    subtask_metric_ids_by_vertex = {}
    parallelism_by_vertex = {
        vertex.get("id"): non_negative_int(vertex.get("parallelism"))
        for vertex in raw["job"].get("vertices", [])
        if isinstance(vertex, dict) and valid_vertex_id(vertex.get("id"))
    }
    fallback_subtask_budget = 0
    fallback_eligible = set()
    for vertex_id in vertex_ids:
        encoded_vertex = urllib.parse.quote(vertex_id, safe="")
        prefix = f"jobs/{encoded_job}/vertices/{encoded_vertex}"
        metric_ids_by_vertex[vertex_id] = collect_vertex_metric_ids(client, prefix)
        parallelism = parallelism_by_vertex.get(vertex_id)
        if (
            not metric_ids_by_vertex[vertex_id]
            and parallelism is not None
            and 0 < parallelism <= MAX_FALLBACK_SUBTASKS_PER_VERTEX
            and fallback_subtask_budget + parallelism <= MAX_FALLBACK_SUBTASKS_TOTAL
        ):
            fallback_eligible.add(vertex_id)
            fallback_subtask_budget += parallelism
            subtask_metric_ids_by_vertex[vertex_id] = {}

    samples = []
    collection_start = time.monotonic()
    vertices_with_metrics = set()
    for sample_index in range(args.sample_count):
        sample_start = time.monotonic()
        vertices = {}
        for vertex_id in vertex_ids:
            encoded_vertex = urllib.parse.quote(vertex_id, safe="")
            prefix = f"jobs/{encoded_job}/vertices/{encoded_vertex}"
            if not metric_ids_by_vertex.get(vertex_id):
                # A just-submitted job can be RUNNING before its subtasks expose
                # metrics. Retry empty inventories instead of caching startup
                # emptiness for the whole collection window.
                metric_ids_by_vertex[vertex_id] = collect_vertex_metric_ids(
                    client, prefix
                )
            if metric_ids_by_vertex.get(vertex_id):
                metrics = collect_vertex_metrics(
                    client, prefix, metric_ids_by_vertex[vertex_id]
                )
            elif vertex_id in fallback_eligible:
                metrics = collect_subtask_metrics(
                    client,
                    prefix,
                    parallelism_by_vertex[vertex_id],
                    subtask_metric_ids_by_vertex[vertex_id],
                )
            else:
                metrics = []
            if metrics:
                vertices_with_metrics.add(vertex_id)
            backpressure = client.get(prefix + "/backpressure", "vertex backpressure")
            vertices[vertex_id] = {
                "metrics": metrics,
                "backpressure": backpressure,
            }
        samples.append(
            {
                "offset_ms": round((time.monotonic() - collection_start) * 1000),
                "vertices": vertices,
            }
        )
        if sample_index + 1 < args.sample_count:
            elapsed = time.monotonic() - sample_start
            time.sleep(max(0, args.sample_interval_seconds - elapsed))

    # Refresh mutable summaries after sampling. Without this, a collection that
    # starts during deployment can permanently report CREATED vertices and zero
    # lifetime counters even though all runtime samples came from a running job.
    latest_overview = client.get("overview", "cluster overview refresh")
    if isinstance(latest_overview, dict):
        raw["overview"] = latest_overview
    latest_job = client.get(f"jobs/{encoded_job}", "job details refresh")
    if isinstance(latest_job, dict):
        raw["job"] = latest_job
    refreshed_vertex_details = {}
    for vertex_id in vertex_ids:
        encoded_vertex = urllib.parse.quote(vertex_id, safe="")
        value = client.get(
            f"jobs/{encoded_job}/vertices/{encoded_vertex}",
            "vertex details refresh",
        )
        if isinstance(value, dict):
            refreshed_vertex_details[vertex_id] = value
    if refreshed_vertex_details:
        raw["vertex_details"] = refreshed_vertex_details
    for key, endpoint, label in (
        (
            "checkpoints",
            f"jobs/{encoded_job}/checkpoints",
            "checkpoint statistics refresh",
        ),
        ("exceptions", f"jobs/{encoded_job}/exceptions", "job exceptions refresh"),
    ):
        value = client.get(endpoint, label)
        if isinstance(value, dict):
            raw[key] = value
    if vertex_ids and len(vertices_with_metrics) < len(vertex_ids):
        client.warnings.append(
            "runtime metrics unavailable for "
            f"{len(vertex_ids) - len(vertices_with_metrics)} of {len(vertex_ids)} vertices"
        )
    skipped_fallback_vertices = [
        vertex_id
        for vertex_id in vertex_ids
        if not metric_ids_by_vertex.get(vertex_id)
        and vertex_id not in fallback_eligible
        and (parallelism_by_vertex.get(vertex_id) or 0) > 0
    ]
    if skipped_fallback_vertices:
        client.warnings.append(
            "subtask metric fallback skipped for one or more vertices because "
            "parallelism exceeds the request safety limit"
        )
    raw["runtime_samples"] = samples
    return raw, client.warnings, job_id


def collect_snapshot(args):
    raw = read_stable_json(args.snapshot_file, "REST snapshot")
    if not isinstance(raw, dict) or raw.get("schema") != SNAPSHOT_SCHEMA:
        fail(f"--snapshot-file must use schema {SNAPSHOT_SCHEMA}")
    if not isinstance(raw.get("job"), dict):
        fail("REST snapshot is missing job details")
    if raw.get("plan") is None:
        fail("REST snapshot is missing the job plan")
    job_id = args.job_id or raw["job"].get("jid")
    if not valid_vertex_id(job_id):
        fail("snapshot must contain a 32-character hexadecimal job id")
    samples = raw.get("runtime_samples")
    if samples is not None and not isinstance(samples, list):
        fail("snapshot runtime_samples must be an array")
    if isinstance(samples, list) and len(samples) > MAX_SAMPLES:
        fail(f"snapshot exceeds the {MAX_SAMPLES}-sample safety limit")
    return raw, [], job_id.lower()


def collect_history_archive(args, capability_doc):
    document = read_stable_json(args.archive_file, "Flink history archive")
    archive = document.get("archive") if isinstance(document, dict) else None
    if not isinstance(archive, list) or not archive:
        fail("Flink history archive must contain a non-empty archive array")
    if len(archive) > MAX_ARCHIVE_ENTRIES:
        fail(
            f"Flink history archive exceeds the {MAX_ARCHIVE_ENTRIES}-entry safety limit"
        )

    endpoints = {}
    job_ids = set()
    for index, entry in enumerate(archive):
        if not isinstance(entry, dict):
            fail(f"Flink history archive entry {index} is not an object")
        path = entry.get("path")
        embedded = entry.get("json")
        if not isinstance(path, str) or not path or len(path) > 2048:
            fail(f"Flink history archive entry {index} has an invalid REST path")
        if not isinstance(embedded, str):
            fail(f"Flink history archive entry {index} has no embedded JSON string")
        normalized_path = "/" + path.lstrip("/")
        if (
            "?" in normalized_path
            or "#" in normalized_path
            or ".." in normalized_path.split("/")
            or any(ord(character) < 32 for character in normalized_path)
        ):
            fail(f"Flink history archive entry {index} has an unsafe REST path")
        if normalized_path in endpoints:
            fail("Flink history archive repeats a REST path")
        try:
            endpoints[normalized_path] = json.loads(embedded)
        except json.JSONDecodeError as error:
            fail(
                f"Flink history archive entry {index} has invalid embedded JSON: {error}"
            )
        match = ARCHIVE_JOB_PATH_RE.match(normalized_path)
        if match is not None:
            job_ids.add(match.group(1).lower())

    if len(job_ids) != 1:
        fail("Flink history archive must contain REST entries for exactly one job")
    job_id = next(iter(job_ids))
    if args.job_id is not None:
        if JOB_ID_RE.fullmatch(args.job_id) is None:
            fail("--job-id must be 32 hexadecimal characters")
        if args.job_id.lower() != job_id:
            fail("--job-id does not match the Flink history archive")
    if args.archive_flink_version != capability_doc["flink_version"]:
        fail("Flink history archive version does not match the capability inventory")

    prefix = f"/jobs/{job_id}"
    job = endpoints.get(prefix)
    plan = endpoints.get(prefix + "/plan")
    if not isinstance(job, dict):
        fail("Flink history archive is missing the job details REST entry")
    if plan is None:
        fail("Flink history archive is missing the job plan REST entry")
    raw = {
        "schema": SNAPSHOT_SCHEMA,
        "overview": {"flink-version": args.archive_flink_version},
        "job": job,
        "plan": plan,
        "config": endpoints.get(prefix + "/config"),
        "checkpoint_config": endpoints.get(prefix + "/checkpoints/config"),
        "checkpoints": endpoints.get(prefix + "/checkpoints"),
        "exceptions": endpoints.get(prefix + "/exceptions"),
        "runtime_samples": [],
        "vertex_details": {},
    }
    vertex_prefix = re.escape(prefix) + r"/vertices/([0-9a-fA-F]{32})\Z"
    for path, value in endpoints.items():
        match = re.fullmatch(vertex_prefix, path)
        if match is not None and isinstance(value, dict):
            raw["vertex_details"][match.group(1).lower()] = value
    warnings = ["Flink history archive has no live short-period runtime metric samples"]
    for key, endpoint in (
        ("config", "job configuration"),
        ("checkpoint_config", "checkpoint configuration"),
        ("checkpoints", "checkpoint statistics"),
        ("exceptions", "job exceptions"),
    ):
        if raw[key] is None:
            warnings.append(f"Flink history archive has no {endpoint} REST entry")
    return raw, warnings, job_id


def sanitize_runtime_samples(raw_samples, vertex_id_map):
    if not isinstance(raw_samples, list):
        return []
    result = []
    previous_offset = -1
    for index, sample in enumerate(raw_samples):
        if not isinstance(sample, dict):
            continue
        offset = non_negative_int(sample.get("offset_ms"))
        if offset is None:
            offset = index
        offset = max(previous_offset, offset)
        previous_offset = offset
        raw_vertices = sample.get("vertices")
        if not isinstance(raw_vertices, dict):
            raw_vertices = {}
        vertices = []
        for raw_id, safe_id in vertex_id_map.items():
            value = raw_vertices.get(raw_id)
            if not isinstance(value, dict):
                continue
            sanitized_vertex = {
                "id": safe_id,
                "metrics": sanitize_metrics(value.get("metrics")),
                "backpressure": sanitize_backpressure(value.get("backpressure")),
            }
            subtasks = sanitize_subtask_metrics(value.get("metrics"))
            if subtasks:
                sanitized_vertex["subtasks"] = subtasks
            vertices.append(sanitized_vertex)
        result.append({"offset_ms": offset, "vertices": vertices})
    return result


def summarize_explain(raw_explain, pseudonymizer, family_specs):
    if raw_explain is None:
        return None
    families = Counter(
        classify_families(list(strings_in(raw_explain)), "", family_specs)
    )
    return {
        "document_fingerprint_hmac": pseudonymizer.digest(
            "explain", canonical_json(raw_explain)
        ),
        "family_counts": dict(sorted(families.items())),
        "shape_risks": plan_risk_summary(raw_explain),
        "raw_text_retained": False,
    }


def build_bundle(
    raw,
    raw_explain,
    job_id,
    warnings,
    pseudonymizer,
    capability_doc,
    family_specs,
    args,
):
    raw_job = raw.get("job") if isinstance(raw.get("job"), dict) else {}
    raw_plan = raw.get("plan")
    raw_name = raw_job.get("name")
    if not isinstance(raw_name, str):
        parsed_plan = plan_object(raw_plan)
        raw_name = (
            parsed_plan.get("name") if isinstance(parsed_plan, dict) else "unnamed"
        )
    if not isinstance(raw_name, str):
        raw_name = "unnamed"

    vertex_details = raw.get("vertex_details")
    if not isinstance(vertex_details, dict):
        vertex_details = {}
    vertices, vertex_id_map = sanitize_vertices(raw_job, vertex_details, family_specs)
    plan = sanitize_plan(raw_plan, pseudonymizer, family_specs)
    explain = summarize_explain(raw_explain, pseudonymizer, family_specs)

    duration = non_negative_int(raw_job.get("duration"))
    if duration is None:
        start = non_negative_int(raw_job.get("start-time"))
        end = non_negative_int(raw_job.get("end-time"))
        if start is not None and end is not None and end >= start:
            duration = end - start

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "collector_version": COLLECTOR_VERSION,
        "collection": {
            "source": (
                "flink-rest"
                if args.rest_url
                else "flink-history-archive"
                if args.archive_file
                else "offline-rest-snapshot"
            ),
            "sample_count": len(raw.get("runtime_samples") or []),
            "requested_sample_interval_seconds": (
                args.sample_interval_seconds
                if args.rest_url and args.sample_count > 1
                else None
            ),
            "requested_startup_wait_seconds": (
                args.startup_wait_seconds if args.rest_url else None
            ),
            "pseudonyms_stable_across_runs": pseudonymizer.stable,
            "raw_rest_payloads_retained": False,
        },
        "privacy": {
            "raw_sql_retained": False,
            "job_and_operator_names_retained": False,
            "table_and_field_names_retained": False,
            "rest_url_retained": False,
            "exception_messages_retained": False,
            "user_configuration_retained": False,
            "credentials_retained": False,
        },
        "capability_inventory": {
            "schema": capability_doc["schema"],
            "flink_version": capability_doc.get("flink_version"),
            "family_count": len(family_specs),
            "sha256": hashlib.sha256(
                canonical_json(capability_doc).encode("utf-8")
            ).hexdigest(),
        },
        "cluster": sanitize_overview(raw.get("overview")),
        "job": {
            "job_id_hmac": pseudonymizer.digest("job-id", job_id),
            "job_name_hmac": pseudonymizer.digest("job-name", raw_name),
            "state": safe_state(raw_job.get("state", raw_job.get("status"))),
            "duration_ms": duration,
            "vertices": vertices,
        },
        "plan": plan,
        "runtime_samples": sanitize_runtime_samples(
            raw.get("runtime_samples"), vertex_id_map
        ),
        "execution_config": sanitize_job_config(raw.get("config")),
        "checkpoint_config": sanitize_checkpoint_config(raw.get("checkpoint_config")),
        "checkpoints": sanitize_checkpoints(raw.get("checkpoints")),
        "failures": sanitize_exceptions(raw.get("exceptions")),
        "warnings": sorted(set(warnings)),
    }
    cluster_version = bundle["cluster"].get("flink_version")
    capability_version = capability_doc.get("flink_version")
    if cluster_version and capability_version and cluster_version != capability_version:
        bundle["warnings"] = sorted(
            set(
                bundle["warnings"]
                + ["cluster Flink version differs from the capability inventory"]
            )
        )
    if explain is not None:
        bundle["explain"] = explain
    return bundle


def metric_stat(metric, preferred=("avg", "value", "sum", "max")):
    if not isinstance(metric, dict):
        return None
    for name in preferred:
        value = finite_number(metric.get(name))
        if value is not None:
            return float(value)
    return None


def mean(values):
    return sum(values) / len(values) if values else None


def runtime_by_vertex(bundle):
    samples = defaultdict(lambda: defaultdict(list))
    backpressure_levels = defaultdict(list)
    parallelism_by_id = {
        vertex.get("id"): non_negative_int(vertex.get("parallelism"))
        for vertex in bundle.get("job", {}).get("vertices", [])
        if isinstance(vertex, dict)
    }
    for sample in bundle.get("runtime_samples", []):
        for vertex in sample.get("vertices", []):
            vertex_id = vertex.get("id")
            for name in RUNTIME_METRICS:
                value = sample_metric_value(
                    vertex, name, parallelism_by_id.get(vertex_id)
                )
                if value is not None:
                    samples[vertex_id][name].append(value)
            level = vertex.get("backpressure", {}).get("level")
            if level in {"ok", "low", "high"}:
                backpressure_levels[vertex_id].append(level)
    result = {}
    for vertex_id in set(samples).union(backpressure_levels):
        result[vertex_id] = {
            "metrics": {
                name: mean(values)
                for name, values in sorted(samples[vertex_id].items())
            },
            "backpressure_levels": backpressure_levels[vertex_id],
        }
    return result


def is_compute_vertex(vertex):
    return any(
        family not in COMPUTE_BOUNDARY_FAMILIES for family in vertex.get("families", [])
    )


def runtime_evidence_summary(bundle, runtime=None):
    runtime = runtime if runtime is not None else runtime_by_vertex(bundle)
    samples = bundle.get("runtime_samples", [])
    vertices = bundle.get("job", {}).get("vertices", [])
    compute_vertices = [vertex for vertex in vertices if is_compute_vertex(vertex)]
    busy_vertices = [
        vertex
        for vertex in compute_vertices
        if runtime.get(vertex.get("id"), {})
        .get("metrics", {})
        .get("busyTimeMsPerSecond")
        is not None
    ]
    rate_vertices = [
        vertex
        for vertex in compute_vertices
        if any(
            runtime.get(vertex.get("id"), {}).get("metrics", {}).get(name) is not None
            for name in ("numRecordsInPerSecond", "numRecordsOutPerSecond")
        )
    ]
    runtime_vertices = [
        vertex
        for vertex in compute_vertices
        if runtime.get(vertex.get("id"), {}).get("metrics")
    ]
    offsets = [
        offset
        for sample in samples
        if (offset := non_negative_int(sample.get("offset_ms"))) is not None
    ]
    window_ms = max(offsets) - min(offsets) if len(offsets) >= 2 else 0
    requested_interval = finite_number(
        bundle.get("collection", {}).get("requested_sample_interval_seconds")
    )
    requested_window_ms = (
        requested_interval * max(0, len(samples) - 1) * 1000.0
        if requested_interval is not None
        else None
    )
    long_window_reached = window_ms >= 10_000 or (
        requested_window_ms is not None
        and requested_window_ms >= 10_000
        and window_ms >= requested_window_ms * 0.95
    )
    coverage = len(busy_vertices) / len(compute_vertices) if compute_vertices else 0.0
    rate_coverage = (
        len(rate_vertices) / len(compute_vertices) if compute_vertices else 0.0
    )
    runtime_coverage = (
        len(runtime_vertices) / len(compute_vertices) if compute_vertices else 0.0
    )
    if len(samples) >= 5 and long_window_reached and coverage == 1.0:
        quality = "strong"
    elif (len(samples) >= 2 and coverage == 1.0) or (
        len(samples) >= 5 and long_window_reached and runtime_coverage == 1.0
    ):
        quality = "moderate"
    elif samples and (runtime or not compute_vertices):
        quality = "limited"
    else:
        quality = "insufficient"
    return {
        "quality": quality,
        "sample_count": len(samples),
        "sample_window_ms": window_ms,
        "requested_sample_window_ms": round(requested_window_ms)
        if requested_window_ms is not None
        else None,
        "long_window_reached": long_window_reached,
        "compute_vertex_count": len(compute_vertices),
        "compute_vertices_with_busy_metrics": len(busy_vertices),
        "compute_busy_metric_coverage": round(coverage, 6),
        "compute_vertices_with_rate_metrics": len(rate_vertices),
        "compute_rate_metric_coverage": round(rate_coverage, 6),
        "compute_vertices_with_runtime_metrics": len(runtime_vertices),
        "compute_runtime_metric_coverage": round(runtime_coverage, 6),
    }


def analyze_bundle(bundle, family_specs):
    spec_by_name = {spec["name"]: spec for spec in family_specs}
    runtime = runtime_by_vertex(bundle)
    vertices = bundle["job"]["vertices"]
    compute_vertices = [vertex for vertex in vertices if is_compute_vertex(vertex)]
    use_runtime = bool(compute_vertices) and all(
        runtime.get(vertex["id"], {}).get("metrics", {}).get("busyTimeMsPerSecond")
        is not None
        for vertex in compute_vertices
    )
    family_weights = defaultdict(float)
    backpressure_counts = Counter()
    for vertex in vertices:
        families = [family for family in vertex["families"] if family != "Unknown"] or [
            "Unknown"
        ]
        if use_runtime:
            busy = (
                runtime.get(vertex["id"], {})
                .get("metrics", {})
                .get("busyTimeMsPerSecond")
            )
            weight = max(0.0, busy or 0.0) * max(1, vertex.get("parallelism") or 1)
        else:
            weight = float(max(1, vertex.get("parallelism") or 1))
        for family in families:
            family_weights[family] += weight / len(families)
        backpressure_counts.update(
            runtime.get(vertex["id"], {}).get("backpressure_levels", [])
        )

    if not family_weights:
        for family, count in bundle["plan"].get("family_counts", {}).items():
            safe_count = finite_number(count)
            if (
                (family in spec_by_name or family == "Unknown")
                and safe_count is not None
                and safe_count >= 0
            ):
                family_weights[family] += float(safe_count)

    compute_weights = {
        family: weight
        for family, weight in family_weights.items()
        if family not in COMPUTE_BOUNDARY_FAMILIES
    }
    total_compute = sum(compute_weights.values())
    candidate_weight = sum(
        weight
        for family, weight in compute_weights.items()
        if spec_by_name.get(family, {}).get("coverage") == "library-provided"
    )
    unknown_weight = compute_weights.get("Unknown", 0.0)
    candidate_share = candidate_weight / total_compute if total_compute > 0 else 0.0
    unknown_share = unknown_weight / total_compute if total_compute > 0 else 0.0

    family_rows = []
    for family, weight in sorted(
        family_weights.items(), key=lambda item: (-item[1], item[0])
    ):
        spec = spec_by_name.get(family)
        family_rows.append(
            {
                "family": family,
                "weight": round(weight, 6),
                "share": round(weight / sum(family_weights.values()), 6)
                if sum(family_weights.values()) > 0
                else 0.0,
                "coverage": spec["coverage"] if spec else "unknown",
                "review_priority": spec["review_priority"] if spec else "unknown",
                "review_checks": list(spec["review_checks"]) if spec else [],
                "library_api": spec["library_api"] if spec else None,
            }
        )

    if total_compute <= 0:
        priority = "unknown"
        next_action = "collect_a_job_with_compute_operators_and_runtime_metrics"
    elif unknown_share >= UNKNOWN_COMPUTE_SHARE_THRESHOLD:
        priority = "unknown"
        next_action = "provide_explain_output_and_expand_the_capability_inventory"
    elif candidate_share >= HIGH_CANDIDATE_SHARE_THRESHOLD:
        priority = "high"
        next_action = "rewrite_candidate_segments_with_gpu_library_and_compare_with_cpu"
    elif candidate_share >= MEDIUM_CANDIDATE_SHARE_THRESHOLD:
        priority = "medium"
        next_action = "review_hot_operator_shapes_then_compare_cpu_and_gpu"
    else:
        priority = "low"
        next_action = "focus_on_cpu_side_or_unclassified_bottlenecks_before_gpu_testing"

    shape_risk_counts = Counter()

    def add_shape_risks(entries):
        if not isinstance(entries, list):
            return
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or entry.get("code") not in PLAN_RISK_PATTERNS
            ):
                continue
            mentions = non_negative_int(entry.get("mentions"))
            if mentions is not None:
                shape_risk_counts[entry["code"]] += mentions

    add_shape_risks(bundle["plan"].get("shape_risks", []))
    if "explain" in bundle:
        add_shape_risks(bundle["explain"].get("shape_risks", []))
    shape_risks = [
        {"code": code, "mentions": count}
        for code, count in sorted(shape_risk_counts.items())
    ]
    shape_review_required = bool(shape_risks)
    if priority == "high" and shape_review_required:
        next_action = "review_operator_shapes_then_compare_cpu_and_gpu"

    evidence = runtime_evidence_summary(bundle, runtime)
    qualification_reasons = []
    if total_compute <= 0:
        qualification_reasons.append("no_compute_operator_evidence")
    elif unknown_share >= UNKNOWN_COMPUTE_SHARE_THRESHOLD:
        qualification_reasons.append("at_least_half_of_compute_is_unclassified")
    elif candidate_share >= HIGH_CANDIDATE_SHARE_THRESHOLD:
        qualification_reasons.append("candidate_families_dominate_compute_evidence")
    elif candidate_share >= MEDIUM_CANDIDATE_SHARE_THRESHOLD:
        qualification_reasons.append("candidate_families_are_a_material_compute_share")
    else:
        qualification_reasons.append("candidate_families_are_a_small_compute_share")
    if evidence["quality"] in {"limited", "insufficient"}:
        qualification_reasons.append("runtime_evidence_is_incomplete")
    if shape_risks:
        qualification_reasons.append("shape_risk_signals_require_review")

    limitations = [
        "A family match is not proof that the concrete expressions, types, changelog, or window semantics are supported.",
        "No GPU speedup is predicted; only a controlled CPU/GPU comparison on the same job can measure it.",
        "REST metric samples are diagnostic snapshots, not a replacement for a retained Prometheus time series.",
        "Source, sink, network exchange, UDF, Python, and arbitrary DataStream code are outside the library operator inventory; pure DataStream jobs (user-named operators) cannot be classified.",
    ]
    return {
        "schema": ANALYSIS_SCHEMA,
        "priority": priority,
        "next_action": next_action,
        "assessment_scope": "operator-family-candidate-only",
        "exact_gpu_support_verified": False,
        "shape_review_required": shape_review_required,
        "shape_risks": shape_risks,
        "qualification_reasons": qualification_reasons,
        "evidence_basis": "runtime-busy-time"
        if use_runtime
        else "plan-and-parallelism",
        "evidence": evidence,
        "candidate_compute_share": round(candidate_share, 6),
        "unknown_compute_share": round(unknown_share, 6),
        "family_breakdown": family_rows,
        "backpressure_samples": dict(sorted(backpressure_counts.items())),
        "limitations": limitations,
    }


def distribution_summary(values):
    clean = [float(value) for value in values if finite_number(value) is not None]
    if not clean:
        return None
    return {
        "samples": len(clean),
        "min": round(min(clean), 6),
        "p50": round(float(percentile(clean, 0.50)), 6),
        "p95": round(float(percentile(clean, 0.95)), 6),
        "max": round(max(clean), 6),
        "avg": round(mean(clean), 6),
    }


def sample_metric_value(vertex_sample, name, parallelism):
    subtasks = vertex_sample.get("subtasks", [])
    subtask_values = []
    for subtask in subtasks:
        value = metric_stat(subtask.get("metrics", {}).get(name))
        if value is not None:
            subtask_values.append(value)
    complete_subtasks = bool(subtask_values) and (
        parallelism is None or len(subtask_values) >= max(1, parallelism)
    )
    if complete_subtasks:
        if name in ADDITIVE_RUNTIME_METRICS:
            return sum(subtask_values)
        return mean(subtask_values)
    statistics = vertex_sample.get("metrics", {}).get(name)
    preferred = (
        ("sum", "value", "avg", "max")
        if name in ADDITIVE_RUNTIME_METRICS
        else ("avg", "value", "max", "sum")
    )
    return metric_stat(statistics, preferred=preferred)


def vertex_runtime_profile(vertex, runtime_samples):
    vertex_id = vertex.get("id")
    parallelism = non_negative_int(vertex.get("parallelism"))
    samples = []
    for sample in runtime_samples:
        for candidate in sample.get("vertices", []):
            if candidate.get("id") == vertex_id:
                samples.append(candidate)
                break

    metrics = {}
    metric_sources = {}
    for name in RUNTIME_METRICS:
        values = [
            value
            for sample in samples
            if (value := sample_metric_value(sample, name, parallelism)) is not None
        ]
        summary = distribution_summary(values)
        if summary is not None:
            metrics[name] = summary
            metric_sources[name] = "runtime-samples"

    lifetime = vertex.get("lifetime_metrics", {})
    duration_ms = non_negative_int(vertex.get("duration_ms"))
    for name in ("numRecordsIn", "numRecordsOut", "numBytesIn", "numBytesOut"):
        if name in metrics:
            continue
        value = metric_stat(lifetime.get(name), preferred=("value", "sum", "avg"))
        if value is not None:
            metrics[name] = distribution_summary([value])
            metric_sources[name] = "job-lifetime-counter"
    if duration_ms not in {None, 0}:
        lifetime_time_mapping = {
            "busyTimeMsPerSecond": "accumulatedBusyTimeMs",
            "backPressuredTimeMsPerSecond": "accumulatedBackpressuredTimeMs",
            "idleTimeMsPerSecond": "accumulatedIdleTimeMs",
        }
        for output_name, lifetime_name in lifetime_time_mapping.items():
            if output_name in metrics:
                continue
            value = metric_stat(
                lifetime.get(lifetime_name), preferred=("value", "sum", "avg")
            )
            if value is not None:
                metrics[output_name] = distribution_summary(
                    [min(1000.0, max(0.0, value / duration_ms * 1000.0))]
                )
                metric_sources[output_name] = "derived-from-job-lifetime"
        rate_mapping = {
            "numRecordsInPerSecond": "numRecordsIn",
            "numRecordsOutPerSecond": "numRecordsOut",
        }
        for output_name, counter_name in rate_mapping.items():
            if output_name in metrics:
                continue
            value = metric_stat(
                lifetime.get(counter_name), preferred=("value", "sum", "avg")
            )
            if value is not None:
                metrics[output_name] = distribution_summary(
                    [value * 1000.0 / duration_ms]
                )
                metric_sources[output_name] = "derived-from-job-lifetime"

    levels = Counter()
    ratio_values = []
    for sample in samples:
        backpressure = sample.get("backpressure", {})
        level = backpressure.get("level")
        if level in {"ok", "low", "high"}:
            levels[level] += 1
        ratio = backpressure.get("ratios", {}).get("backpressureratio", {})
        ratio_value = metric_stat(ratio)
        if ratio_value is not None:
            ratio_values.append(ratio_value)

    subtask_series = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        for subtask in sample.get("subtasks", []):
            index = non_negative_int(subtask.get("index"))
            if index is None or index > MAX_SUBTASK_INDEX:
                continue
            for name, statistics in subtask.get("metrics", {}).items():
                if name not in RUNTIME_METRICS:
                    continue
                value = metric_stat(statistics)
                if value is not None:
                    subtask_series[index][name].append(value)
    subtask_rows = []
    for index, series in sorted(subtask_series.items()):
        subtask_rows.append(
            {
                "index": index,
                "metrics": {
                    name: summary
                    for name, values in sorted(series.items())
                    if (summary := distribution_summary(values)) is not None
                },
            }
        )

    expected_subtasks = max(1, parallelism or 1)
    subtask_coverage = min(1.0, len(subtask_rows) / expected_subtasks)
    skew = {
        "status": "insufficient-evidence",
        "metric": None,
        "observed_subtasks": len(subtask_rows),
        "expected_subtasks": expected_subtasks,
        "coverage": round(subtask_coverage, 6),
        "max_to_mean_ratio": None,
    }
    if len(subtask_rows) >= 2 and subtask_coverage >= 0.80:
        for candidate in (
            "busyTimeMsPerSecond",
            "numRecordsOutPerSecond",
            "numRecordsInPerSecond",
        ):
            values = [
                row["metrics"][candidate]["avg"]
                for row in subtask_rows
                if candidate in row["metrics"]
            ]
            if len(values) != len(subtask_rows) or mean(values) in {None, 0}:
                continue
            ratio = max(values) / mean(values)
            spread = max(values) - min(values)
            threshold = 200.0 if candidate in TIME_PER_SECOND_METRICS else 0.0
            skew = {
                "status": "detected"
                if ratio >= 1.50 and spread >= threshold
                else "not-detected",
                "metric": candidate,
                "observed_subtasks": len(subtask_rows),
                "expected_subtasks": expected_subtasks,
                "coverage": round(subtask_coverage, 6),
                "max_to_mean_ratio": round(ratio, 6),
            }
            break

    busy = metrics.get("busyTimeMsPerSecond", {}).get("avg")
    idle = metrics.get("idleTimeMsPerSecond", {}).get("avg")
    backpressured = metrics.get("backPressuredTimeMsPerSecond", {}).get("avg")
    if backpressured is None and ratio_values:
        backpressured = mean(ratio_values) * 1000.0
    if backpressured is not None and backpressured >= 500:
        dominant_state = "backpressured"
    elif busy is not None and busy >= 700:
        dominant_state = "busy"
    elif idle is not None and idle >= 700:
        dominant_state = "idle"
    elif any(value is not None for value in (busy, idle, backpressured)):
        dominant_state = "mixed"
    else:
        dominant_state = "unknown"
    pressure = sum(value or 0.0 for value in (busy, backpressured))
    pressure_score = min(100.0, max(0.0, pressure / 10.0))

    pending_values = [
        value
        for sample in samples
        if (value := sample_metric_value(sample, "pendingRecords", parallelism))
        is not None
    ]
    pending_trend = None
    if len(pending_values) >= 2:
        pending_trend = {
            "first": round(pending_values[0], 6),
            "last": round(pending_values[-1], 6),
            "delta": round(pending_values[-1] - pending_values[0], 6),
        }

    return {
        "id": vertex_id,
        "families": vertex.get("families", []),
        "parallelism": parallelism,
        "state": vertex.get("state", "UNKNOWN"),
        "sample_count": len(samples),
        "metrics": metrics,
        "metric_sources": metric_sources,
        "backpressure_levels": dict(sorted(levels.items())),
        "backpressure_ratio": distribution_summary(ratio_values),
        "subtask_skew": skew,
        "subtasks": subtask_rows,
        "pending_records_trend": pending_trend,
        "dominant_state": dominant_state,
        "pressure_score": round(pressure_score, 6),
    }


def checkpoint_profile(bundle):
    checkpoints = bundle.get("checkpoints", {})
    config = bundle.get("checkpoint_config", {})
    if not isinstance(checkpoints, dict):
        checkpoints = {}
    if not isinstance(config, dict):
        config = {}
    raw_counts = checkpoints.get("counts", {})
    if not isinstance(raw_counts, dict):
        raw_counts = {}
    counts = {
        key: value
        for key in ("total", "in_progress", "completed", "failed", "restored")
        if (value := non_negative_int(raw_counts.get(key))) is not None
    }
    failed = non_negative_int(counts.get("failed")) or 0
    raw_duration = checkpoints.get("duration_ms", {})
    if not isinstance(raw_duration, dict):
        raw_duration = {}
    duration = {
        key: value
        for key in ("min", "p50", "p95", "max")
        if (value := finite_number(raw_duration.get(key))) is not None and value >= 0
    }
    raw_size = checkpoints.get("checkpointed_size_bytes", {})
    if not isinstance(raw_size, dict):
        raw_size = {}
    size = {
        key: value
        for key in ("min", "p50", "p95", "max")
        if (value := finite_number(raw_size.get(key))) is not None and value >= 0
    }
    enabled = config.get("enabled") if isinstance(config.get("enabled"), bool) else None
    p95 = finite_number(duration.get("p95"))
    interval = finite_number(config.get("interval"))
    ratio = p95 / interval if p95 is not None and interval not in {None, 0} else None
    if failed > 0 or (ratio is not None and ratio >= 1.0):
        health = "needs-attention"
    elif enabled is False:
        health = "disabled"
    elif ratio is not None and ratio >= 0.50:
        health = "watch"
    elif counts or duration or size:
        health = "healthy"
    else:
        health = "unavailable"
    return {
        "health": health,
        "enabled": enabled,
        "counts": counts,
        "duration_ms": duration or None,
        "checkpointed_size_bytes": size or None,
        "configured_interval_ms": interval,
        "p95_duration_to_interval_ratio": round(ratio, 6)
        if ratio is not None
        else None,
    }


def analyze_profile(bundle, qualification):
    runtime_samples = bundle.get("runtime_samples", [])
    vertices = [
        vertex_runtime_profile(vertex, runtime_samples)
        for vertex in bundle.get("job", {}).get("vertices", [])
    ]
    evidence = runtime_evidence_summary(bundle)
    findings = []

    def add_finding(code, severity, summary, vertex=None, evidence_values=None):
        finding = {"code": code, "severity": severity, "summary": summary}
        if vertex is not None:
            finding["vertex_id"] = vertex["id"]
            finding["families"] = vertex["families"]
        if evidence_values:
            finding["evidence"] = evidence_values
        findings.append(finding)

    if evidence["sample_count"] < 3:
        add_finding(
            "short_sampling_window",
            "warning",
            "Fewer than three runtime samples cannot show a stable performance pattern.",
            evidence_values={
                "sample_count": evidence["sample_count"],
                "sample_window_ms": evidence["sample_window_ms"],
            },
        )
    if evidence["compute_busy_metric_coverage"] < 1.0:
        add_finding(
            "incomplete_busy_metrics",
            "warning",
            "Busy-time metrics are missing for one or more compute vertices.",
            evidence_values={
                "coverage": evidence["compute_busy_metric_coverage"],
                "covered": evidence["compute_vertices_with_busy_metrics"],
                "total": evidence["compute_vertex_count"],
            },
        )

    for vertex in vertices:
        metrics = vertex["metrics"]
        busy = metrics.get("busyTimeMsPerSecond", {}).get("avg")
        backpressured = metrics.get("backPressuredTimeMsPerSecond", {}).get("avg")
        if backpressured is None and vertex["backpressure_ratio"] is not None:
            backpressured = vertex["backpressure_ratio"]["avg"] * 1000.0
        levels = vertex["backpressure_levels"]
        if levels.get("high", 0) > 0 or (backpressured or 0) >= 500:
            add_finding(
                "high_backpressure",
                "critical",
                "This vertex is frequently blocked because downstream cannot keep up.",
                vertex,
                {
                    "backpressured_ms_per_second": round(backpressured or 0.0, 6),
                    "levels": levels,
                },
            )
        elif levels.get("low", 0) > 0 or (backpressured or 0) > 100:
            add_finding(
                "backpressure_present",
                "warning",
                "This vertex shows material downstream backpressure.",
                vertex,
                {
                    "backpressured_ms_per_second": round(backpressured or 0.0, 6),
                    "levels": levels,
                },
            )
        if busy is not None and busy >= 850 and (backpressured or 0) < 100:
            add_finding(
                "compute_saturation",
                "warning",
                "This vertex spends most of the sample window doing work.",
                vertex,
                {"busy_ms_per_second": round(busy, 6)},
            )
        if vertex["subtask_skew"]["status"] == "detected":
            add_finding(
                "subtask_skew",
                "warning",
                "Parallel subtasks carry materially different load.",
                vertex,
                vertex["subtask_skew"],
            )
        trend = vertex.get("pending_records_trend")
        if trend is not None and trend["delta"] > 0 and trend["last"] > 0:
            add_finding(
                "pending_work_is_growing",
                "warning",
                "Pending records grew during the sampling window.",
                vertex,
                trend,
            )

    checkpoint = checkpoint_profile(bundle)
    if checkpoint["health"] == "needs-attention":
        add_finding(
            "checkpoint_health",
            "critical",
            "Checkpoint failures or duration indicate a reliability bottleneck.",
            evidence_values={
                "failed": checkpoint["counts"].get("failed", 0),
                "p95_duration_to_interval_ratio": checkpoint[
                    "p95_duration_to_interval_ratio"
                ],
            },
        )
    elif checkpoint["health"] == "watch":
        add_finding(
            "checkpoint_overhead",
            "warning",
            "Checkpoint p95 duration consumes at least half of its configured interval.",
            evidence_values={
                "p95_duration_to_interval_ratio": checkpoint[
                    "p95_duration_to_interval_ratio"
                ]
            },
        )
    elif checkpoint["health"] == "disabled":
        add_finding(
            "checkpointing_disabled",
            "info",
            "Periodic checkpointing is disabled; confirm that this matches production policy.",
        )
    exception_count = (
        non_negative_int(bundle.get("failures", {}).get("exception_count")) or 0
    )
    if exception_count:
        add_finding(
            "job_exceptions_observed",
            "critical",
            "The job has recorded exceptions; performance results may include recovery work.",
            evidence_values={"exception_count": exception_count},
        )
    if qualification.get("shape_risks"):
        add_finding(
            "operator_shape_risks",
            "warning",
            "The plan contains features that require review against the library operator limits.",
            evidence_values={"signals": qualification["shape_risks"]},
        )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(
        key=lambda finding: (
            severity_order.get(finding["severity"], 99),
            finding.get("vertex_id", ""),
            finding["code"],
        )
    )
    critical_count = sum(finding["severity"] == "critical" for finding in findings)
    warning_count = sum(finding["severity"] == "warning" for finding in findings)
    if critical_count:
        status = "needs-attention"
    elif evidence["quality"] in {"insufficient", "limited"}:
        status = "insufficient-evidence"
    elif warning_count:
        status = "review-findings"
    else:
        status = "no-immediate-bottleneck-detected"

    hotspots = []
    for vertex in sorted(
        vertices, key=lambda vertex: (-vertex["pressure_score"], vertex["id"])
    )[:10]:
        metrics = vertex["metrics"]
        backpressured = metrics.get("backPressuredTimeMsPerSecond", {}).get("avg")
        if backpressured is None and vertex["backpressure_ratio"] is not None:
            backpressured = vertex["backpressure_ratio"]["avg"] * 1000.0
        hotspots.append(
            {
                "vertex_id": vertex["id"],
                "families": vertex["families"],
                "parallelism": vertex["parallelism"],
                "pressure_score": vertex["pressure_score"],
                "dominant_state": vertex["dominant_state"],
                "busy_ms_per_second": metrics.get("busyTimeMsPerSecond", {}).get("avg"),
                "backpressured_ms_per_second": backpressured,
                "records_in_per_second": metrics.get("numRecordsInPerSecond", {}).get(
                    "avg"
                ),
                "records_out_per_second": metrics.get("numRecordsOutPerSecond", {}).get(
                    "avg"
                ),
                "subtask_skew": vertex["subtask_skew"],
            }
        )

    recommendation_codes = []
    finding_codes = {finding["code"] for finding in findings}
    if "high_backpressure" in finding_codes or "backpressure_present" in finding_codes:
        recommendation_codes.append("inspect_downstream_of_backpressured_vertices")
    if "compute_saturation" in finding_codes:
        recommendation_codes.append("run_targeted_parallelism_and_cpu_gpu_comparison")
    if "subtask_skew" in finding_codes:
        recommendation_codes.append("inspect_partition_keys_and_hot_keys")
    if "checkpoint_health" in finding_codes or "checkpoint_overhead" in finding_codes:
        recommendation_codes.append("inspect_checkpoint_alignment_state_and_storage")
    if "checkpointing_disabled" in finding_codes:
        recommendation_codes.append("confirm_checkpoint_policy_matches_production")
    if evidence["sample_count"] < 5 or not evidence["long_window_reached"]:
        recommendation_codes.append("collect_at_least_five_samples_over_ten_seconds")
    if evidence["compute_busy_metric_coverage"] < 1.0:
        recommendation_codes.append(
            "enable_busy_time_metrics_or_add_prometheus_profiler_evidence"
        )
    if qualification.get("priority") == "high":
        recommendation_codes.append(
            "rewrite_candidate_segments_with_gpu_library_and_compare_with_cpu"
        )

    cluster = bundle.get("cluster", {})
    if not isinstance(cluster, dict):
        cluster = {}
    resource_context = {
        key: value
        for key in ("taskmanagers", "slots_total", "slots_available")
        if (value := non_negative_int(cluster.get(key))) is not None
    }
    resource_context["vertex_parallelism_total"] = sum(
        vertex.get("parallelism") or 0
        for vertex in bundle.get("job", {}).get("vertices", [])
    )

    raw_version = bundle.get("collector_version")
    collector_version = (
        raw_version
        if isinstance(raw_version, str) and SAFE_VERSION_RE.fullmatch(raw_version)
        else "unknown"
    )
    raw_job_state = bundle.get("job", {}).get("state")

    return {
        "schema": PROFILE_SCHEMA,
        "collector_version": collector_version,
        "status": status,
        "job": {
            "state": safe_state(raw_job_state),
            "duration_ms": non_negative_int(bundle.get("job", {}).get("duration_ms")),
        },
        "resource_context": resource_context,
        "evidence": evidence,
        "finding_counts": {
            "critical": critical_count,
            "warning": warning_count,
            "info": sum(finding["severity"] == "info" for finding in findings),
        },
        "hotspots": hotspots,
        "vertices": vertices,
        "checkpoint": checkpoint,
        "findings": findings,
        "recommendations": list(dict.fromkeys(recommendation_codes)),
        "qualification_summary": {
            "priority": qualification.get("priority"),
            "candidate_compute_share": qualification.get("candidate_compute_share"),
            "exact_gpu_support_verified": qualification.get(
                "exact_gpu_support_verified", False
            ),
            "shape_review_required": qualification.get("shape_review_required", True),
        },
        "limitations": [
            "Short REST samples do not replace a retained Prometheus time series.",
            "Backpressure is observed on the upstream producer; the slow component is usually downstream.",
            "A chained Flink vertex may contain multiple operator families that cannot be timed separately here.",
            "Throughput is only comparable across runs when input, parallelism, hardware, and sampling windows match.",
            "Method-level CPU, allocation, lock, native, and GPU-kernel costs require profiler traces not collected here.",
        ],
    }


def markdown_report(bundle, analysis):
    checkpoint_entries = (
        non_negative_int(bundle.get("checkpoints", {}).get("history_entries")) or 0
    )
    exception_count = (
        non_negative_int(bundle.get("failures", {}).get("exception_count")) or 0
    )
    lines = [
        "# Flink workload review",
        "",
        "## Result",
        "",
        f"- Review priority: `{analysis['priority']}`",
        f"- Next action: `{analysis['next_action']}`",
        f"- Evidence basis: `{analysis['evidence_basis']}`",
        f"- Evidence quality: `{analysis['evidence']['quality']}`",
        f"- Candidate compute share: `{analysis['candidate_compute_share'] * 100:.1f}%`",
        "",
        (
            "A library-provided family only means the GPU library has an operator for this "
            "node type. It does not mean this exact SQL can be rewritten to use it."
        ),
        "",
        "## Operator families",
        "",
        "| Family | Evidence share | Library status | Library API | Review priority |",
        "| --- | ---: | --- | --- | --- |",
    ]
    lines.extend(
        (
            f"| `{row['family']}` | {row['share'] * 100:.1f}% | "
            f"{row['coverage']} | `{row['library_api'] or '-'}` | {row['review_priority']} |"
        )
        for row in analysis["family_breakdown"]
    )
    if not analysis["family_breakdown"]:
        lines.append("| `Unknown` | 0.0% | insufficient evidence | - | unknown |")
    lines.extend(
        [
            "",
            "## Runtime evidence",
            "",
            f"- Job state: `{bundle['job']['state']}`",
            f"- Sanitized vertices: `{len(bundle['job']['vertices'])}`",
            f"- Runtime samples: `{len(bundle['runtime_samples'])}`",
            f"- Checkpoint history entries: `{checkpoint_entries}`",
            f"- Exception count: `{exception_count}`",
            "",
            "## Privacy boundary",
            "",
            (
                "The bundle does not retain raw SQL, REST URLs, credentials, job/operator names, "
                "table/field names, exception text, or user configuration. Names and IDs are "
                "represented only by HMAC fingerprints or local synthetic IDs."
            ),
            "",
            "## Important limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in analysis["limitations"])
    if bundle["warnings"]:
        lines.extend(["", "## Missing evidence", ""])
        lines.extend(f"- {warning}" for warning in bundle["warnings"])
    lines.append("")
    return "\n".join(lines)


def profile_markdown_report(bundle, qualification, profile):
    def number_text(value):
        return "n/a" if value is None else f"{value:,.1f}"

    def skew_text(skew):
        ratio = skew.get("max_to_mean_ratio")
        ratio_text = "" if ratio is None else f" ({ratio:.3f}x)"
        return f"{skew['status']}{ratio_text}"

    evidence = profile["evidence"]
    lines = [
        "# Flink performance profile",
        "",
        "## Summary",
        "",
        f"- Profile status: `{profile['status']}`",
        f"- Evidence quality: `{evidence['quality']}`",
        f"- Runtime samples: `{evidence['sample_count']}`",
        f"- Sampling window: `{evidence['sample_window_ms']} ms`",
        f"- Qualification priority: `{qualification['priority']}`",
        f"- Candidate compute share: `{qualification['candidate_compute_share'] * 100:.1f}%`",
        "- Exact GPU support verified: `no`",
        "",
        "## Hot vertices",
        "",
        (
            "| Vertex | Families | Parallelism | Busy ms/s | Backpressure ms/s | "
            "Input records/s | Output records/s | Subtask skew | State |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for hotspot in profile["hotspots"]:
        families = ", ".join(hotspot["families"]) or "Unknown"
        lines.append(
            f"| `{hotspot['vertex_id']}` | `{families}` | "
            f"{hotspot['parallelism'] or 'n/a'} | "
            f"{number_text(hotspot['busy_ms_per_second'])} | "
            f"{number_text(hotspot['backpressured_ms_per_second'])} | "
            f"{number_text(hotspot['records_in_per_second'])} | "
            f"{number_text(hotspot['records_out_per_second'])} | "
            f"`{skew_text(hotspot['subtask_skew'])}` | "
            f"`{hotspot['dominant_state']}` |"
        )
    if not profile["hotspots"]:
        lines.append(
            "| none | `Unknown` | n/a | n/a | n/a | n/a | n/a | "
            "`insufficient-evidence` | `unknown` |"
        )

    resources = profile["resource_context"]
    lines.extend(
        [
            "",
            "## Runtime context",
            "",
            f"- Job state: `{profile['job']['state']}`",
            f"- Job duration: `{number_text(profile['job']['duration_ms'])} ms`",
            f"- TaskManagers: `{resources.get('taskmanagers', 'n/a')}`",
            (
                f"- Total / available slots: `{resources.get('slots_total', 'n/a')}` / "
                f"`{resources.get('slots_available', 'n/a')}`"
            ),
            f"- Sum of vertex parallelism: `{resources['vertex_parallelism_total']}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            "| Severity | Finding | Vertex | Summary |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        (
            f"| `{finding['severity']}` | `{finding['code']}` | "
            f"`{finding.get('vertex_id', '-')}` | {finding['summary']} |"
        )
        for finding in profile["findings"]
    )
    if not profile["findings"]:
        lines.append("| `info` | `none` | `-` | No immediate finding. |")

    checkpoint = profile["checkpoint"]
    checkpoint_duration = checkpoint["duration_ms"] or {}
    checkpoint_size = checkpoint["checkpointed_size_bytes"] or {}
    checkpoint_enabled = (
        "yes"
        if checkpoint["enabled"] is True
        else "no"
        if checkpoint["enabled"] is False
        else "n/a"
    )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            f"- Health: `{checkpoint['health']}`",
            f"- Enabled: `{checkpoint_enabled}`",
            (
                f"- Total / completed: `{checkpoint['counts'].get('total', 0)}` / "
                f"`{checkpoint['counts'].get('completed', 0)}`"
            ),
            f"- Failed: `{checkpoint['counts'].get('failed', 0)}`",
            f"- Configured interval: `{number_text(checkpoint['configured_interval_ms'])} ms`",
            f"- P95 duration: `{number_text(checkpoint_duration.get('p95'))} ms`",
            (
                "- P95 checkpointed size: `"
                f"{number_text(checkpoint_size.get('p95'))} bytes`"
            ),
            (
                "- P95 duration / configured interval: `"
                + (
                    f"{checkpoint['p95_duration_to_interval_ratio']:.3f}"
                    if checkpoint["p95_duration_to_interval_ratio"] is not None
                    else "not available"
                )
                + "`"
            ),
            "",
            "## Recommended next actions",
            "",
        ]
    )
    if profile["recommendations"]:
        lines.extend(f"- `{code}`" for code in profile["recommendations"])
    else:
        lines.append("- `keep_monitoring`")
    lines.extend(
        [
            "",
            "## Privacy boundary",
            "",
            (
                "This report retains no raw SQL, REST URL, credential, job/operator name, "
                "table/field name, exception text, or user configuration."
            ),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in profile["limitations"])
    if bundle.get("warnings"):
        lines.extend(["", "## Missing evidence", ""])
        lines.extend(f"- {warning}" for warning in bundle["warnings"])
    lines.append("")
    return "\n".join(lines)


def secure_write(path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        encoded = data.encode("utf-8") if isinstance(data, str) else data
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_named_outputs(output_dir, files):
    output = Path(output_dir).absolute()
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        fail("--output-dir already exists; refusing to overwrite evidence")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        os.chmod(temporary, 0o700)
        if not files or any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) is None
            for name in files
        ):
            fail("output file set is empty or contains an invalid name")
        for name, content in files.items():
            if not isinstance(content, str):
                fail(f"output file is not text: {name}")
            secure_write(temporary / name, content)
        manifest = "".join(
            f"{hashlib.sha256(content.encode('utf-8')).hexdigest()}  {name}\n"
            for name, content in sorted(files.items())
        )
        secure_write(temporary / "SHA256SUMS", manifest)
        if output.exists() or output.is_symlink():
            fail("--output-dir appeared during collection; refusing to overwrite it")
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_outputs(output_dir, bundle, analysis):
    return write_named_outputs(
        output_dir,
        {
            "workload.json": json_text(bundle),
            "analysis.json": json_text(analysis),
            "report.md": markdown_report(bundle, analysis),
        },
    )


def environment_secret(env_name, label, minimum_length):
    if env_name is None:
        return None
    if ENV_NAME_RE.fullmatch(env_name) is None:
        fail(f"{label} must name a valid environment variable")
    value = os.environ.get(env_name)
    if value is None:
        fail(f"environment variable named by {label} is not set")
    if len(value.encode("utf-8")) < minimum_length:
        fail(f"environment variable named by {label} is too short")
    return value


def parse_args():
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect a pseudonymized Flink plan/runtime bundle and produce a conservative "
            "Flink workload review for the FEP GPU library. The collector only performs HTTP GET requests."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--rest-url", help="Flink JobManager or History Server base URL"
    )
    source.add_argument(
        "--snapshot-file", help=f"offline raw snapshot using schema {SNAPSHOT_SCHEMA}"
    )
    source.add_argument(
        "--archive-file",
        help="raw job archive written by Flink's FsJobArchivist for History Server",
    )
    parser.add_argument("--job-id", help="32-character hexadecimal Flink job id")
    parser.add_argument(
        "--archive-flink-version",
        help="Flink version that wrote --archive-file; currently must be 2.2.1",
    )
    parser.add_argument(
        "--output-dir", required=True, help="new directory for sanitized output"
    )
    parser.add_argument(
        "--capabilities",
        default=str(project_dir / "qualification/flink-2.2.1-operator-families.json"),
        help="family-level operator inventory",
    )
    parser.add_argument(
        "--explain-file",
        help="optional Flink EXPLAIN text/JSON; only a structural summary is retained",
    )
    parser.add_argument("--sample-count", type=positive_int, default=1)
    parser.add_argument(
        "--sample-interval-seconds",
        type=positive_float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--timeout-seconds", type=positive_float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--startup-wait-seconds",
        type=non_negative_float,
        default=DEFAULT_STARTUP_WAIT_SECONDS,
        help="wait for newly submitted vertices to finish deployment before sampling",
    )
    parser.add_argument(
        "--authorization-env",
        help="environment variable containing the complete Authorization header value",
    )
    parser.add_argument(
        "--pseudonym-key-env",
        help="environment variable containing a stable pseudonymization secret (min 16 bytes)",
    )
    parser.add_argument("--ca-file", help="custom CA bundle for an HTTPS REST endpoint")
    parser.add_argument("--version", action="version", version=COLLECTOR_VERSION)
    return parser.parse_args()


def collect_bundle(args):
    if args.rest_url:
        if args.job_id is None or JOB_ID_RE.fullmatch(args.job_id) is None:
            fail(
                "--job-id is required for REST collection and must be 32 hexadecimal characters"
            )
        if (
            args.sample_count - 1
        ) * args.sample_interval_seconds > MAX_COLLECTION_SECONDS:
            fail(f"requested sampling window exceeds {MAX_COLLECTION_SECONDS} seconds")
    elif args.sample_count != 1:
        fail(
            "--sample-count is only valid with --rest-url; offline inputs carry their own samples"
        )
    if bool(args.archive_file) != bool(args.archive_flink_version):
        fail("--archive-flink-version is required exactly when --archive-file is used")
    if args.ca_file and not args.rest_url:
        fail("--ca-file is only valid with --rest-url")
    authorization = environment_secret(args.authorization_env, "--authorization-env", 1)
    pseudonym_secret = environment_secret(
        args.pseudonym_key_env, "--pseudonym-key-env", 16
    )
    pseudonymizer = Pseudonymizer(
        pseudonym_secret.encode("utf-8")
        if pseudonym_secret
        else secrets.token_bytes(32),
        stable=pseudonym_secret is not None,
    )
    capability_doc, family_specs = load_capabilities(args.capabilities)
    if args.rest_url:
        raw, warnings, job_id = collect_rest(args, authorization)
    elif args.archive_file:
        if authorization is not None:
            fail("--authorization-env is only valid with --rest-url")
        raw, warnings, job_id = collect_history_archive(args, capability_doc)
    else:
        if authorization is not None:
            fail("--authorization-env is only valid with --rest-url")
        raw, warnings, job_id = collect_snapshot(args)
    raw_explain = None
    if args.explain_file:
        raw_explain = read_stable_text_or_json(args.explain_file, "EXPLAIN file")
    bundle = build_bundle(
        raw,
        raw_explain,
        job_id,
        warnings,
        pseudonymizer,
        capability_doc,
        family_specs,
        args,
    )
    return bundle, family_specs


def main():
    os.umask(0o077)
    args = parse_args()
    bundle, family_specs = collect_bundle(args)
    analysis = analyze_bundle(bundle, family_specs)
    output = write_outputs(args.output_dir, bundle, analysis)
    print(
        "FEP_FLINK_WORKLOAD_COLLECTION_OK "
        f"priority={analysis['priority']} output_dir={output}"
    )


if __name__ == "__main__":
    try:
        main()
    except CollectorError as error:
        print(f"FEP_FLINK_WORKLOAD_COLLECTION_FAILED: {error}", file=sys.stderr)
        sys.exit(1)
