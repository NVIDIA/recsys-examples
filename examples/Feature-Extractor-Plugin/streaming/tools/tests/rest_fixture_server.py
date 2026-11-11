#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local read-only Flink REST fixture used by self_test.py."""

import argparse
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--request-log", required=True)
    return parser.parse_args()


def secure_create(path, text):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, text.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def live_metrics(snapshot, vertex_id, only_subtask=None):
    sample = snapshot.get("runtime_samples", [{}])[0]
    vertex = sample.get("vertices", {}).get(vertex_id, {})
    raw_metrics = vertex.get("metrics")
    if not isinstance(raw_metrics, list):
        return None
    parallelism = next(
        (
            item.get("parallelism", 1)
            for item in snapshot.get("job", {}).get("vertices", [])
            if item.get("id") == vertex_id
        ),
        1,
    )
    subtask_indexes = [only_subtask] if only_subtask is not None else range(parallelism)
    result = []
    for entry in raw_metrics:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        default_value = entry.get(
            "avg", entry.get("value", entry.get("max", entry.get("min", 0)))
        )
        for subtask_index in subtask_indexes:
            subtask_value = default_value
            if (
                vertex_id == "cccccccccccccccccccccccccccccccc"
                and entry["id"] == "busyTimeMsPerSecond"
            ):
                subtask_value = (100, 100, 100, 1000)[subtask_index]
            metric_id = (
                entry["id"]
                if only_subtask is not None
                else f"{subtask_index}.{entry['id']}"
            )
            result.append({"id": metric_id, "value": str(subtask_value)})
    sensitive_id = "secret_customer.Writer.numRecordsIn"
    result.append(
        {
            "id": sensitive_id if only_subtask is not None else f"0.{sensitive_id}",
            "value": "999999",
        }
    )
    return result


class FixtureHandler(BaseHTTPRequestHandler):
    server_version = "FlinkCollectorFixture/1"

    def log_message(self, _format, *_args):
        return

    def record_request(self):
        with self.server.request_log_lock:
            self.server.request_log.write(f"{self.command} {self.path}\n")
            self.server.request_log.flush()

    def respond(self, status, value):
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.record_request()
        if self.headers.get("Authorization") != self.server.expected_authorization:
            self.respond(401, {"error": "unauthorized"})
            return

        parsed_url = urllib.parse.urlsplit(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)
        snapshot = self.server.snapshot
        job_id = snapshot["job"]["jid"]
        job_prefix = f"/jobs/{job_id}"
        exact_routes = {
            "/overview": snapshot.get("overview"),
            job_prefix: snapshot.get("job"),
            job_prefix + "/plan": snapshot.get("plan"),
            job_prefix + "/config": snapshot.get("config"),
            job_prefix + "/checkpoints": snapshot.get("checkpoints"),
            job_prefix + "/checkpoints/config": snapshot.get("checkpoint_config"),
            job_prefix + "/exceptions": snapshot.get("exceptions"),
        }
        if path in exact_routes:
            value = exact_routes[path]
            if value is None:
                self.respond(404, {"error": "fixture unavailable"})
            else:
                self.respond(200, value)
            return

        vertex_prefix = job_prefix + "/vertices/"
        if not path.startswith(vertex_prefix):
            self.respond(404, {"error": "unknown fixture endpoint"})
            return
        suffix = path[len(vertex_prefix) :]
        pieces = suffix.split("/")
        vertex_id = pieces[0]
        if len(pieces) == 1:
            value = snapshot.get("vertex_details", {}).get(vertex_id)
        elif len(pieces) == 2 and pieces[1] in {"metrics", "backpressure"}:
            sample = snapshot.get("runtime_samples", [{}])[0]
            vertex = sample.get("vertices", {}).get(vertex_id, {})
            value = vertex.get(pieces[1])
            if pieces[1] == "metrics" and isinstance(value, list):
                metrics = live_metrics(snapshot, vertex_id)
                requested = {
                    metric_id
                    for request_value in query.get("get", [])
                    for metric_id in request_value.split(",")
                }
                if not requested:
                    with self.server.metric_inventory_lock:
                        attempts = self.server.metric_inventory_attempts.get(
                            vertex_id, 0
                        )
                        self.server.metric_inventory_attempts[vertex_id] = attempts + 1
                    if attempts == 0:
                        # Model a real startup race: Flink can mark a job RUNNING
                        # before the first vertex metric inventory is populated.
                        self.respond(200, [])
                        return
                    if vertex_id == self.server.subtask_fallback_vertex_id:
                        self.respond(200, [])
                        return
                if requested:
                    value = [entry for entry in metrics if entry["id"] in requested]
                else:
                    value = [{"id": entry["id"]} for entry in metrics]
        elif (
            len(pieces) == 4
            and pieces[1] == "subtasks"
            and pieces[2].isdigit()
            and pieces[3] == "metrics"
        ):
            subtask_index = int(pieces[2])
            metrics = live_metrics(snapshot, vertex_id, only_subtask=subtask_index)
            requested = {
                metric_id
                for request_value in query.get("get", [])
                for metric_id in request_value.split(",")
            }
            if isinstance(metrics, list):
                value = (
                    [entry for entry in metrics if entry["id"] in requested]
                    if requested
                    else [{"id": entry["id"]} for entry in metrics]
                )
            else:
                value = None
        else:
            value = None
        if value is None:
            self.respond(404, {"error": "fixture vertex unavailable"})
        else:
            self.respond(200, value)

    def do_POST(self):
        self.record_request()
        self.respond(405, {"error": "read only"})

    def do_DELETE(self):
        self.record_request()
        self.respond(405, {"error": "read only"})

    def do_PATCH(self):
        self.record_request()
        self.respond(405, {"error": "read only"})

    def do_PUT(self):
        self.record_request()
        self.respond(405, {"error": "read only"})


def main():
    args = parse_args()
    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    expected_authorization = os.environ.get("FIXTURE_AUTHORIZATION")
    if not expected_authorization:
        raise SystemExit("FIXTURE_AUTHORIZATION is required")
    with open(args.request_log, "x", encoding="utf-8") as request_log:
        os.chmod(args.request_log, 0o600)
        with ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler) as server:
            server.snapshot = snapshot
            server.expected_authorization = expected_authorization
            server.request_log = request_log
            server.request_log_lock = threading.Lock()
            server.metric_inventory_attempts = {}
            server.metric_inventory_lock = threading.Lock()
            server.subtask_fallback_vertex_id = snapshot["job"]["vertices"][0]["id"]
            secure_create(args.port_file, str(server.server_address[1]) + "\n")
            server.serve_forever()


if __name__ == "__main__":
    main()
