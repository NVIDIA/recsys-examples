<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# fep-flink-tools: qualification and profiling for Flink SQL jobs

Two read-only commands that answer: **which parts of an existing Flink SQL job could be rewritten to call the GPU library, and where does the job actually spend its time?** Python 3 standard library only. No GPU, no Flink jar, no network access except the REST calls you point it at.

- `qualification`: maps the job's SQL physical nodes to the library's operators and reports how much of the compute time sits in them.
- `profiling`: ranks vertices by busy time, flags back pressure, subtask skew, checkpoint problems and exceptions.

Derived in part from [Mega-Flink](https://github.com/cb521/Mega-Flink) (see [NOTICE](NOTICE)). The XGBoost speedup prediction and the compare/calibrate/train commands of the original were **not** taken over: this tool never predicts a speedup.

## How to use (5 lines)

```bash
cd streaming/tools                       # needs python3 >= 3.8
./fep-flink-tools --version
./fep-flink-tools qualification --rest-url http://JOBMANAGER:8081 --job-id <32-hex-id> --output-dir out/qual   # live job (GET only)
./fep-flink-tools profiling --archive-file <job-archive> --archive-flink-version 2.2.1 --output-dir out/prof   # History Server archive
./fep-flink-tools qualification --snapshot-file fixtures/rest-snapshot.json --explain-file fixtures/explain.txt --output-dir out/demo   # offline
```

`--output-dir` must not exist yet. Optional: `--explain-file` (output of `EXPLAIN`; raw text is never kept), `--sample-count N --sample-interval-seconds S` for live runs (use at least 5 samples over 10 s), `--authorization-env VAR` (name of an environment variable holding the Authorization header), `--pseudonym-key-env VAR`, `--ca-file`, `--capabilities` (a different operator-family file).

## How to read the output

Each run writes `qualification.{md,json}`, `workload.json` (the sanitized input, reusable with `--workload-file`), `SHA256SUMS`, and for `profiling` also `profiling.{md,json}`.

- **Review priority** (`qualification`): share of compute time in library-provided families: at least 60% is `high`, at least 25% is `medium`, below that `low`; `unknown` if at least half of the compute cannot be classified or there is no compute evidence. These thresholds are heuristics inherited from the original tool and have not been validated against measured GPU gains.
- **Library status / Library API** per operator family: `library-provided` (the library has an operator, API named), `not-provided` (stays on CPU), `cpu-boundary` (source, sink, exchange). A match means "worth reviewing", not "will run on GPU or will be faster". The `review_checks` in `qualification.json` list what to confirm by hand (window kind, join type, key types, and so on).
- **Hot vertices / Findings** (`profiling`): busy and back-pressure milliseconds per second, subtask skew, checkpoint health. Back pressure is seen on the upstream producer; the slow vertex is usually downstream of it.
- Sensitive strings (SQL text, job and table names, credentials, exception text) are not kept; vertices are renamed `vertex_001`, ...

Operator-family file: [qualification/flink-2.2.1-operator-families.json](qualification/flink-2.2.1-operator-families.json). Current mapping (the API names are those of the Java library in `../library`: `GpuWindows.tumble`, `GpuWindows.hop` and `GpuJoins.lookupInner`; the file is checked against the library sources by hand, not by a test):

| Library operator | SQL physical node (Flink 2.2.1) | Library API |
|---|---|---|
| Window aggregation (TUMBLE / HOP) | `StreamExecWindowAggregate`, and its two-phase split `StreamExecLocalWindowAggregate` / `StreamExecGlobalWindowAggregate` | `GpuWindows.tumble` / `GpuWindows.hop` |
| Bounded INNER dimension-table lookup | `StreamExecLookupJoin` | `GpuJoins.lookupInner` |

All other families (join, group aggregate, deduplicate, rank/TopN, over, sort, expand, correlate, calc, interval/window/temporal join, and so on) are `not-provided`. The node name does not show the window kind or join type, so CUMULATE/SESSION windows and LEFT/SEMI/ANTI lookups also appear as matches; the `review_checks` flag them.

## Self-test

```bash
python3 tests/self_test.py
```

It checks the inventory contract, runs both commands on the bundled synthetic fixtures, checks that fixture secrets do not reach any output, re-uses a `workload.json`, runs a live REST collection against a local fixture server and verifies it sends only GET requests, and checks that `compare`/`calibrate`/`train` are gone. Result on Flink 2.2.1 fixtures, Python 3.10:

```text
$ python3 tests/self_test.py
outputs kept in /tmp/fep-flink-tools-test-xxxxxxxx
fep-flink-tools self-test: OK
```

Qualification on the fixture (a synthetic job: source, lookup join, TUMBLE window aggregate, sink; the numbers in it are made up and say nothing about performance):

```text
$ ./fep-flink-tools qualification --snapshot-file fixtures/rest-snapshot.json --explain-file fixtures/explain.txt --output-dir out/qual
FEP_FLINK_QUALIFICATION_OK priority=high output_dir=out/qual
```

```text
- Review priority: `high`
- Next action: `rewrite_candidate_segments_with_gpu_library_and_compare_with_cpu`
- Evidence basis: `runtime-busy-time`   Evidence quality: `moderate`
- Candidate compute share: `100.0%`

| Family | Evidence share | Library status | Library API | Review priority |
| --- | ---: | --- | --- | --- |
| `LookupJoin` | 43.1% | library-provided | `GpuJoins.lookupInner` | high |
| `WindowAggregate` | 38.6% | library-provided | `GpuWindows.tumble / GpuWindows.hop` | high |
| `Source` | 10.6% | cpu-boundary | `-` | low |
| `Sink` | 7.7% | cpu-boundary | `-` | low |
```

Profiling on the same fixture:

```text
$ ./fep-flink-tools profiling --snapshot-file fixtures/rest-snapshot.json --explain-file fixtures/explain.txt --output-dir out/prof
FEP_FLINK_PROFILING_OK status=needs-attention priority=high output_dir=out/prof
```

```text
| Vertex | Families | Parallelism | Busy ms/s | Backpressure ms/s | Input records/s | Output records/s | Subtask skew | State |
| `vertex_002` | `LookupJoin` | 4 | 870.0 | 120.0 | 100,000.0 | 65,000.0 | `insufficient-evidence` | `busy` |
| `vertex_003` | `WindowAggregate` | 4 | 780.0 | 200.0 | 65,000.0 | 1,800.0 | `insufficient-evidence` | `busy` |
| `vertex_004` | `Sink` | 2 | 310.0 | n/a | 1,800.0 | 1,652.9 | `insufficient-evidence` | `mixed` |
| `vertex_001` | `Source` | 4 | 215.0 | 50.0 | 0.0 | 100,000.0 | `insufficient-evidence` | `mixed` |

| Severity | Finding | Vertex |
| `critical` | `checkpoint_health` | `-` |
| `critical` | `job_exceptions_observed` | `-` |
| `critical` | `high_backpressure` | `vertex_003` |
| `warning` | `short_sampling_window` | `-` |
| `warning` | `backpressure_present` | `vertex_002` |
```

## Limitations

- **Operators are recognized by SQL physical node name** (`StreamExec...` in the plan or vertex names). Pure DataStream jobs, where the user names the operators, cannot be classified and show up as `Unknown`. The tool answers "which segment of an existing SQL job is worth rewriting", not "which DataStream code".
- **Read-only**: it uses HTTP GET against the Flink REST API or reads a History Server job archive. It never submits, cancels or changes a job.
- **Verified only on Flink 2.2.1.** Other versions are not verified; REST fields and node names may differ, and `--archive-flink-version` rejects anything else.
- A family match is not proof that the concrete expressions, types, changelog mode or window semantics are supported by the library.
- REST samples are short snapshots; they do not replace a retained metrics time series. Method-level CPU, allocation and GPU-kernel costs are not collected.
- It does not predict a GPU speedup. Measure with a controlled CPU/GPU comparison of the same job.
- Tested only against the bundled synthetic fixtures; not yet run against a real production job or a real History Server archive in this repository.

## Files

| Path | Purpose |
|---|---|
| `fep-flink-tools` | shell launcher |
| `fep_flink_tools.py` | command line, input validation, output writing |
| `fep_flink_collector.py` | REST/archive/snapshot collection, sanitizing, analysis, reports |
| `qualification/flink-2.2.1-operator-families.json` | operator families, SQL node names, library API mapping |
| `fixtures/` | synthetic REST snapshot and EXPLAIN text |
| `tests/self_test.py`, `tests/rest_fixture_server.py` | self-test and local read-only REST server |
