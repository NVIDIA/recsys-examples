# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase-1 stats extractor.

Drives a fixed sequence of ``veloq`` queries over an nsys trace and
writes a workspace of JSON artifacts plus a markdown report skeleton.
The agent (Phase 2) reads the workspace and fills in each chapter's
``### Analysis`` block.

Key step beyond plain veloq queries: this script reads
``<trace>.veloq/parquetdir/NVTX_EVENTS.parquet`` produced by ``veloq
summary``, walks the forward NVTX hierarchy to build a ``seq → fwd module
path`` mapping, then stamps every ``XxxBackward, seq=N`` event with its
forward call's ``talos::`` label. The module breakdown then attributes each
CUDA kernel to those ranges by launch time (see ``correlate.py``) —
symmetric fwd/bwd attribution without any model-side instrumentation cost.

Workspace layout::

    .torch_perf_analysis/
        basics.json               run-level basics, four keys:
                                    trace_meta   - span + capabilities
                                    hardware     - CPU / GPU / NIC inventory
                                    steps        - step_* slices + uniformity (CV)
                                    typical_step - which step was picked
        step_<N>/
            stream_<id>/          one folder per CUDA stream:
                timeline_bucket.json    stream in 100 segments: each segment's
                                    busy/idle/util, kernel & H2D/D2H/D2D counts,
                                    and GPU time by source line (file:line)
                util.json           per-segment util array (+ min/mean/max) —
                                    quick scan to spot low-util segments
                source_attribution.json    whole-stream GPU time by source line,
                                    largest first
            cpu_thread_<i>/       one folder per CPU thread:
                torch_function_util.json  torch-function util per segment (frac.
                                    inside talos:: ranges) + role (forward /
                                    backward autograd worker)
            memcpy.json           by_path: nvtx-path + source-line attribution
            sync.json             by_path: surfaces hidden .item()/.cpu()
                                  stalls by source line
            idle.json             idle gaps > 100us
            gpu_util.json         device-level util (union across ALL streams),
                                  bucketed + per-stream means — the aggregate
                                  view the per-stream util.json files can't show
            exposed_idle.json     hotspots ranked by step-time impact — each row
                                  carries both levers (exposed_idle_ms + fusible_ms
                                  + a per-row lever hint), chosen per hotspot;
                                  by_region / by_source / by_sync_source
            fusion_candidates.json source lines ranked by fusible (memory-bound
                                  elementwise/reduction) GPU time — tech4 targets
    (perf_analysis_report.md is not emitted for now — read the JSON directly.)
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

try:  # package module (python -m …python.stats)
    from . import correlate
except ImportError:  # flat script (tests / vendored copy)
    import correlate


# ── Workspace ────────────────────────────────────────────────────────


DEFAULT_WORKSPACE = ".torch_perf_analysis"


def workspace_dir() -> Path:
    """Default workspace. Override per-invocation with ``--workspace``."""
    return Path(DEFAULT_WORKSPACE)


# ── veloq invocation ─────────────────────────────────────────────────


def run_veloq(args: list[str]) -> dict:
    """Run veloq and return the parsed JSON envelope on stdout."""
    result = subprocess.run(["veloq", *args], capture_output=True, text=True)
    if result.returncode != 0:
        # A trace captured with all GPUs visible (no CUDA_VISIBLE_DEVICES)
        # records every device, and veloq then refuses device-scoped queries
        # as ambiguous. Only one device has activity in a single-rank run, so
        # an explicit aggregate is the same answer — retry with it.
        try:
            code = (json.loads(result.stdout).get("error") or {}).get("code", "")
        except (json.JSONDecodeError, AttributeError):
            code = ""
        if code.endswith("multi-device-ambiguous") and "--all-devices" not in args:
            return run_veloq([*args, "--all-devices"])
        sys.stderr.write(
            f"[stats] veloq failed (exit {result.returncode}): veloq {' '.join(args)}\n"
        )
        sys.stderr.write(result.stderr)
        sys.stderr.write(result.stdout[:500] + "\n")
        sys.exit(result.returncode)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        sys.exit(f"[stats] veloq returned non-JSON: {result.stdout[:300]}")


def _rows(env: dict) -> list[dict]:
    # `stats`/`timeline` wrap rows under `data`; `search` puts them at the
    # top level — accept both.
    return (env.get("data") or {}).get("rows") or env.get("rows") or []


# ── Parquet synthesis: turn `seq` into module attribution ───────────


# ── Trace format ────────────────────────────────────────────────────
# talos_observer.cpp emits registered-string NVTX with the autograd seq in
# the 64-bit event payload (parquet column ``int64Value``):
#   * forward outermost:  range  ``talos::<op>...``     payload = own seq
#   * forward nested:     MARK   ``talos_seq:<aten op>``   payload = seq
#   * backward node:      range  ``MulBackward0`` (bare)   payload = seq
TALOS_SEQ_MARK_PREFIX = "talos_seq:"
BWD_BARE_RE = re.compile(r"^\w+Backward\d*$")


def _payload_seq(row: dict) -> int | None:
    """The autograd seq carried in the NVTX payload, if any."""
    v = row.get("int64Value")
    if v is None:
        return None
    v = int(v)
    return v if v >= 0 else None


def _bwd_seq(row: dict, text: str) -> int | None:
    """seq of a backward-node row."""
    if BWD_BARE_RE.match(text):
        return _payload_seq(row)
    return None


# The autograd engine wraps each grad_fn's full evaluation (the XxxBackward
# apply *and* the gradient-accumulate ops that sit outside it) in this range —
# it is the true "backward of one forward op" boundary.
EVAL_FN_PREFIX = "autograd::engine::evaluate_function"
# The observer's label tags *where* a forward call happened, ending in
# ``!!<file>:<line>``:
#   talos::<func>#<path>@<Class>!!<file>:<line>   call inside a module
#   talos::<func>!!<file>:<line>                   free call (no module)
# ``synthesize_bwd_nvtx`` copies that label *verbatim* onto the matching
# ``XxxBackward`` (via the seq join), so a backward kernel attributes to the
# same module and source line as its forward. The label carries no direction
# token — fwd/bwd is read from the surrounding ``autograd::`` range.
TALOS_SEG_RE = re.compile(r"^talos::([^#!]+)(?:#(.+)@([^!@]+))?(?:!!(.+))?$")


def _parse_ta_seg(seg: str) -> tuple[str, str | None, str | None, str | None] | None:
    """Parse a ``talos::`` segment into ``(func, path, class, call_site)``.
    ``path``/``class`` are ``None`` for a free call. ``call_site`` (file:line)
    follows ``!!`` and is carried by forward and synthesized-backward rows
    alike (the synth copies the forward label verbatim). Returns ``None`` for
    non-``talos::`` segments."""
    m = TALOS_SEG_RE.match(seg)
    if m is None:
        return None
    return (m.group(1), m.group(2), m.group(3), m.group(4))


def _find_ta_seg(
    nvtx_path: str,
) -> tuple[str, str | None, str | None, str | None] | None:
    """Innermost (deepest) ``talos::`` segment in ``nvtx_path``."""
    parsed = None
    for seg in nvtx_path.split("/"):
        p = _parse_ta_seg(seg)
        if p is not None:
            parsed = p
    return parsed


def parquetdir_for(trace: Path) -> Path:
    """veloq materializes ``<trace>.veloq/parquetdir/`` on first query."""
    return trace.with_name(trace.name + ".veloq") / "parquetdir"


def invalidate_veloq_caches(trace: Path) -> list[str]:
    """Remove the NVTX-derived caches so veloq rebuilds them on next
    query and sees our synthesized rows. ``correlation.bin`` stays put
    (it's CUPTI-only, not NVTX-derived)."""
    base = trace.with_name(trace.name + ".veloq")
    removed = []
    for name in ("meta.bin", "nvtx-tree.parquet", "nvtx-parent.parquet"):
        p = base / name
        if p.exists():
            p.unlink()
            removed.append(name)
    return removed


def synthesize_bwd_nvtx(nvtx_parquet: Path) -> int:
    """Give each backward op the forward call's attribution. For every
    ``XxxBackward, seq=N`` event we look up the forward call's
    ``talos::<func>#<path>@<Class>!!<file>:<line>`` label (via ``seq``) and inject
    a row carrying it *verbatim* — so the backward kernel attributes to the
    same module and source line as its forward.

    Crucially we place that row over the enclosing **``evaluate_function``**
    range, not the inner ``XxxBackward``: ``evaluate_function`` is the full
    backward of one forward op and also covers the gradient-accumulate ops
    (``aten::add`` …) that sit *outside* ``XxxBackward`` but inside the node's
    evaluation — those would otherwise be unattributed. The row is placed
    strictly inside the ``evaluate_function`` (``[start+1, end-1]``) so analysis
    doesn't mistake it for the ``loss.backward()`` driver (which *encloses* the
    engine). Falls back to the ``XxxBackward`` extent if no enclosing
    ``evaluate_function`` is found. Direction is read from the surrounding
    ``autograd::`` range, not the label. Idempotent.

    NVTX text can live in two places depending on who emitted it:
      - direct ``text`` column (torch.cuda.nvtx and other plain-text ranges)
      - ``textId`` → ``StringIds.parquet`` (registered strings, ours)
    We resolve both, then match + synthesize on the unified string."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(nvtx_parquet)
    rows = table.to_pylist()

    sid_map = _load_string_ids(nvtx_parquet.parent / "StringIds.parquet")
    # Cheap-resolve every row once; "" if neither text nor textId resolves.
    texts: list[str] = [_resolve_text(r, sid_map) for r in rows]

    # evaluate_function spans per thread (sorted) — the full backward-node range
    # each synth row will be stamped over.
    eval_by_tid: dict[int, list[tuple[int, int]]] = {}
    for r, t in zip(rows, texts):
        if (
            t.startswith(EVAL_FN_PREFIX)
            and r.get("globalTid") is not None
            and r.get("start") is not None
            and r.get("end") is not None
        ):
            eval_by_tid.setdefault(int(r["globalTid"]), []).append(
                (int(r["start"]), int(r["end"]))
            )
    for ivs in eval_by_tid.values():
        ivs.sort()

    def enclosing_eval(tid, s, e):
        ivs = eval_by_tid.get(tid)
        if not ivs:
            return None
        starts = [iv[0] for iv in ivs]
        i = bisect.bisect_right(starts, s) - 1  # latest eval starting ≤ s
        while i >= 0:
            es, ee = ivs[i]
            if ee >= e:  # …that also contains the end ⇒ encloses
                return (es, ee)
            i -= 1
        return None

    # Idempotence: a synth row is placed at evaluate_function.start+1 (or, as a
    # fallback, XxxBackward.start-1). A live forward range — incl. checkpoint
    # recompute — never starts at exactly those offsets.
    synth_starts = {(tid, es + 1) for tid, ivs in eval_by_tid.items() for es, _ in ivs}
    synth_starts |= {
        (int(r["globalTid"]), int(r["start"]) - 1)
        for r, t in zip(rows, texts)
        if _bwd_seq(r, t) is not None
        and r.get("globalTid") is not None
        and r.get("start") is not None
    }
    if any(
        t.startswith("talos::")
        and r.get("globalTid") is not None
        and r.get("start") is not None
        and (int(r["globalTid"]), int(r["start"])) in synth_starts
        for r, t in zip(rows, texts)
    ):
        sys.stderr.write("[stats]      synthesis already present; skip\n")
        return 0

    seq_to_label = _build_seq_to_label(rows, texts)
    if not seq_to_label:
        sys.stderr.write(
            "[stats] WARN: no seq-carrying forward events found (neither "
            "`aten::*, seq=N` ranges, `talos_seq:` marks, nor talos:: payload seqs); "
            "no backward rows synthesized.\n"
        )
        return 0

    schema_cols = set(table.column_names)

    new_rows: list[dict] = []
    for i, r in enumerate(rows):
        seq = _bwd_seq(r, texts[i])
        if seq is None:
            continue
        label = seq_to_label.get(seq)
        if label is None or r.get("start") is None or r.get("end") is None:
            continue
        tid, s, e = r.get("globalTid"), int(r["start"]), int(r["end"])
        ev = enclosing_eval(int(tid), s, e) if tid is not None else None
        if ev is not None and ev[1] - 1 > ev[0] + 1:
            new_start, new_end = (
                ev[0] + 1,
                ev[1] - 1,
            )  # full backward node, strictly inside it
        else:
            new_start, new_end = s - 1, e + 1  # fallback: XxxBackward extent (±1 ns)
        new_row = dict(r)
        # Stamp the forward call's talos:: label (verbatim); clear the inherited
        # textId AND payload seq (a synth talos:: row must not feed the direct
        # seq→label mapping on a later pass).
        new_row["text"] = label
        if "textId" in schema_cols:
            new_row["textId"] = None
        if "int64Value" in schema_cols:
            new_row["int64Value"] = None
        new_row["start"], new_row["end"] = new_start, new_end
        new_rows.append(new_row)

    if not new_rows:
        return 0

    new_table = pa.Table.from_pylist(new_rows, schema=table.schema)
    combined = pa.concat_tables([table, new_table])
    tmp = nvtx_parquet.with_suffix(".parquet.tmp")
    pq.write_table(combined, tmp)
    tmp.replace(nvtx_parquet)
    return len(new_rows)


def _load_string_ids(stringids_parquet: Path) -> dict[int, str]:
    """Build the {id → value} map from StringIds.parquet (NVTX rows often
    reference text via ``textId`` rather than the direct ``text`` column)."""
    if not stringids_parquet.exists():
        return {}
    import pyarrow.parquet as pq

    sids = pq.read_table(stringids_parquet).to_pylist()
    return {r["id"]: r.get("value") for r in sids if r.get("id") is not None}


def _resolve_text(row: dict, sid_map: dict[int, str]) -> str:
    """Return the NVTX label of a row, regardless of whether it's stored
    directly in ``text`` or via ``textId``+StringIds."""
    t = row.get("text")
    if t:
        return t
    tid = row.get("textId")
    if tid is None:
        return ""
    return sid_map.get(int(tid)) or ""


def _build_seq_to_label(rows: list[dict], texts: list[str]) -> dict[int, str]:
    """For each forward ``aten::*, seq=N`` event, find its enclosing
    ``talos::<func>#<path>@<Class>!!<file>:<line>`` range (per thread, by
    interval containment) and record ``seq → that label verbatim`` — to stamp
    on the matching ``XxxBackward, seq=N`` so its kernels attribute to the
    same module *and source line*.

    The two sides may live in different NVTX domains. veloq's nvtx-path
    traversal walks intervals on the same thread regardless of domain, so
    we match purely on (thread, time-interval containment) here too."""
    by_tid: dict[int, list[tuple[dict, str]]] = {}
    for r, t in zip(rows, texts):
        tid = r.get("globalTid")
        if tid is None:
            continue
        by_tid.setdefault(int(tid), []).append((r, t))

    seq_to_label: dict[int, str] = {}
    for _tid, group in by_tid.items():
        # Forward talos:: ranges → precompute (interval, label) once; the label
        # is copied verbatim so the backward op keeps the same source line.
        ta_intervals: list[tuple[int, int, str]] = []
        for r, t in group:
            if t.startswith("talos::"):
                ta_intervals.append((int(r["start"]), int(r["end"]), t))
        if not ta_intervals:
            continue
        ta_intervals.sort()
        starts = [iv[0] for iv in ta_intervals]

        # cpp observer: an outermost ``talos::`` range carries its own seq in
        # the payload — map it directly, no containment search needed.
        for r, t in group:
            if t.startswith("talos::"):
                seq = _payload_seq(r)
                if seq is not None:
                    seq_to_label[seq] = t

        for r, t in group:
            # nested ``talos_seq:<op>`` MARK — seq in the payload
            if t.startswith(TALOS_SEQ_MARK_PREFIX):
                seq = _payload_seq(r)
            else:
                continue
            if seq is None or r.get("start") is None:
                continue
            t_start = int(r["start"])
            t_end = int(r["end"]) if r.get("end") is not None else t_start
            idx = bisect.bisect_right(starts, t_start) - 1
            while idx >= 0:
                _fs, fe, slabel = ta_intervals[idx]
                if fe >= t_end:
                    seq_to_label[seq] = slabel
                    break
                idx -= 1
    return seq_to_label


# ── Phase 1.1: trace meta + hardware ────────────────────────────────


def collect_trace_meta(trace: Path) -> dict:
    env = run_veloq(["summary", str(trace)])
    span_ns = env.get("trace_span", {}).get("span_ns", 0)
    return {
        "trace_path": str(trace),
        "source": env.get("source"),
        "trace_span_ns": span_ns,
        "trace_span_s": span_ns / 1e9,
        "capabilities": env.get("data", {}).get("auxiliary", {}).get("capabilities"),
    }


def collect_hardware(trace: Path) -> dict:
    env = run_veloq(["hardware", str(trace)])
    rows = _rows(env)
    return rows[0] if rows else {}


# ── Phase 1.2: steps + typical-step selection ────────────────────────


def collect_steps(trace: Path) -> dict:
    # veloq's slices sort keys drift across versions (`cpu.start:asc` in
    # 0.3+, `start:asc` in 0.2.x); we sort by step number in Python instead.
    # Step labels are emitted as ``ta:step_<N>`` by profiler.py.
    env = run_veloq(["slices", str(trace), "--name", "ta:step_*"])
    steps: list[dict] = []
    for r in _rows(env):
        try:
            name = r["name"]
            n = int(name.split("_", 1)[1])
        except (KeyError, ValueError, IndexError):
            continue
        cpu = r.get("cpu") or {}
        s, e = cpu.get("start_ns"), cpu.get("end_ns")
        if s is None or e is None:
            continue
        steps.append(
            {
                "step": n,
                "name": r["name"],
                "fwd_start_ns": s,
                "fwd_end_ns": e,
                "fwd_duration_ms": (e - s) / 1e6,
                "fwd_attributed_kernel_ms": (r.get("attributed_kernel_ns") or 0) / 1e6,
            }
        )
    steps.sort(key=lambda s: s["step"])

    # Full-step boundary = current fwd_start → next fwd_start
    for i, s in enumerate(steps):
        if i + 1 < len(steps):
            s["full_end_ns"] = steps[i + 1]["fwd_start_ns"]
        else:
            fwd_dur = s["fwd_end_ns"] - s["fwd_start_ns"]
            s["full_end_ns"] = s["fwd_end_ns"] + 2 * fwd_dur
        s["full_duration_ms"] = (s["full_end_ns"] - s["fwd_start_ns"]) / 1e6

    durations = [s["full_duration_ms"] for s in steps]
    avg = statistics.mean(durations) if durations else 0.0
    sd = statistics.stdev(durations) if len(durations) > 1 else 0.0
    cv = (sd / avg) if avg > 0 else 0.0

    summary = {
        "count": len(steps),
        "first_step": steps[0]["step"] if steps else None,
        "last_step": steps[-1]["step"] if steps else None,
        "avg_full_duration_ms": avg,
        "stdev_full_duration_ms": sd,
        "cv_full_duration": cv,
        "uniform": cv < 0.10,
        "steps": steps,
    }
    return summary


def select_typical_step(summary: dict) -> dict | None:
    """Drop step 0 (warmup) and the last step (boundary), median by full
    duration. Falls back to median over all when fewer than 3 steps."""
    steps = summary.get("steps") or []
    if not steps:
        return None
    if len(steps) >= 3:
        candidates = steps[1:-1]
        reason = (
            f"dropped step_{steps[0]['step']} (warmup) and "
            f"step_{steps[-1]['step']} (last); median of {len(candidates)}"
        )
    else:
        candidates = steps
        reason = f"only {len(steps)} steps; median over all"

    ranked = sorted(candidates, key=lambda s: s["full_duration_ms"])
    pick = ranked[len(ranked) // 2]
    out = {
        "step": pick["step"],
        "name": pick["name"],
        "fwd_start_ns": pick["fwd_start_ns"],
        "fwd_end_ns": pick["fwd_end_ns"],
        "full_end_ns": pick["full_end_ns"],
        "fwd_duration_ms": pick["fwd_duration_ms"],
        "full_duration_ms": pick["full_duration_ms"],
        "reason": reason,
    }
    return out


# ── Phase 1.3: per-step evidence ─────────────────────────────────────


def collect_timeline_bucket(
    trace: Path, ws: Path, pick: dict, n_buckets: int = 100
) -> dict:
    """Per-stream GPU timeline for the typical step. Each CUDA stream gets its
    own folder ``step_<N>/stream_<id>/`` holding three files:
      * ``timeline_bucket.json`` — the stream sliced into ``n_buckets``
        segments (busy/idle/util, kernel & copy counts, per-segment source
        lines),
      * ``util.json`` — just the ``n_buckets``-long per-segment util array (+
        min/mean/max), a lightweight scan to spot low-utilisation segments and
        drill into the matching bucket of ``timeline_bucket.json``,
      * ``source_attribution.json`` — the *whole-stream* GPU time by source
        line (``file:line``), largest first.

    Built directly from the parquet tables (see ``correlate.build_timeline``):
    each kernel / memcpy is clipped to the bucket windows it overlaps (so util
    ≤ 100% even when one kernel spans several segments) and its GPU time is attributed to the
    source line of the launching ``torch.*`` call — forward and synthesized
    backward alike.

    CPU side: one folder ``step_<N>/cpu_thread_<i>/`` per **active** thread
    (mirrors ``stream_<id>/``), holding a single ``torch_function_util.json`` —
    that thread's **torch-function utilisation** per segment (the fraction
    inside a ``talos::`` range). ``role`` is ``forward`` (main thread), ``backward``
    (autograd worker), or ``other`` (busy launching raw CUDA but no torch ops —
    util ≈ 0, see ``cuda_api_cover_ms``). High util ⇒ low framework-glue
    overhead. Near-idle helper threads are filtered out (see
    ``correlate._cpu_threads``).

    Returns a summary: per-stream (totals + util array) and per-thread util."""
    s_ns, e_ns = pick["fwd_start_ns"], pick["full_end_ns"]
    pqdir = parquetdir_for(trace)
    tl = correlate.build_timeline(str(pqdir), s_ns, e_ns, n_buckets)
    window = tl["window"]

    step_dir = ws / f"step_{pick['step']}"
    for old in step_dir.glob("stream_*"):  # clear prior streams / threads
        if old.is_dir():
            shutil.rmtree(old)
    for old in step_dir.glob("cpu_thread_*"):
        if old.is_dir():
            shutil.rmtree(old)
    if (step_dir / "cpu").exists():  # remove the old flat layout
        shutil.rmtree(step_dir / "cpu")

    index_streams = []
    for sid, st in tl["streams"].items():
        sdir = step_dir / f"stream_{sid}"
        sdir.mkdir(parents=True, exist_ok=True)
        utils = [b["util"] for b in st["buckets"]]
        _write_json(
            sdir / "timeline_bucket.json",
            {
                "stream": st["stream"],
                "window": window,
                "totals": st["totals"],
                "buckets": st["buckets"],
            },
        )
        _write_json(
            sdir / "util.json",
            {
                "stream": st["stream"],
                "n_buckets": window["n_buckets"],
                "bucket_ns": window["bucket_ns"],
                "start_ns": window["start_ns"],
                "mean_util": round(sum(utils) / len(utils), 4) if utils else 0.0,
                "min_util": round(min(utils), 4) if utils else 0.0,
                "max_util": round(max(utils), 4) if utils else 0.0,
                # keyed by bucket index (matches timeline_bucket.json's bucket `i`)
                "util": {str(i): u for i, u in enumerate(utils)},
            },
        )
        _write_json(
            sdir / "source_attribution.json",
            {
                "stream": st["stream"],
                "busy_ms": st["totals"]["busy_ms"],
                "by_source": st["by_source"],
            },
        )
        index_streams.append(
            {"stream": st["stream"], "totals": st["totals"], "util": utils}
        )
    index_streams.sort(key=lambda s: -s["totals"]["busy_ms"])

    # CPU side: one folder per thread (mirrors stream_<id>/), holding a single
    # torch_function_util.json — that thread's torch-function utilisation per
    # segment (fraction inside a talos:: range). High = low glue overhead (Python
    # on fwd threads / autograd engine on the bwd worker).
    index_threads = []
    for i, th in enumerate(tl["threads"]):
        u = th["util"]
        tdir = step_dir / f"cpu_thread_{i}"
        tdir.mkdir(parents=True, exist_ok=True)
        _write_json(
            tdir / "torch_function_util.json",
            {
                "thread_index": i,
                "global_tid": th["global_tid"],
                "role": th["role"],
                "cover_ms": th["cover_ms"],  # torch-function (talos::) coverage
                "cuda_api_cover_ms": th[
                    "cuda_api_cover_ms"
                ],  # CUDA-API coverage (busyness; ~0 util threads)
                "n_buckets": window["n_buckets"],
                "bucket_ns": window["bucket_ns"],
                "start_ns": window["start_ns"],
                "mean_util": round(sum(u) / len(u), 4) if u else 0.0,
                "min_util": round(min(u), 4) if u else 0.0,
                "max_util": round(max(u), 4) if u else 0.0,
                # torch_function util keyed by bucket index
                "util": {str(b): v for b, v in enumerate(u)},
            },
        )
        index_threads.append(
            {
                "thread_index": i,
                "global_tid": th["global_tid"],
                "role": th["role"],
                "cover_ms": th["cover_ms"],
                "cuda_api_cover_ms": th["cuda_api_cover_ms"],
                "util": u,
            }
        )

    if not tl["streams"]:
        sys.stderr.write(
            "[stats] WARN: no GPU events in the step window — timeline is empty. "
            "Either the parquetdir is missing CUPTI tables, or the capture host's "
            "talos is a different version than this stats.py.\n"
        )
    return {
        "step": pick["step"],
        "window": window,
        "streams": index_streams,
        "threads": index_threads,
    }


# ── memcpy / sync / idle ────────────────────────────────────────────


def _by_nvtx_path(env_rows: list[dict], span_ns: int) -> list[dict]:
    """Turn ``stats --group-by nvtx-path`` rows (for memcpy / sync) into
    source-attributed entries. The ``nvtx_path`` is a ``/``-joined string;
    its innermost ``talos::`` segment gives the ``source`` (``file:line`` of
    the triggering ``torch.*`` call, e.g. a ``.item()`` / ``.cpu()``) and
    ``module`` (``<path>@<Class>`` or null for a free call). When no
    ``talos::`` segment is present the event sits under the user's own NVTX
    ranges (or ``__no_nvtx__``) — the raw ``nvtx_path`` is kept so that's
    still visible.

    NB: this uses ``--group-by nvtx-path``, not ``search``: veloq's
    ``search`` leaves the per-event ``nvtx_context`` null for memcpy/sync,
    so the path grouping is the only thing that carries attribution."""
    out = []
    for r in env_rows:
        path = r.get("nvtx_path") or ""
        ta = _find_ta_seg(path)
        module = source = None
        if ta is not None:
            _f, p, c, cs = ta
            module = f"{p}@{c}" if p else None
            source = cs
        total = r.get("total_ns") or 0
        out.append(
            {
                "nvtx_path": path,
                "module": module,  # <path>@<Class> (null for a free call)
                "source": source,  # file:line of the triggering torch.* call (or null)
                "count": r.get("count"),
                "total_ms": total / 1e6,
                "pct_of_step": (total / span_ns * 100) if span_ns > 0 else 0,
            }
        )
    return out


def collect_memcpy(trace: Path, ws: Path, pick: dict) -> dict:
    """Where the memcpy time goes, by NVTX path, with source attribution."""
    s_ns, e_ns = pick["fwd_start_ns"], pick["full_end_ns"]
    span_ns = e_ns - s_ns
    bypath = run_veloq(
        [
            "stats",
            str(trace),
            "--from",
            f"@{s_ns}",
            "--to",
            f"@{e_ns}",
            "--type",
            "memcpy",
            "--group-by",
            "nvtx-path",
            "--sort",
            "total:desc",
            "--limit",
            "25",
        ]
    )
    data = {"by_path": _by_nvtx_path(_rows(bypath), span_ns)}
    _write_json(ws / f"step_{pick['step']}" / "memcpy.json", data)
    return data


def collect_sync(trace: Path, ws: Path, pick: dict) -> dict:
    """Where the sync time goes, by NVTX path, with source attribution —
    this is what surfaces a hidden ``.item()`` / ``.cpu()`` stall."""
    s_ns, e_ns = pick["fwd_start_ns"], pick["full_end_ns"]
    span_ns = e_ns - s_ns
    bypath = run_veloq(
        [
            "stats",
            str(trace),
            "--from",
            f"@{s_ns}",
            "--to",
            f"@{e_ns}",
            "--type",
            "sync",
            "--group-by",
            "nvtx-path",
            "--sort",
            "total:desc",
            "--limit",
            "25",
        ]
    )
    data = {"by_path": _by_nvtx_path(_rows(bypath), span_ns)}
    _write_json(ws / f"step_{pick['step']}" / "sync.json", data)
    return data


def collect_idle(trace: Path, ws: Path, pick: dict) -> dict:
    s_ns, e_ns = pick["fwd_start_ns"], pick["full_end_ns"]
    env = run_veloq(
        [
            "gaps",
            str(trace),
            "--from",
            f"@{s_ns}",
            "--to",
            f"@{e_ns}",
            "--min-duration",
            "100us",
            "--sort",
            "duration:desc",
            "--limit",
            "30",
        ]
    )
    rows = _rows(env)
    durations = [r.get("duration_ns") or 0 for r in rows]
    data = {
        "gap_count": len(rows),
        "total_idle_ms": sum(durations) / 1e6,
        "longest_us": max(durations) / 1000 if durations else 0,
        "gaps": [
            {
                "key": r.get("key"),
                "start_ns": r.get("start_ns"),
                "duration_us": (r.get("duration_ns") or 0) / 1000,
                "prev_event_row_id": r.get("prev_event_row_id"),
                "next_event_row_id": r.get("next_event_row_id"),
            }
            for r in rows
        ],
    }
    _write_json(ws / f"step_{pick['step']}" / "idle.json", data)
    return data


def collect_gpu_util(trace: Path, ws: Path, pick: dict) -> dict:
    """Device-level GPU utilisation — the union of busy time across ALL
    streams, bucketed. Individual streams can each look moderate while the
    device is nearly saturated (busy windows interleave across streams);
    this is the whole-GPU view the per-stream ``stream_<id>/util.json``
    files can't show. Written at the step level, outside ``stream_*`` /
    ``cpu_thread_*``."""
    s_ns, e_ns = pick["fwd_start_ns"], pick["full_end_ns"]
    pqdir = parquetdir_for(trace)
    data = correlate.build_gpu_util(str(pqdir), s_ns, e_ns)
    data = {"step": pick["step"], **data}
    _write_json(ws / f"step_{pick['step']}" / "gpu_util.json", data)
    return data


def collect_exposed_idle(trace: Path, ws: Path, pick: dict) -> dict:
    """Rank hotspots by step-time impact — the "does fixing this actually make
    the step faster?" view, as opposed to self-time (``sync.json`` /
    ``source_attribution.json``). See ``correlate.build_exposed_idle``: device
    idle is charged to the launch that ends each gap (idle lever), and each row
    also carries ``busy_ms`` / ``fusible_ms`` (busy lever) with a per-row
    ``lever`` hint, so the lever is chosen per hotspot — a model mixes
    idle-bound (starved) and busy-bound (high-util fusible) regions."""
    s_ns, e_ns = pick["fwd_start_ns"], pick["full_end_ns"]
    pqdir = parquetdir_for(trace)
    data = correlate.build_exposed_idle(str(pqdir), s_ns, e_ns)
    data = {"step": pick["step"], **data}
    _write_json(ws / f"step_{pick['step']}" / "exposed_idle.json", data)
    return data


def collect_fusion_candidates(trace: Path, ws: Path, pick: dict) -> dict:
    """Rank source-line regions by fusible (memory-bound elementwise /
    reduction) GPU time — the tech4 (CUDA-fusion) candidate list. Compute-bound
    kernels (gemm / conv / attention) and non-fusible ones (comm / sort) are
    excluded by name classification (see ``correlate.classify_kernel``). CUPTI
    carries no per-kernel byte count, so ``gpu_ms`` is the memory-traffic proxy;
    a region ranks high whether it is launch-bound (many tiny kernels, low util)
    or bandwidth-bound (few large elementwise kernels at high util)."""
    s_ns, e_ns = pick["fwd_start_ns"], pick["full_end_ns"]
    pqdir = parquetdir_for(trace)
    rows = correlate.build_fusion_candidates(str(pqdir), s_ns, e_ns)
    data = {
        "step": pick["step"],
        "metric": "gpu_ms is the memory-traffic proxy (no per-kernel bytes in CUPTI); "
        "rank by fusible_gpu_ms. by_kind shows the kernel mix per source line: "
        "'elementwise' fuses directly (pointwise islands); 'reduction' "
        "(sum/mean/norm/softmax) fuses with its surrounding pointwise, not in "
        "isolation. Compute-bound (gemm/conv/attention) and non-fusible "
        "(comm/sort) kinds are listed but excluded from fusible_gpu_ms.",
        "candidates": rows,
    }
    _write_json(ws / f"step_{pick['step']}" / "fusion_candidates.json", data)
    return data


# ── helpers ─────────────────────────────────────────────────────────


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# ── Main ────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="talos.torch_perf_analysis.python.stats",
        description="Extract baseline performance evidence from an nsys trace.",
    )
    parser.add_argument("trace", help="path to .nsys-rep (or pre-exported _pqtdir/)")
    parser.add_argument(
        "--workspace",
        default=None,
        help=f"workspace dir to write the JSON evidence into (default: {DEFAULT_WORKSPACE})",
    )
    args = parser.parse_args(argv)

    if not shutil.which("veloq"):
        sys.exit("[stats] `veloq` not on PATH; reinstall talos")

    trace_orig = Path(args.trace).resolve()
    if not trace_orig.exists():
        sys.exit(f"[stats] trace not found: {trace_orig}")

    ws = Path(args.workspace).resolve() if args.workspace else workspace_dir().resolve()
    ws.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"[stats] workspace: {ws}\n")

    # Symlink the trace into the workspace so veloq's auto-generated
    # ``<trace>.veloq/`` artifact root (parquetdir + caches) lands inside
    # the workspace alongside our JSON outputs, instead of polluting the
    # directory holding the user's .nsys-rep.
    trace = ws / trace_orig.name
    if trace.exists() and trace.samefile(trace_orig):
        # The trace already lives in the workspace; nothing to link.
        pass
    elif trace.is_symlink():
        # Includes broken links, which resolve() still reports a path for.
        if trace.resolve() != trace_orig:
            trace.unlink()
        trace.symlink_to(trace_orig)
    elif trace.exists():
        sys.exit(
            f"[stats] {trace} already exists and is a different file than "
            f"{trace_orig}.\n[stats] Refusing to analyze it — remove it or "
            f"pass a different --workspace."
        )
    else:
        trace.symlink_to(trace_orig)

    sys.stderr.write("[stats] 1/4 trace meta + hardware\n")
    trace_meta = collect_trace_meta(trace)
    hardware = collect_hardware(trace)

    nvtx_pq = parquetdir_for(trace) / "NVTX_EVENTS.parquet"
    if not nvtx_pq.exists():
        sys.stderr.write(
            f"[stats] WARN: {nvtx_pq} not found; skipping bwd synthesis "
            "(timeline source attribution will be forward-only)\n"
        )
    else:
        sys.stderr.write("[stats] 2/4 synthesizing bwd: NVTX rows\n")
        n_new = synthesize_bwd_nvtx(nvtx_pq)
        sys.stderr.write(f"[stats]      wrote {n_new} synthesized bwd: rows\n")
        if n_new:
            removed = invalidate_veloq_caches(trace)
            if removed:
                sys.stderr.write(
                    f"[stats]      invalidated veloq caches: {', '.join(removed)}\n"
                )

    sys.stderr.write("[stats] 3/4 step structure + typical step\n")
    steps_summary = collect_steps(trace)
    pick = select_typical_step(steps_summary)

    # One consolidated file for the run-level basics (trace span +
    # capabilities, hardware inventory, step uniformity, the picked step).
    _write_json(
        ws / "basics.json",
        {
            "trace_meta": trace_meta,
            "hardware": hardware,
            "steps": steps_summary,
            "typical_step": pick,
        },
    )

    if pick is not None:
        n = pick["step"]
        sys.stderr.write(f"[stats] 4/4 step_{n} evidence\n")
        collect_timeline_bucket(trace, ws, pick)
        collect_memcpy(trace, ws, pick)
        collect_sync(trace, ws, pick)
        collect_idle(trace, ws, pick)
        collect_gpu_util(trace, ws, pick)
        collect_exposed_idle(trace, ws, pick)
        collect_fusion_candidates(trace, ws, pick)
    else:
        sys.stderr.write(
            "[stats]      no step_* ranges found; skipping per-step phases\n"
        )

    sys.stderr.write(f"[stats] wrote JSON evidence to {ws}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
