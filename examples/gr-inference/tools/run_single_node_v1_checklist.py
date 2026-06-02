"""Run the single-node v1 validation/profile checklist."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from tool_utils import bootstrap_repo_paths

REPO_ROOT = bootstrap_repo_paths(__file__)

from gr_inference.gr_models.qwen3 import (  # noqa: E402
    DEFAULT_QWEN3_VARIANT,
    KNOWN_QWEN3_VARIANTS,
    resolve_qwen3_model_dir,
)

DEFAULT_BEAM_SCHEDULE = "0:256,1:192,2:192"
DEFAULT_DYNAMIC_BEAM_QUALITY_SCHEDULES = (
    "0:256,1:192,2:192",
    "0:256,1:192,2:128",
    "0:256,1:128,2:64",
)


FOCUSED_TESTS = (
    "tests/test_continuous_scheduler.py",
    "tests/test_estimate_gr_memory_tool.py",
    "tests/test_logits_processor.py",
    "tests/test_beam_search.py",
    "tests/test_batched_beam_search.py",
    "tests/test_serving_engine.py",
)


def build_plan(args) -> list[dict[str, object]]:
    args.model_dir = resolve_qwen3_model_dir(
        getattr(args, "model_dir", None),
        variant=getattr(args, "model_variant", None),
    )
    output_dir = Path(args.output_dir)
    profile_json = output_dir / "qwen3_single_node_v1_profile.json"
    profile_summary_json = output_dir / "qwen3_single_node_v1_profile_summary.json"
    memory_json = output_dir / "qwen3_single_node_v1_memory.json"
    dynamic_beam_quality_json = output_dir / "qwen3_dynamic_beam_quality.json"
    dynamic_beam_quality_md = output_dir / "qwen3_dynamic_beam_quality.md"
    soak_json = output_dir / "qwen3_single_node_v1_soak.json"
    soak_report_json = output_dir / "qwen3_single_node_v1_soak_report.json"
    steps: list[dict[str, object]] = []

    if not args.skip_tests:
        steps.append(
            {
                "name": "focused_pytest",
                "command": [
                    sys.executable,
                    "-m",
                    "pytest",
                    *FOCUSED_TESTS,
                    "-ra",
                ],
            }
        )
        steps.append(
            {
                "name": "full_pytest",
                "command": [sys.executable, "-m", "pytest", "-q"],
            }
        )
        if not args.skip_diff_check and _is_git_worktree(REPO_ROOT):
            steps.append({"name": "diff_check", "command": ["git", "diff", "--check"]})

    steps.append(
        {
            "name": "memory_estimate",
            "command": [
                sys.executable,
                "tools/estimate_gr_memory.py",
                "--batch-size",
                str(args.profile_requests),
                "--num-layers",
                str(args.num_layers),
                "--context-len",
                str(args.context_len),
                "--max-decode-steps",
                str(args.decode_steps),
                "--max-beam-width",
                str(args.beam_width),
                "--active-beam-width",
                str(args.active_beam_width or args.beam_width),
                "--num-kv-heads",
                str(args.num_kv_heads),
                "--head-dim",
                str(args.head_dim),
                "--vocab-size",
                str(args.vocab_size),
                "--pretty",
            ],
            "output_json": str(memory_json),
        }
    )

    if not args.skip_profile:
        steps.append(
            {
                "name": "profile_serving",
                "command": [
                    sys.executable,
                    "tools/run_qwen3_real_weight_serving.py",
                    "--model-dir",
                    args.model_dir,
                    "--context-len",
                    str(args.context_len),
                    "--decode-steps",
                    str(args.decode_steps),
                    "--beam-width",
                    str(args.beam_width),
                    "--beam-schedule",
                    args.beam_schedule,
                    "--requests",
                    str(args.profile_requests),
                    "--max-batch-size",
                    str(args.max_batch_size),
                    "--batched-decode",
                    "--continuous",
                    "--decode-backend",
                    args.decode_backend,
                    "--device",
                    args.device,
                    "--beam-kv-pool-capacity",
                    str(args.beam_kv_pool_capacity),
                    "--profile-continuous-decode",
                    "--profile-detail",
                    args.profile_detail,
                    "--warmup-runs",
                    str(args.warmup_runs),
                    "--repeat",
                    str(args.repeat),
                    "--output-json",
                    str(profile_json),
                ],
            }
        )
        steps.append(
            {
                "name": "profile_summary",
                "command": [
                    sys.executable,
                    "tools/summarize_serving_profile.py",
                    str(profile_json),
                    "--output-json",
                    str(profile_summary_json),
                ],
            }
        )

    if args.run_dynamic_beam_quality:
        command = [
            sys.executable,
            "tools/compare_beam_policies.py",
            "--model-dir",
            args.model_dir,
            "--context-len",
            str(args.context_len),
            "--decode-steps",
            str(args.decode_steps),
            "--beam-width",
            str(args.beam_width),
        ]
        schedules = (
            args.dynamic_beam_quality_schedule
            if args.dynamic_beam_quality_schedule
            else DEFAULT_DYNAMIC_BEAM_QUALITY_SCHEDULES
        )
        for schedule in schedules:
            command.extend(["--schedule", schedule])
        command.extend(
            [
                "--score-margins",
                *[str(margin) for margin in args.dynamic_beam_quality_score_margins],
                "--score-margin-min-widths",
                *[str(width) for width in args.dynamic_beam_quality_min_widths],
                "--warmup-cases",
                str(args.dynamic_beam_quality_warmup_cases),
                "--cases",
                str(args.dynamic_beam_quality_cases),
                "--compare-top-k",
                str(args.dynamic_beam_quality_compare_top_k),
                "--decode-backend",
                args.decode_backend,
                "--device",
                args.device,
                "--output-json",
                str(dynamic_beam_quality_json),
                "--output-markdown",
                str(dynamic_beam_quality_md),
                "--fail-on-quality-gate",
            ]
        )
        steps.append({"name": "dynamic_beam_quality", "command": command})

    if args.run_soak:
        steps.append(
            {
                "name": "http_soak",
                "server_command": _server_command(args),
                "command": [
                    sys.executable,
                    "tools/soak_http_serving.py",
                    "--base-url",
                    f"http://{args.host}:{args.port}",
                    "--requests",
                    str(args.soak_requests),
                    "--submit-batch-size",
                    str(args.soak_submit_batch_size),
                    "--input-len",
                    str(args.soak_input_len),
                    "--decode-steps",
                    str(args.soak_decode_steps),
                    "--beam-width",
                    str(args.soak_beam_width),
                    "--max-polls",
                    str(args.soak_max_polls),
                    "--ready-sample-interval",
                    str(args.soak_ready_sample_interval),
                    "--progress-interval",
                    str(args.soak_progress_interval),
                    "--output-json",
                    str(soak_json),
                ],
                "report_command": [
                    sys.executable,
                    "tools/summarize_http_soak.py",
                    str(soak_json),
                    "--expected-requests",
                    str(args.soak_requests),
                    "--expected-cancelled",
                    "0",
                    "--output-json",
                    str(soak_report_json),
                    "--fail-on-error",
                ],
            }
        )

    return steps


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.model_dir = resolve_qwen3_model_dir(
        args.model_dir,
        variant=args.model_variant,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = args.pythonpath
    if args.gr_decode_atten_root:
        env["GR_DECODE_ATTEN_ROOT"] = args.gr_decode_atten_root

    plan = build_plan(args)
    plan_path = output_dir / "single_node_v1_checklist_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return 0

    for step in plan:
        name = str(step["name"])
        if name == "http_soak":
            _run_http_soak_step(step, env=env, args=args, output_dir=output_dir)
            continue
        completed = _run_command(
            step["command"],  # type: ignore[arg-type]
            env=env,
            log_path=output_dir / f"{name}.log",
        )
        if "output_json" in step:
            Path(str(step["output_json"])).write_text(
                completed.stdout, encoding="utf-8"
            )
    print(f"single-node v1 checklist passed; artifacts: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-variant",
        default=DEFAULT_QWEN3_VARIANT,
        choices=[spec.canonical_name for spec in KNOWN_QWEN3_VARIANTS],
    )
    parser.add_argument(
        "--model-dir",
        help="Checkpoint directory. Defaults to the selected Qwen3 variant path.",
    )
    parser.add_argument(
        "--gr-decode-atten-root", default="/cb/gr_inference/gr-decode_atten"
    )
    parser.add_argument("--output-dir", default="profiles/single_node_v1")
    parser.add_argument("--pythonpath", default="src")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--skip-diff-check",
        action="store_true",
        help="Skip git diff --check even when the checkout has git metadata.",
    )
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--run-soak", action="store_true")
    parser.add_argument("--context-len", type=int, default=4700)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--active-beam-width", type=int)
    parser.add_argument("--beam-schedule", default=DEFAULT_BEAM_SCHEDULE)
    parser.add_argument("--profile-requests", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--beam-kv-pool-capacity", type=int, default=2)
    parser.add_argument("--decode-backend", choices=["fake", "real"], default="real")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--profile-detail", choices=["coarse", "fine"], default="fine")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--soak-requests", type=int, default=8192)
    parser.add_argument("--soak-submit-batch-size", type=int, default=16)
    parser.add_argument("--soak-input-len", type=int, default=16)
    parser.add_argument("--soak-decode-steps", type=int, default=1)
    parser.add_argument("--soak-beam-width", type=int, default=128)
    parser.add_argument("--soak-max-polls", type=int, default=2000)
    parser.add_argument("--soak-ready-sample-interval", type=int, default=512)
    parser.add_argument("--soak-progress-interval", type=int, default=512)
    parser.add_argument(
        "--run-dynamic-beam-quality",
        action="store_true",
        help="Run dynamic beam quality/latency comparison and fail if no policy passes.",
    )
    parser.add_argument(
        "--dynamic-beam-quality-schedule",
        action="append",
        default=None,
        help="Schedule to include in dynamic beam quality sweep. Can be repeated.",
    )
    parser.add_argument(
        "--dynamic-beam-quality-score-margins",
        type=float,
        nargs="*",
        default=[1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--dynamic-beam-quality-min-widths",
        type=int,
        nargs="*",
        default=[64, 128],
    )
    parser.add_argument("--dynamic-beam-quality-warmup-cases", type=int, default=1)
    parser.add_argument("--dynamic-beam-quality-cases", type=int, default=32)
    parser.add_argument("--dynamic-beam-quality-compare-top-k", type=int, default=10)
    return parser


def _is_git_worktree(path: Path) -> bool:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _server_command(args) -> list[str]:
    return [
        sys.executable,
        "tools/serve_qwen3_gr_http.py",
        "--model-dir",
        args.model_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--context-len",
        str(args.soak_input_len),
        "--decode-steps",
        str(args.soak_decode_steps),
        "--beam-width",
        str(args.soak_beam_width),
        "--max-batch-size",
        str(args.max_batch_size),
        "--decode-backend",
        args.decode_backend,
        "--device",
        args.device,
        "--beam-kv-pool-capacity",
        str(args.beam_kv_pool_capacity),
        "--max-http-submit-many",
        str(args.soak_submit_batch_size),
    ]


def _run_command(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
        )
    return completed


def _run_http_soak_step(
    step: dict[str, object],
    *,
    env: dict[str, str],
    args,
    output_dir: Path,
) -> None:
    server_log = (output_dir / "http_server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        step["server_command"],  # type: ignore[arg-type]
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_until_ready(f"http://{args.host}:{args.port}/ready")
        _run_command(
            step["command"],  # type: ignore[arg-type]
            env=env,
            log_path=output_dir / "http_soak.log",
        )
        _run_command(
            step["report_command"],  # type: ignore[arg-type]
            env=env,
            log_path=output_dir / "http_soak_report.log",
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        server_log.close()


def _wait_until_ready(url: str, *, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2.0) as response:  # noqa: S310
                if response.status == 200:
                    return
        except URLError as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"server did not become ready at {url}: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())
