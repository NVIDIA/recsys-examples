# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep real-weight serving smoke parameters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tool_utils import bootstrap_repo_paths

REPO_ROOT = bootstrap_repo_paths(__file__)
SERVING_TOOL = REPO_ROOT / "tools" / "run_qwen3_real_weight_serving.py"


def parse_csv_ints(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def parse_csv_strings(value: str) -> list[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one value")
    return parsed


def scheduled_widths(
    *,
    beam_width: int,
    decode_steps: int,
    fractions: str,
) -> dict[int, int]:
    parsed_fractions = [
        float(part.strip()) for part in fractions.split(",") if part.strip()
    ]
    if not parsed_fractions:
        raise ValueError("expected at least one scheduled beam fraction")
    if parsed_fractions[0] != 1.0:
        raise ValueError("scheduled beam fractions must start with 1.0")
    schedule: dict[int, int] = {}
    for step in range(decode_steps):
        fraction = parsed_fractions[min(step, len(parsed_fractions) - 1)]
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("scheduled beam fractions must be in (0, 1]")
        schedule[step] = max(1, int(round(beam_width * fraction)))
    return schedule


def format_beam_schedule(schedule: dict[int, int]) -> str:
    return ",".join(f"{step}:{schedule[step]}" for step in sorted(schedule))


def build_command(
    args,
    *,
    context_len: int,
    decode_steps: int,
    beam_width: int,
    requests: int,
    beam_policy: str,
) -> list[str]:
    command = [
        sys.executable,
        str(SERVING_TOOL),
        "--model-dir",
        args.model_dir,
        "--context-len",
        str(context_len),
        "--decode-steps",
        str(decode_steps),
        "--beam-width",
        str(beam_width),
        "--requests",
        str(requests),
        "--max-batch-size",
        str(requests),
        "--batched-decode",
        "--decode-backend",
        args.decode_backend,
        "--device",
        args.device,
        "--beam-score-mode",
        args.beam_score_mode,
        "--warmup-runs",
        str(args.warmup_runs),
        "--repeat",
        str(args.repeat),
    ]
    if beam_policy == "scheduled":
        command.extend(
            [
                "--beam-schedule",
                format_beam_schedule(
                    scheduled_widths(
                        beam_width=beam_width,
                        decode_steps=decode_steps,
                        fractions=args.scheduled_fractions,
                    )
                ),
            ]
        )
    elif beam_policy != "fixed":
        raise ValueError(f"unsupported beam policy: {beam_policy}")
    if args.continuous:
        command.append("--continuous")
    arrival_stagger_ticks = getattr(args, "arrival_stagger_ticks", 0)
    if arrival_stagger_ticks:
        command.extend(["--arrival-stagger-ticks", str(arrival_stagger_ticks)])
        command.extend(
            ["--arrival-burst-size", str(getattr(args, "arrival_burst_size", 1))]
        )
    if args.profile_continuous_decode:
        command.append("--profile-continuous-decode")
        command.extend(["--profile-detail", args.profile_detail])
    if args.beam_kv_pool_capacity:
        command.extend(["--beam-kv-pool-capacity", str(args.beam_kv_pool_capacity)])
    return command


def parse_summary(output: str) -> dict[str, float | str]:
    summary: dict[str, float | str] = {}
    for line in output.splitlines():
        if ": " not in line:
            continue
        key, raw_value = line.split(": ", 1)
        if key.endswith("_median"):
            summary[key] = float(raw_value)
    return summary


def run_sweep(args) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    env = os.environ.copy()
    if args.gr_decode_atten_root:
        env["GR_DECODE_ATTEN_ROOT"] = args.gr_decode_atten_root
    for context_len in parse_csv_ints(args.context_lens):
        for decode_steps in parse_csv_ints(args.decode_steps_list):
            for beam_width in parse_csv_ints(args.beam_widths):
                for requests in parse_csv_ints(args.batch_sizes):
                    for beam_policy in parse_csv_strings(args.beam_policies):
                        command = build_command(
                            args,
                            context_len=context_len,
                            decode_steps=decode_steps,
                            beam_width=beam_width,
                            requests=requests,
                            beam_policy=beam_policy,
                        )
                        print(
                            "Running "
                            f"context={context_len} decode_steps={decode_steps} "
                            f"beam={beam_width} batch={requests} policy={beam_policy}",
                            flush=True,
                        )
                        try:
                            completed = subprocess.run(
                                command,
                                env=env,
                                check=True,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                            )
                        except subprocess.CalledProcessError as exc:
                            print(exc.stdout, end="", flush=True)
                            raise
                        summary = parse_summary(completed.stdout)
                        results.append(
                            {
                                "context_len": context_len,
                                "decode_steps": decode_steps,
                                "beam_width": beam_width,
                                "batch_size": requests,
                                "beam_policy": beam_policy,
                                "arrival_stagger_ticks": getattr(
                                    args,
                                    "arrival_stagger_ticks",
                                    0,
                                ),
                                "arrival_burst_size": getattr(
                                    args,
                                    "arrival_burst_size",
                                    1,
                                ),
                                **summary,
                            }
                        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--context-lens", default="16")
    parser.add_argument("--decode-steps-list", default="1")
    parser.add_argument("--beam-widths", default="64,128,256")
    parser.add_argument("--batch-sizes", default="2")
    parser.add_argument("--beam-policies", default="fixed")
    parser.add_argument(
        "--scheduled-fractions",
        default="1.0,0.5,0.25",
        help="Fractions of beam width for scheduled policy, one per decode step",
    )
    parser.add_argument("--decode-backend", choices=["fake", "real"], default="real")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--beam-score-mode", choices=["raw_logits", "logprob"], default="logprob"
    )
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument(
        "--arrival-stagger-ticks",
        type=int,
        default=0,
        help="For continuous serving, submit later requests every N scheduler ticks",
    )
    parser.add_argument(
        "--arrival-burst-size",
        type=int,
        default=1,
        help="With staggered arrivals, submit this many requests per arrival tick",
    )
    parser.add_argument("--profile-continuous-decode", action="store_true")
    parser.add_argument(
        "--profile-detail", choices=["coarse", "fine"], default="coarse"
    )
    parser.add_argument("--beam-kv-pool-capacity", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gr-decode-atten-root", default=None)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    print("Real-weight serving sweep")
    print("=" * 72)
    results = run_sweep(args)
    for result in results:
        print(result)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(results, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
