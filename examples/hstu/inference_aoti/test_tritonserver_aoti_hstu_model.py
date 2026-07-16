# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_TENSORS = [
    ("INPUT__0", "values"),
    ("INPUT__1", "lengths"),
    ("INPUT__2", "num_candidates"),
    ("INPUT__3", "user_ids"),
    ("INPUT__4", "total_history_lengths"),
]
InputCase = tuple[int, list[np.ndarray]]


def _load_dumped_tensor(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing dumped tensor: {path}")
    module = torch.jit.load(str(path), map_location="cpu")
    tensor = module.tensor.detach().cpu().contiguous()
    return tensor.numpy()


def _make_input(httpclient, name: str, array: np.ndarray):
    if array.dtype != np.int64:
        array = array.astype(np.int64, copy=False)
    infer_input = httpclient.InferInput(name, array.shape, "INT64")
    infer_input.set_data_from_numpy(array)
    return infer_input


def _find_batch_indices(dump_dir: Path) -> list[int]:
    indices = []
    for values_path in dump_dir.glob("batch_*_values.pt"):
        batch_id = values_path.name.removeprefix("batch_").removesuffix("_values.pt")
        indices.append(int(batch_id))
    return sorted(indices)


def _load_input_cases(dump_dir: Path) -> list[InputCase]:
    input_cases = []
    for batch_index in _find_batch_indices(dump_dir):
        prefix = dump_dir / f"batch_{batch_index:06d}"
        input_cases.append(
            (
                batch_index,
                [
                    _load_dumped_tensor(Path(f"{prefix}_{suffix}.pt"))
                    for _, suffix in INPUT_TENSORS
                ],
            )
        )
    if not input_cases:
        raise FileNotFoundError(f"No dumped input cases found in {dump_dir}")
    return input_cases


def _make_inputs(httpclient, input_case: list[np.ndarray]):
    return [
        _make_input(httpclient, input_name, array)
        for (input_name, _), array in zip(INPUT_TENSORS, input_case)
    ]


def _run_input_cases(
    client,
    httpclient,
    model_name: str,
    input_cases: list[InputCase],
    outputs,
):
    result = None
    for batch_index, input_case in input_cases:
        try:
            result = client.infer(
                model_name,
                inputs=_make_inputs(httpclient, input_case),
                outputs=outputs,
            )
        except Exception:
            print(f"Request failed for batch_{batch_index:06d}")
            raise
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay dumped HSTU KV-cache batches to Triton."
    )
    parser.add_argument(
        "--dump_dir",
        type=Path,
        default=SCRIPT_DIR / "export_test_dump",
        help="Directory containing batch_000000_*.pt dump files.",
    )
    parser.add_argument("--url", type=str, default="localhost:8000")
    parser.add_argument("--model_name", type=str, default="hstu_gr_ranking_kvcache")
    parser.add_argument("--post_warmup_sleep_seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import tritonclient.http as httpclient

    input_cases = _load_input_cases(args.dump_dir)
    outputs = [
        httpclient.InferRequestedOutput("OUTPUT__0"),
        httpclient.InferRequestedOutput("OUTPUT__1"),
    ]

    client = httpclient.InferenceServerClient(url=args.url)

    print(f"Loaded {len(input_cases)} input cases from {args.dump_dir}")
    warmup_case = input_cases[:1]
    measured_input_cases = input_cases[1:-1]
    if not measured_input_cases:
        raise RuntimeError("Need at least two input cases to warm up and measure")

    _run_input_cases(
        client,
        httpclient,
        args.model_name,
        warmup_case,
        outputs,
    )
    print("Warmup: sent input case 0, skipping it in measured runs")
    if args.post_warmup_sleep_seconds > 0:
        time.sleep(args.post_warmup_sleep_seconds)
        print(f"Slept {args.post_warmup_sleep_seconds:.3f} seconds after warmup")

    result = None
    for run_index, label in enumerate(("no kvcache", "with kvcache"), start=1):
        start_time = time.perf_counter()
        result = _run_input_cases(
            client,
            httpclient,
            args.model_name,
            measured_input_cases,
            outputs,
        )
        elapsed_seconds = time.perf_counter() - start_time
        print(
            f"Run {run_index} ({label}): {elapsed_seconds:.6f} seconds "
            f"for {len(measured_input_cases)} cases"
        )

    if result is None:
        raise RuntimeError("No Triton requests were sent")

    # logits = result.as_numpy("OUTPUT__0")
    # offload_task_ids = result.as_numpy("OUTPUT__1")
    # print(f"OUTPUT__0 logits: shape={logits.shape}, dtype={logits.dtype}")
    # print(logits)
    # print(
    #     f"OUTPUT__1 kvcache_task_ids: shape={offload_task_ids.shape}, "
    #     f"dtype={offload_task_ids.dtype}"
    # )
    # print(offload_task_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
