# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one dumped HSTU KV-cache batch to Triton."
    )
    parser.add_argument(
        "--dump_dir",
        type=Path,
        default=SCRIPT_DIR / "export_test_dump",
        help="Directory containing batch_000000_*.pt dump files.",
    )
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--url", type=str, default="localhost:8000")
    parser.add_argument("--model_name", type=str, default="hstu_gr_ranking_kvcache")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import tritonclient.http as httpclient

    prefix = args.dump_dir / f"batch_{args.batch_index:06d}"
    values = _load_dumped_tensor(Path(f"{prefix}_values.pt"))
    lengths = _load_dumped_tensor(Path(f"{prefix}_lengths.pt"))
    num_candidates = _load_dumped_tensor(Path(f"{prefix}_num_candidates.pt"))
    user_ids = _load_dumped_tensor(Path(f"{prefix}_user_ids.pt"))
    total_history_lengths = _load_dumped_tensor(
        Path(f"{prefix}_total_history_lengths.pt")
    )

    inputs = [
        _make_input(httpclient, "INPUT__0", values),
        _make_input(httpclient, "INPUT__1", lengths),
        _make_input(httpclient, "INPUT__2", num_candidates),
        _make_input(httpclient, "INPUT__3", user_ids),
        _make_input(httpclient, "INPUT__4", total_history_lengths),
    ]
    outputs = [
        httpclient.InferRequestedOutput("OUTPUT__0"),
        httpclient.InferRequestedOutput("OUTPUT__1"),
    ]

    client = httpclient.InferenceServerClient(url=args.url)
    result = client.infer(args.model_name, inputs=inputs, outputs=outputs)

    logits = result.as_numpy("OUTPUT__0")
    offload_task_ids = result.as_numpy("OUTPUT__1")
    print(f"OUTPUT__0 logits: shape={logits.shape}, dtype={logits.dtype}")
    print(logits)
    print(
        f"OUTPUT__1 kvcache_task_ids: shape={offload_task_ids.shape}, "
        f"dtype={offload_task_ids.dtype}"
    )
    print(offload_task_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())