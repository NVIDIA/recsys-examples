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

"""Train a compact HSTU-style recommender.

Usage:
    python3 examples/hstu/train.py --steps 20
    python3 examples/hstu/train.py --dataset ml-1m --download --steps 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from data import get_dataloader, get_movielens_dataloader
from model import HSTUSequentialRecommender


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic", "ml-1m"], default="synthetic")
    p.add_argument(
        "--data-dir", type=str, default=str(Path(__file__).resolve().parent / "data")
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download the selected real dataset if it is missing.",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--attention-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--min-rating", type=float, default=4.0)
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--max-users", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dataset == "ml-1m":
        if args.download:
            from prepare_movielens import ensure_movielens_1m

            ensure_movielens_1m(args.data_dir)
        loader, vocab_size = get_movielens_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            min_rating=args.min_rating,
            max_samples=args.max_samples,
            max_users=args.max_users,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        print(
            f"Dataset: MovieLens 1M ({len(loader.dataset)} windows, vocab_size={vocab_size})"
        )
    else:
        vocab_size = args.vocab_size
        loader = get_dataloader(
            batch_size=args.batch_size,
            num_samples=max(args.batch_size * (args.steps + 2), 1024),
            seq_len=args.seq_len,
            vocab_size=vocab_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        print(
            f"Dataset: synthetic ({len(loader.dataset)} windows, vocab_size={vocab_size})"
        )

    model = HSTUSequentialRecommender(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: HSTUSequentialRecommender ({params:.2f}M params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    step = 0
    for input_ids, targets, user_group in loader:
        if step >= args.steps:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        user_group = user_group.to(device, non_blocking=True)

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, user_group)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0,
        )
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000

        step += 1
        print(f"step {step:>4d} | loss: {loss.item():.4f} | {dt_ms:.1f} ms")

    print("Done.")


if __name__ == "__main__":
    main()
