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

"""Train a compact OneRec-style SID generator with value heads.

Usage:
    python3 examples/onerec/train.py --steps 20
    python3 examples/onerec/train.py --dataset kuairand-pure --download --steps 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from data import get_dataloader, get_kuairand_dataloader
from model import OneRecSIDWithValue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset", choices=["synthetic", "kuairand-pure"], default="synthetic"
    )
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
    p.add_argument("--max-hist-len", type=int, default=64)
    p.add_argument("--sid-depth", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=64)
    p.add_argument("--user-feat-dim", type=int, default=24)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--value-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--reward-weight", type=float, default=0.2)
    p.add_argument("--ltv-weight", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=500_000)
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument("--include-random", action="store_true")
    p.add_argument("--min-hist-len", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.85)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dataset == "kuairand-pure":
        if args.download:
            from prepare_kuairand import ensure_kuairand_pure

            ensure_kuairand_pure(args.data_dir)
        loader = get_kuairand_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            max_hist_len=args.max_hist_len,
            sid_depth=args.sid_depth,
            num_classes=args.num_classes,
            user_feat_dim=args.user_feat_dim,
            max_rows=args.max_rows,
            max_samples=args.max_samples,
            max_users=args.max_users,
            include_random=args.include_random,
            min_hist_len=args.min_hist_len,
            gamma=args.gamma,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        print(
            f"Dataset: KuaiRand-Pure ({len(loader.dataset)} windows, "
            f"num_classes={args.num_classes})"
        )
    else:
        loader = get_dataloader(
            batch_size=args.batch_size,
            num_samples=max(args.batch_size * (args.steps + 2), 1024),
            max_hist_len=args.max_hist_len,
            sid_depth=args.sid_depth,
            num_classes=args.num_classes,
            user_feat_dim=args.user_feat_dim,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        print(
            f"Dataset: synthetic ({len(loader.dataset)} windows, num_classes={args.num_classes})"
        )

    model = OneRecSIDWithValue(
        num_classes=args.num_classes,
        sid_depth=args.sid_depth,
        max_hist_len=args.max_hist_len,
        user_feat_dim=args.user_feat_dim,
        dim=args.dim,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        value_layers=args.value_layers,
        dropout=args.dropout,
    ).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: OneRecSIDWithValue ({params:.2f}M params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    step = 0
    for target_sid, user_feat, hist_sid, hist_len, immediate_reward, ltv in loader:
        if step >= args.steps:
            break
        target_sid = target_sid.to(device, non_blocking=True)
        user_feat = user_feat.to(device, non_blocking=True)
        hist_sid = hist_sid.to(device, non_blocking=True)
        hist_len = hist_len.to(device, non_blocking=True)
        immediate_reward = immediate_reward.to(device, non_blocking=True)
        ltv = ltv.to(device, non_blocking=True)

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, logs = model.compute_loss(
            target_sid=target_sid,
            user_feat=user_feat,
            hist_sid=hist_sid,
            hist_len=hist_len,
            immediate_reward=immediate_reward,
            ltv=ltv,
            reward_weight=args.reward_weight,
            ltv_weight=args.ltv_weight,
        )
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000

        step += 1
        print(
            f"step {step:>4d} | loss: {logs['loss']:.4f} | "
            f"cls: {logs['cls']:.4f} | reward: {logs['reward']:.4f} | "
            f"ltv: {logs['ltv']:.4f} | {dt_ms:.1f} ms"
        )

    print("Done.")


if __name__ == "__main__":
    main()
