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

"""Sequential recommendation data for the HSTU example."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticSequenceDataset(Dataset):
    """Small Markov-style item sequences with predictable next-item targets."""

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        num_user_groups: int = 32,
        seed: int = 0,
    ) -> None:
        if vocab_size < num_user_groups + 8:
            raise ValueError(
                "vocab_size must be comfortably larger than num_user_groups"
            )

        g = torch.Generator().manual_seed(seed)
        user_group = torch.randint(1, num_user_groups + 1, (num_samples,), generator=g)
        first_item = torch.randint(1, vocab_size, (num_samples,), generator=g)
        noise = torch.randint(0, 5, (num_samples, seq_len + 1), generator=g)
        drift = torch.arange(seq_len + 1).unsqueeze(0)

        seq = (
            first_item.unsqueeze(1) + 7 * user_group.unsqueeze(1) + 3 * drift + noise
        ) % (vocab_size - 1)
        seq = seq + 1

        self.input_ids = seq[:, :-1].long()
        self.targets = seq[:, 1:].long()
        self.user_group = user_group.long()

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.targets[idx], self.user_group[idx]


def get_dataloader(
    batch_size: int,
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SyntheticSequenceDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class MovieLensSequenceDataset(Dataset):
    """MovieLens 1M next-item windows sorted by user timestamp."""

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int,
        min_rating: float = 4.0,
        max_samples: int | None = 100_000,
        max_users: int | None = None,
        num_user_groups: int = 32,
        seed: int = 0,
    ) -> None:
        self.seq_len = seq_len
        self.num_user_groups = num_user_groups
        ratings_path = _find_movielens_ratings(data_dir)

        by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
        item_ids: set[int] = set()
        with ratings_path.open("r", encoding="latin-1") as f:
            for line in f:
                user_raw, item_raw, rating_raw, ts_raw = line.rstrip("\n").split("::")
                rating = float(rating_raw)
                if rating < min_rating:
                    continue
                user_id = int(user_raw)
                item_id = int(item_raw)
                timestamp = int(ts_raw)
                by_user[user_id].append((timestamp, item_id))
                item_ids.add(item_id)

        if max_users is not None:
            keep = set(sorted(by_user)[:max_users])
            by_user = {
                user_id: events
                for user_id, events in by_user.items()
                if user_id in keep
            }
            item_ids = {item_id for events in by_user.values() for _, item_id in events}

        self.item_to_dense = {
            item_id: idx + 1 for idx, item_id in enumerate(sorted(item_ids))
        }
        self.vocab_size = len(self.item_to_dense) + 1
        self.user_sequences: list[torch.Tensor] = []
        self.user_groups: list[int] = []
        self.index: list[tuple[int, int]] = []

        for user_idx, user_id in enumerate(sorted(by_user)):
            events = sorted(by_user[user_id])
            seq = [self.item_to_dense[item_id] for _, item_id in events]
            if len(seq) < 2:
                continue
            seq_idx = len(self.user_sequences)
            self.user_sequences.append(torch.tensor(seq, dtype=torch.long))
            self.user_groups.append(user_idx % num_user_groups + 1)
            for end_pos in range(1, len(seq)):
                self.index.append((seq_idx, end_pos))

        if not self.index:
            raise ValueError(f"No MovieLens training windows found in {ratings_path}")

        if max_samples is not None and len(self.index) > max_samples:
            g = torch.Generator().manual_seed(seed)
            chosen = torch.randperm(len(self.index), generator=g)[:max_samples].tolist()
            self.index = [self.index[i] for i in chosen]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_idx, end_pos = self.index[idx]
        seq = self.user_sequences[seq_idx]
        start = max(0, end_pos - self.seq_len)
        window = seq[start : end_pos + 1]
        input_window = window[:-1]
        target_window = window[1:]

        input_ids = torch.zeros(self.seq_len, dtype=torch.long)
        targets = torch.zeros(self.seq_len, dtype=torch.long)
        offset = self.seq_len - input_window.numel()
        input_ids[offset:] = input_window
        targets[offset:] = target_window
        user_group = torch.tensor(self.user_groups[seq_idx], dtype=torch.long)
        return input_ids, targets, user_group


def _find_movielens_ratings(data_dir: str | Path) -> Path:
    root = Path(data_dir).expanduser()
    candidates = [
        root / "ml-1m" / "ratings.dat",
        root / "ratings.dat",
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "MovieLens 1M ratings.dat was not found. Run "
        "`python3 examples/hstu/prepare_movielens.py --data-dir examples/hstu/data` first.\n"
        f"Tried:\n{tried}"
    )


def get_movielens_dataloader(
    data_dir: str | Path,
    batch_size: int,
    seq_len: int,
    min_rating: float = 4.0,
    max_samples: int | None = 100_000,
    max_users: int | None = None,
    seed: int = 0,
    num_workers: int = 0,
) -> tuple[DataLoader, int]:
    dataset = MovieLensSequenceDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        min_rating=min_rating,
        max_samples=max_samples,
        max_users=max_users,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.vocab_size
