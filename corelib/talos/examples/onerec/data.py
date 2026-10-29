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

"""Session recommendation data for the OneRec example."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticOneRecDataset(Dataset):
    """Hierarchical semantic-ID sessions with reward and LTV labels."""

    def __init__(
        self,
        num_samples: int,
        max_hist_len: int,
        sid_depth: int,
        num_classes: int,
        user_feat_dim: int,
        min_hist_len: int = 3,
        num_intents: int = 16,
        seed: int = 0,
    ) -> None:
        if num_classes < 8:
            raise ValueError(
                "num_classes must be at least 8; token 0 is reserved for padding"
            )
        if max_hist_len < min_hist_len:
            raise ValueError("max_hist_len must be >= min_hist_len")

        g = torch.Generator().manual_seed(seed)
        intent = torch.randint(0, num_intents, (num_samples,), generator=g)
        hist_len = torch.randint(
            min_hist_len, max_hist_len + 1, (num_samples,), generator=g
        )

        depth_offsets = torch.arange(sid_depth).view(1, 1, sid_depth)
        time_offsets = torch.arange(max_hist_len).view(1, max_hist_len, 1)
        base = (intent.view(num_samples, 1, 1) * 5 + depth_offsets * 11) % (
            num_classes - 1
        )
        jitter = torch.randint(
            0, 4, (num_samples, max_hist_len, sid_depth), generator=g
        )
        hist_sid = (base + 2 * time_offsets + jitter) % (num_classes - 1)
        hist_sid = hist_sid + 1

        valid = torch.arange(max_hist_len).view(1, max_hist_len) < hist_len.view(
            num_samples, 1
        )
        hist_sid = hist_sid.masked_fill(~valid.unsqueeze(-1), 0).long()

        target_jitter = torch.randint(0, 4, (num_samples, sid_depth), generator=g)
        target_sid = (
            intent.view(num_samples, 1) * 5
            + torch.arange(sid_depth).view(1, sid_depth) * 11
            + 2 * hist_len.view(num_samples, 1)
            + target_jitter
        ) % (num_classes - 1)
        target_sid = (target_sid + 1).long()

        user_feat = torch.randn(num_samples, user_feat_dim, generator=g) * 0.05
        hot_dims = min(user_feat_dim, num_intents)
        user_feat[:, :hot_dims] = 0.0
        user_feat[torch.arange(num_samples), intent % hot_dims] = 1.0
        if user_feat_dim > hot_dims:
            user_feat[:, hot_dims:] += (
                intent.float().unsqueeze(1) / max(num_intents - 1, 1)
            ) * 0.5

        depth_scale = torch.linspace(0.5, 1.0, sid_depth).view(1, sid_depth)
        immediate_reward = ((target_sid.float() % 7.0) / 6.0) * depth_scale
        immediate_reward = (
            immediate_reward + (intent.float().view(num_samples, 1) % 3.0) * 0.05
        )
        discounts = torch.pow(
            torch.full((sid_depth,), 0.85), torch.arange(sid_depth).float()
        )
        ltv = (immediate_reward * discounts.view(1, sid_depth)).sum(dim=1)

        self.hist_sid = hist_sid
        self.hist_len = hist_len.long()
        self.target_sid = target_sid
        self.user_feat = user_feat.float()
        self.immediate_reward = immediate_reward.float()
        self.ltv = ltv.float()

    def __len__(self) -> int:
        return self.target_sid.size(0)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            self.target_sid[idx],
            self.user_feat[idx],
            self.hist_sid[idx],
            self.hist_len[idx],
            self.immediate_reward[idx],
            self.ltv[idx],
        )


def get_dataloader(
    batch_size: int,
    num_samples: int,
    max_hist_len: int,
    sid_depth: int,
    num_classes: int,
    user_feat_dim: int,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SyntheticOneRecDataset(
        num_samples=num_samples,
        max_hist_len=max_hist_len,
        sid_depth=sid_depth,
        num_classes=num_classes,
        user_feat_dim=user_feat_dim,
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


KUAI_LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
KUAI_RANDOM_LOG_FILE = "log_random_4_22_to_5_08_pure.csv"


class KuaiRandSequenceDataset(Dataset):
    """KuaiRand-Pure sessions converted to OneRec semantic-ID targets."""

    def __init__(
        self,
        data_dir: str | Path,
        max_hist_len: int,
        sid_depth: int,
        num_classes: int,
        user_feat_dim: int,
        max_rows: int | None = 500_000,
        max_samples: int | None = 100_000,
        max_users: int | None = None,
        include_random: bool = False,
        min_hist_len: int = 1,
        gamma: float = 0.85,
        seed: int = 0,
    ) -> None:
        if num_classes < 2:
            raise ValueError(
                "num_classes must reserve 0 for padding and at least one SID token."
            )
        self.max_hist_len = max_hist_len
        self.sid_depth = sid_depth
        self.num_classes = num_classes
        self.user_feat_dim = user_feat_dim

        data_path = _find_kuairand_data_dir(data_dir)
        log_files = [data_path / name for name in KUAI_LOG_FILES]
        if include_random:
            log_files.append(data_path / KUAI_RANDOM_LOG_FILE)
        log_files = [path for path in log_files if path.exists()]
        if not log_files:
            raise FileNotFoundError(
                f"No KuaiRand-Pure log files were found under {data_path}"
            )

        user_rows = _load_kuairand_user_features(data_path, user_feat_dim)
        by_user: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
        item_ids: set[int] = set()
        rows_seen = 0
        for path in log_files:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    if max_rows is not None and rows_seen >= max_rows:
                        break
                    try:
                        user_id = int(row["user_id"])
                        video_id = int(row["video_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    timestamp = _row_timestamp(row, row_idx)
                    reward = _row_reward(row)
                    by_user[user_id].append((timestamp, video_id, reward))
                    item_ids.add(video_id)
                    rows_seen += 1
            if max_rows is not None and rows_seen >= max_rows:
                break

        if max_users is not None:
            keep = set(sorted(by_user)[:max_users])
            by_user = {
                user_id: events
                for user_id, events in by_user.items()
                if user_id in keep
            }
            item_ids = {
                video_id for events in by_user.values() for _, video_id, _ in events
            }

        capacity = (num_classes - 1) ** sid_depth
        if len(item_ids) > capacity:
            raise ValueError(
                f"num_classes={num_classes}, sid_depth={sid_depth} can encode {capacity} items, "
                f"but KuaiRand subset has {len(item_ids)} videos."
            )

        self.item_to_dense = {
            item_id: idx + 1 for idx, item_id in enumerate(sorted(item_ids))
        }
        self.sid_table = _build_sid_table(
            len(self.item_to_dense), sid_depth, num_classes
        )
        self.user_sequences: list[torch.Tensor] = []
        self.user_rewards: list[torch.Tensor] = []
        self.user_ltv: list[torch.Tensor] = []
        self.user_features: list[torch.Tensor] = []
        self.index: list[tuple[int, int]] = []

        for user_id in sorted(by_user):
            events = sorted(by_user[user_id])
            seq = [self.item_to_dense[video_id] for _, video_id, _ in events]
            rewards = [reward for _, _, reward in events]
            if len(seq) <= min_hist_len:
                continue
            seq_idx = len(self.user_sequences)
            self.user_sequences.append(torch.tensor(seq, dtype=torch.long))
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            self.user_rewards.append(reward_tensor)
            self.user_ltv.append(_discounted_returns(reward_tensor, gamma))
            self.user_features.append(
                _encode_user_features(user_rows.get(user_id), user_id, user_feat_dim)
            )
            for pos in range(min_hist_len, len(seq)):
                self.index.append((seq_idx, pos))

        if not self.index:
            raise ValueError(f"No KuaiRand training examples found in {data_path}")

        if max_samples is not None and len(self.index) > max_samples:
            g = torch.Generator().manual_seed(seed)
            chosen = torch.randperm(len(self.index), generator=g)[:max_samples].tolist()
            self.index = [self.index[i] for i in chosen]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_idx, pos = self.index[idx]
        seq = self.user_sequences[seq_idx]
        start = max(0, pos - self.max_hist_len)
        hist_items = seq[start:pos]
        hist_len = hist_items.numel()

        hist_sid = torch.zeros(self.max_hist_len, self.sid_depth, dtype=torch.long)
        hist_sid[:hist_len] = self.sid_table[hist_items]
        target_sid = self.sid_table[seq[pos]].clone()

        reward = self.user_rewards[seq_idx][pos]
        weights = torch.linspace(0.5, 1.0, self.sid_depth)
        immediate_reward = reward * weights
        ltv = self.user_ltv[seq_idx][pos]
        return (
            target_sid,
            self.user_features[seq_idx].clone(),
            hist_sid,
            torch.tensor(hist_len, dtype=torch.long),
            immediate_reward,
            ltv,
        )


def _find_kuairand_data_dir(data_dir: str | Path) -> Path:
    root = Path(data_dir).expanduser()
    candidates = [
        root / "KuaiRand-Pure" / "data",
        root / "data",
        root,
    ]
    for path in candidates:
        if (path / KUAI_LOG_FILES[0]).exists() or (path / KUAI_LOG_FILES[1]).exists():
            return path
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "KuaiRand-Pure logs were not found. Run "
        "`python3 examples/onerec/prepare_kuairand.py --data-dir examples/onerec/data` first.\n"
        f"Tried:\n{tried}"
    )


def _load_kuairand_user_features(
    data_path: Path, user_feat_dim: int
) -> dict[int, torch.Tensor]:
    path = data_path / "user_features_pure.csv"
    if not path.exists():
        return {}
    features: dict[int, torch.Tensor] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                user_id = int(row["user_id"])
            except (KeyError, TypeError, ValueError):
                continue
            features[user_id] = _encode_user_features(row, user_id, user_feat_dim)
    return features


def _row_timestamp(row: dict[str, str], fallback: int) -> int:
    for key in ("time_ms", "timestamp"):
        value = row.get(key)
        if value:
            try:
                return int(float(value))
            except ValueError:
                pass
    date = row.get("date")
    hourmin = row.get("hourmin")
    if date and hourmin:
        try:
            return int(date) * 10_000 + int(hourmin)
        except ValueError:
            pass
    return fallback


def _row_reward(row: dict[str, str]) -> float:
    click = _float_field(row, "is_click")
    long_view = _float_field(row, "long_view")
    like = _float_field(row, "is_like")
    follow = _float_field(row, "is_follow")
    comment = _float_field(row, "is_comment")
    forward = _float_field(row, "is_forward")
    hate = _float_field(row, "is_hate")
    play_time = _float_field(row, "play_time_ms")
    duration = _float_field(row, "duration_ms")
    watch_ratio = min(play_time / duration, 1.0) if duration > 0 else 0.0
    reward = (
        click
        + 0.5 * long_view
        + 0.25 * (like + follow + comment + forward)
        + 0.1 * watch_ratio
    )
    reward = reward - 0.25 * hate
    return max(reward, 0.0)


def _float_field(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def _encode_user_features(
    row: dict[str, str] | torch.Tensor | None, user_id: int, dim: int
) -> torch.Tensor:
    if torch.is_tensor(row):
        return row.float()

    features = torch.zeros(dim, dtype=torch.float32)
    features[user_id % dim] = 1.0
    if not row:
        return features

    offset = 0
    for key in sorted(row):
        if key == "user_id" or offset >= dim:
            continue
        value = row[key]
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
            sign = -1.0 if numeric < 0 else 1.0
            features[offset] = sign * min(
                torch.log1p(torch.tensor(abs(numeric))).item() / 10.0, 1.0
            )
        except ValueError:
            bucket = _stable_bucket(f"{key}:{value}", 997)
            features[offset] = bucket / 996.0
        offset += 1
    return features


def _build_sid_table(num_items: int, sid_depth: int, num_classes: int) -> torch.Tensor:
    base = num_classes - 1
    table = torch.zeros(num_items + 1, sid_depth, dtype=torch.long)
    for dense_item_id in range(1, num_items + 1):
        value = dense_item_id - 1
        tokens = []
        for level in range(sid_depth):
            divisor = base ** (sid_depth - level - 1)
            tokens.append((value // divisor) % base + 1)
        table[dense_item_id] = torch.tensor(tokens, dtype=torch.long)
    return table


def _discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.tensor(0.0, dtype=rewards.dtype)
    for idx in range(rewards.numel() - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def get_kuairand_dataloader(
    data_dir: str | Path,
    batch_size: int,
    max_hist_len: int,
    sid_depth: int,
    num_classes: int,
    user_feat_dim: int,
    max_rows: int | None = 500_000,
    max_samples: int | None = 100_000,
    max_users: int | None = None,
    include_random: bool = False,
    min_hist_len: int = 1,
    gamma: float = 0.85,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    dataset = KuaiRandSequenceDataset(
        data_dir=data_dir,
        max_hist_len=max_hist_len,
        sid_depth=sid_depth,
        num_classes=num_classes,
        user_feat_dim=user_feat_dim,
        max_rows=max_rows,
        max_samples=max_samples,
        max_users=max_users,
        include_random=include_random,
        min_hist_len=min_hist_len,
        gamma=gamma,
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
