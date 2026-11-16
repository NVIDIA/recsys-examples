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

"""Download and extract MovieLens 1M for the HSTU example."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import BadZipFile, ZipFile

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def ensure_movielens_1m(data_dir: str | Path, url: str = MOVIELENS_1M_URL) -> Path:
    root = Path(data_dir).expanduser()
    ratings_path = root / "ml-1m" / "ratings.dat"
    if ratings_path.exists():
        print(f"MovieLens 1M already exists: {ratings_path}")
        return ratings_path

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "ml-1m.zip"
    if archive_path.exists() and not _valid_zip(archive_path):
        print(f"Removing incomplete or invalid archive: {archive_path}")
        archive_path.unlink()
    if not archive_path.exists():
        print(f"Downloading {url} -> {archive_path}")
        urlretrieve(url, archive_path)
    if not _valid_zip(archive_path):
        archive_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded archive is not a valid zip file: {archive_path}")

    print(f"Extracting {archive_path} -> {root}")
    with ZipFile(archive_path) as zf:
        zf.extractall(root)

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Extraction finished but {ratings_path} was not found."
        )
    print(f"Ready: {ratings_path}")
    return ratings_path


def _valid_zip(path: Path) -> bool:
    try:
        with ZipFile(path) as zf:
            return zf.testzip() is None
    except BadZipFile:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir", type=str, default=str(Path(__file__).resolve().parent / "data")
    )
    args = p.parse_args()
    ensure_movielens_1m(args.data_dir)


if __name__ == "__main__":
    main()
