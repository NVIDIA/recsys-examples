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

"""Download and extract KuaiRand-Pure for the OneRec example."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

KUAIRAND_PURE_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
KUAIRAND_PURE_MD5 = "0820331067a3784d9691136f772b35a7"


def ensure_kuairand_pure(data_dir: str | Path, url: str = KUAIRAND_PURE_URL) -> Path:
    root = Path(data_dir).expanduser()
    data_path = root / "KuaiRand-Pure" / "data"
    if (data_path / "log_standard_4_08_to_4_21_pure.csv").exists():
        print(f"KuaiRand-Pure already exists: {data_path}")
        return data_path

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "KuaiRand-Pure.tar.gz"
    if archive_path.exists():
        actual_md5 = _md5(archive_path)
        if actual_md5 != KUAIRAND_PURE_MD5:
            print(
                f"Removing incomplete or invalid archive: {archive_path} ({actual_md5})"
            )
            archive_path.unlink()
    if not archive_path.exists():
        print(f"Downloading {url} -> {archive_path}")
        urlretrieve(url, archive_path)

    actual_md5 = _md5(archive_path)
    if actual_md5 != KUAIRAND_PURE_MD5:
        raise ValueError(
            f"Unexpected md5 for {archive_path}: {actual_md5}; expected {KUAIRAND_PURE_MD5}"
        )

    print(f"Extracting {archive_path} -> {root}")
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(root)

    if not data_path.exists():
        raise FileNotFoundError(f"Extraction finished but {data_path} was not found.")
    print(f"Ready: {data_path}")
    return data_path


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir", type=str, default=str(Path(__file__).resolve().parent / "data")
    )
    args = p.parse_args()
    ensure_kuairand_pure(args.data_dir)


if __name__ == "__main__":
    main()
