#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the KuaiRand-27K data for the streaming benchmark driver.

Downloads the public KuaiRand-27K archive (or uses CSV files you already extracted), then writes
fixed-width little-endian int64 column files that the driver memory-maps:

    OUT/log/<column>.i64     the exposure log, rows globally sorted by (time_ms, original CSV line),
                             so replaying in file order is an event-time ordered stream
    OUT/video/<column>.i64   video_id, author_id, video_duration (in CSV order; duration truncated to an integer)
    OUT/log/meta.txt, OUT/video/meta.txt

Empty or non-numeric fields in a kept column become -2^63 (Long.MIN_VALUE, read by the driver as NULL);
fractional values are truncated toward zero. This is the same rule as driver/src/main/java/fep/KuaiRandConvert.java,
which is kept as an independent reference implementation (see REPRODUCE.md, "Check the data").

Examples
    prepare_stream_data.py --out $FEP_DATA_DIR                          # download (about 9.9 GB), verify, extract, convert
    prepare_stream_data.py --out $FEP_DATA_DIR --src $KUAIRAND/data     # CSV files already extracted
    prepare_stream_data.py --out $SMALL --src $KUAIRAND/data --limit-rows 500000   # small sample for a smoke test

Needs Python 3.8+ and numpy. pandas is optional and makes the full conversion several times faster.
Memory: about 25 GB for the full log (14 int64 columns of 92,919,789 rows plus the sort permutation).

KuaiRand: https://kuairand.com (Gao et al., CIKM 2022, CC BY-SA 4.0). Please follow the dataset page for
citation and license terms.
"""
import argparse
import csv
import hashlib
import os
import sys
import tarfile
import urllib.error
import urllib.request

import numpy as np

URL = "https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz"
MD5 = "3e3c799a24e2d23a4d2c757fbf9adf59"  # from the dataset page
ARCHIVE = "KuaiRand-27K.tar.gz"
LOG_FILE = "log_standard_4_22_to_5_08_27k_part2.csv"
VIDEO_FILE = "video_features_basic_27k.csv"
NULL = np.iinfo(np.int64).min

# kept columns, in the order the driver expects (it looks the files up by name)
LOG_COLS = [
    "user_id",
    "video_id",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "is_rand",
    "tab",
]
VIDEO_COLS = ["video_id", "author_id", "video_duration"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- download


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def download(dest):
    """Resumable download of the archive (HTTP Range). Returns the local path."""
    path = os.path.join(dest, ARCHIVE)
    os.makedirs(dest, exist_ok=True)
    done = os.path.getsize(path) if os.path.exists(path) else 0
    req = urllib.request.Request(
        URL, headers={"Range": f"bytes={done}-"} if done else {}
    )
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete
            return path
        raise
    mode = "ab" if done and resp.status == 206 else "wb"
    log(
        f"downloading {URL} -> {path} (resuming at {done} bytes)"
        if mode == "ab"
        else f"downloading {URL} -> {path}"
    )
    with resp, open(path, mode) as out:
        n = 0
        while True:
            block = resp.read(1 << 22)
            if not block:
                break
            out.write(block)
            n += len(block)
    return path


def extract(archive, dest, names):
    """Extract only the wanted CSV files from the archive (a sequential pass over the .tar.gz)."""
    want = {f"KuaiRand-27K/data/{n}": n for n in names}
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            if m.name in want:
                log(f"extracting {m.name}")
                src = tar.extractfile(m)
                with open(os.path.join(dest, want[m.name]), "wb") as out:
                    for block in iter(lambda: src.read(1 << 24), b""):
                        out.write(block)
                del want[m.name]
                if not want:
                    break
    if want:
        raise SystemExit(f"not found in the archive: {sorted(want)}")


# ---------------------------------------------------------------- CSV -> int64 columns


def parse_token(tok):
    """Integral part of a decimal token, NULL for empty/non-numeric (Java converter's rule)."""
    tok = tok.strip()
    if not tok:
        return NULL
    neg = tok[0] == "-"
    body = tok[1:] if neg else tok
    ip = body.split(".", 1)[0]
    if not ip.isdigit() or any(not (c.isdigit() or c == ".") for c in body):
        return NULL
    v = int(ip)
    return -v if neg else v


def read_python(path, want, limit):
    """Reference parser: csv module, one value at a time. Slow (minutes per 10 million rows) but dependency free."""
    cols = [[] for _ in want]
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        idx = [header.index(w) for w in want]
        for i, row in enumerate(rd):
            if limit and i >= limit:
                break
            for c, j in enumerate(idx):
                cols[c].append(parse_token(row[j]) if j < len(row) else NULL)
    return [np.array(c, dtype=np.int64) for c in cols]


def read_pandas(path, want, limit):
    import pandas as pd

    parts = [[] for _ in want]
    it = pd.read_csv(
        path,
        usecols=want,
        dtype=str,
        chunksize=2_000_000,
        nrows=limit or None,
        keep_default_na=False,
        na_filter=False,
        engine="c",
    )
    for chunk in it:
        for c, w in enumerate(want):
            s = chunk[w]
            num = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
            bad = np.isnan(num)
            vals = np.trunc(np.where(bad, 0.0, num)).astype(np.int64)
            vals[bad] = NULL
            parts[c].append(vals)
    return [np.concatenate(p) for p in parts]


def read_csv_cols(path, want, limit, engine):
    if engine == "auto":
        try:
            import pandas  # noqa: F401

            engine = "pandas"
        except ImportError:
            engine = "python"
            log(
                "pandas not found: using the slow pure-Python parser (fine for a sample, install pandas for the full log)"
            )
    log(f"parsing {path} with the {engine} engine")
    return (read_pandas if engine == "pandas" else read_python)(path, want, limit)


def write_i64(path, arr):
    arr.astype("<i8", copy=False).tofile(path)


def convert_log(src, dst, limit, engine):
    os.makedirs(dst, exist_ok=True)
    cols = read_csv_cols(src, LOG_COLS, limit, engine)
    rows = len(cols[0])
    t = cols[LOG_COLS.index("time_ms")]
    if (t == NULL).any():
        raise SystemExit(
            "time_ms has empty values; cannot build an event-time ordered stream"
        )
    log(
        f"{rows} rows, time_ms {t.min()}..{t.max()}; sorting by (time_ms, original line)"
    )
    perm = np.argsort(t, kind="stable")  # stable sort = ties keep the CSV line order
    for name, col in zip(LOG_COLS, cols):
        write_i64(os.path.join(dst, name + ".i64"), col[perm])
    with open(os.path.join(dst, "meta.txt"), "w") as f:
        f.write(
            f"rows={rows}\ncolumns={','.join(LOG_COLS)}\nsource={os.path.basename(src)}\n"
        )
    return rows


def convert_video(src, dst, limit, engine):
    os.makedirs(dst, exist_ok=True)
    cols = read_csv_cols(src, VIDEO_COLS, limit, engine)
    rows = len(cols[0])
    for name, col in zip(VIDEO_COLS, cols):
        write_i64(os.path.join(dst, name + ".i64"), col)
    with open(os.path.join(dst, "meta.txt"), "w") as f:
        f.write(
            f"rows={rows}\ncolumns={','.join(VIDEO_COLS)}\nsource={os.path.basename(src)}\n"
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--out", required=True, help="output directory (this is $FEP_DATA_DIR)"
    )
    ap.add_argument(
        "--src",
        help="directory with the extracted CSV files (KuaiRand-27K/data); skips the download",
    )
    ap.add_argument(
        "--work",
        help="where the archive and the extracted CSV files go when downloading (default OUT/download)",
    )
    ap.add_argument(
        "--log-file",
        default=LOG_FILE,
        help=f"log CSV to convert (default {LOG_FILE}, 92,919,789 rows)",
    )
    ap.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="only read the first N lines of each CSV (smoke tests; a different, smaller data set)",
    )
    ap.add_argument("--engine", choices=["auto", "pandas", "python"], default="auto")
    ap.add_argument(
        "--skip-md5", action="store_true", help="do not verify the archive checksum"
    )
    a = ap.parse_args()

    src = a.src
    if not src:
        work = a.work or os.path.join(a.out, "download")
        archive = download(work)
        if not a.skip_md5:
            log("verifying md5 (about a minute)")
            got = md5_of(archive)
            if got != MD5:
                raise SystemExit(
                    f"md5 mismatch: got {got}, expected {MD5} (delete the partial file and retry)"
                )
        src = os.path.join(work, "csv")
        extract(archive, src, [a.log_file, VIDEO_FILE])
    n = convert_log(
        os.path.join(src, a.log_file),
        os.path.join(a.out, "log"),
        a.limit_rows,
        a.engine,
    )
    log(f"== log: {n} rows")
    n = convert_video(
        os.path.join(src, VIDEO_FILE),
        os.path.join(a.out, "video"),
        a.limit_rows,
        a.engine,
    )
    log(f"== video: {n} rows")


if __name__ == "__main__":
    sys.exit(main())
