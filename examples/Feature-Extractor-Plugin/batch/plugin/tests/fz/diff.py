"""Bit-exact table comparison (pyarrow + numpy).  Integers/bool/date/timestamp must be identical; float/double compared on raw
bits after mapping every NaN to one canonical NaN (so -0.0 vs 0.0 IS a mismatch, NaN payload differences are not)."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CANON = {32: np.uint32(0x7FC00000), 64: np.uint64(0x7FF8000000000000)}


def load(path):
    t = pq.read_table(path)
    idx = np.argsort(t.column("id").to_numpy())
    return t.take(pa.array(idx))


def _bits(col):
    """(valid_mask, comparable numpy array) for one arrow column"""
    ty = col.type
    valid = col.is_valid().to_numpy(zero_copy_only=False)
    if pa.types.is_floating(ty):
        w = 32 if ty.bit_width == 32 else 64
        a = col.fill_null(0.0).to_numpy(zero_copy_only=False)
        isn = np.isnan(a)
        b = a.view(np.uint32 if w == 32 else np.uint64).copy()
        b[isn] = CANON[w]
        return valid, b
    if pa.types.is_boolean(ty):
        return valid, col.fill_null(False).to_numpy(zero_copy_only=False).astype(
            np.uint8
        )
    if pa.types.is_timestamp(ty):
        return valid, col.cast(pa.timestamp("us")).cast(pa.int64()).fill_null(
            0
        ).to_numpy(zero_copy_only=False)
    if pa.types.is_date(ty):
        return valid, col.cast(pa.int32()).fill_null(0).to_numpy(zero_copy_only=False)
    return valid, col.fill_null(0).to_numpy(zero_copy_only=False)


def compare(a, b, ignore=("id",)):
    """returns {'rowset': (n_only_a, n_only_b, sample_ids) | None, 'cols': {name: {'n': count, 'ids': [...], 'type': (ta, tb)}}}"""
    res = {"rowset": None, "cols": {}}
    ida, idb = a.column("id").to_numpy(), b.column("id").to_numpy()
    if len(ida) != len(idb) or not np.array_equal(ida, idb):
        only_a = np.setdiff1d(ida, idb)
        only_b = np.setdiff1d(idb, ida)
        res["rowset"] = (
            len(only_a),
            len(only_b),
            [int(x) for x in np.concatenate([only_a[:3], only_b[:3]])],
        )
        common = np.intersect1d(ida, idb)
        a = a.take(pa.array(np.searchsorted(ida, common)))
        b = b.take(pa.array(np.searchsorted(idb, common)))
        ida = common
    for name in a.column_names:
        if name in ignore or name not in b.column_names:
            continue
        ca, cb = a.column(name).combine_chunks(), b.column(name).combine_chunks()
        if ca.type != cb.type:
            res["cols"][name] = {
                "n": len(ida),
                "ids": [int(x) for x in ida[:3]],
                "type": (str(ca.type), str(cb.type)),
            }
            continue
        va, xa = _bits(ca)
        vb, xb = _bits(cb)
        bad = (va != vb) | (va & vb & (xa != xb))
        n = int(bad.sum())
        if n:
            res["cols"][name] = {
                "n": n,
                "ids": [int(x) for x in ida[bad][:5]],
                "type": (str(ca.type), str(cb.type)),
            }
    return res


def is_clean(r):
    return r["rowset"] is None and not r["cols"]


def fmt_val(v):
    import math
    import struct

    if isinstance(v, float):
        return f"{v!r} ({struct.pack('>d', v).hex()})" if not math.isnan(v) else "nan"
    return repr(v)


def short_err(e, n=300):
    import re

    msg = str(e)
    m = re.search(r"\[([A-Z_][A-Z_.0-9]+)\][^\n]{0,200}", msg)
    if m:
        return m.group(0)[:n]
    cb = re.findall(r"Caused by: ([^\n]{0,250})", msg)
    if cb:
        return cb[-1][:n]
    m = re.search(r"(\w+Exception|\w+Error)[^\n]{0,200}", msg)
    if m:
        return m.group(0)[:n]
    return " | ".join([l for l in msg.splitlines() if l.strip()][:2])[:n]
