"""Summarize GR/SGLang performance sweep JSON files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from tool_utils import optional_float, read_json


def _number(value: Any) -> float | None:
    return optional_float(value)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _row_for(gr_path: Path, sglang_path: Path) -> dict[str, Any]:
    gr = read_json(gr_path)
    sg = read_json(sglang_path)
    gr_wall = _number(gr.get("wall_ms_median"))
    sg_wall = _number(sg.get("wall_ms_median"))
    speedup = sg_wall / gr_wall if gr_wall and sg_wall else None
    context_len = gr.get("context_len") or sg.get("context_len")
    beam_width = gr.get("engine_status", {}).get("max_beam_width") or sg.get("beam_width")
    requests = gr.get("responses") or sg.get("requests")
    return {
        "context_len": context_len,
        "beam_width": beam_width,
        "batch_requests": requests,
        "gr_wall_ms": gr_wall,
        "sglang_wall_ms": sg_wall,
        "speedup_sglang_over_gr": speedup,
        "winner": "GR" if speedup and speedup > 1.0 else "SGLang",
        "gr_decode_ms": _number(gr.get("decode_ms_median")),
        "gr_prefill_ms": _number(gr.get("prefill_ms_median")),
        "gr_qps": (requests / (gr_wall / 1000.0)) if gr_wall and requests else None,
        "sglang_qps": _number(sg.get("qps_median")),
    }


def build_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for gr_path in sorted(root.glob("gr_ctx*_beam*_req*.json")):
        suffix = gr_path.name.removeprefix("gr_")
        sglang_path = root / f"sglang_{suffix}"
        if not sglang_path.exists():
            continue
        rows.append(_row_for(gr_path, sglang_path))
    rows.sort(key=lambda row: (int(row["context_len"]), int(row["beam_width"]), int(row["batch_requests"])))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "context_len",
        "beam_width",
        "batch_requests",
        "gr_wall_ms",
        "sglang_wall_ms",
        "speedup_sglang_over_gr",
        "winner",
        "gr_prefill_ms",
        "gr_decode_ms",
        "gr_qps",
        "sglang_qps",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# GR vs SGLang Performance Sweep",
        "",
        "| ctx | beam | batch | GR ms | SGLang ms | SGLang/GR | winner | GR prefill | GR decode |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {ctx} | {beam} | {batch} | {gr} | {sg} | {speedup} | {winner} | {prefill} | {decode} |".format(
                ctx=row["context_len"],
                beam=row["beam_width"],
                batch=row["batch_requests"],
                gr=_fmt(row["gr_wall_ms"]),
                sg=_fmt(row["sglang_wall_ms"]),
                speedup=_fmt(row["speedup_sglang_over_gr"]),
                winner=row["winner"],
                prefill=_fmt(row["gr_prefill_ms"]),
                decode=_fmt(row["gr_decode_ms"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    rows = build_rows(root)
    write_csv(rows, root / "summary.csv")
    write_markdown(rows, root / "summary.md")
    print((root / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
