#!/usr/bin/env python3

"""Plot combined HSTU attention benchmark heatmaps from saved JSON results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage

RESULTS_FILE = "hstu_attn_mfu_results.json"
PHASE_LABELS = {"fwd": "FWD", "bwd": "BWD", "e2e": "E2E"}


def load_runs(output_dir: Path) -> list[dict]:
    runs = []
    for path in output_dir.rglob(RESULTS_FILE):
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        runs.append(data)

    if not runs:
        raise ValueError(f"no {RESULTS_FILE} files found under {output_dir}")

    return sorted(
        runs,
        key=lambda run: (
            run.get("kernel_backend", ""),
            run["num_heads"],
            run["dim_per_head"],
            run["_path"],
        ),
    )


def validate_runs(runs: list[dict]) -> None:
    reference = runs[0]
    shared_keys = (
        "device_name",
        "peak_tflops",
        "kernel_backend",
        "batch_sizes",
        "seqlens",
        "phases",
        "performance_percentile",
        "time_percentiles",
    )
    for run in runs[1:]:
        for key in shared_keys:
            if run.get(key) != reference.get(key):
                raise ValueError(
                    "all benchmark JSON files must share "
                    f"{key}: {reference['_path']} != {run['_path']}"
                )

    performance_key = f"p{reference.get('performance_percentile', 10)}"
    for run in runs:
        for result_key, result in run["results"].items():
            for phase in run["phases"]:
                percentiles = result.get(f"{phase}_time_percentiles_ms", {})
                if performance_key not in percentiles:
                    raise ValueError(
                        f"{run['_path']} result {result_key} is missing "
                        f"{phase} {performance_key} timing"
                    )


def build_matrix(run: dict, phase: str, key: str) -> np.ndarray:
    matrix = np.full((len(run["batch_sizes"]), len(run["seqlens"])), np.nan)
    for row, batch_size in enumerate(run["batch_sizes"]):
        for column, seqlen in enumerate(run["seqlens"]):
            result = run["results"].get(f"{batch_size},{seqlen}")
            if result is None:
                continue
            matrix[row, column] = result[f"{phase}_{key}"]
    return matrix


def draw_panel(
    ax: plt.Axes,
    run: dict,
    phase: str,
    show_x_label: bool,
) -> AxesImage:
    tflops = build_matrix(run, phase, "tflops")
    mfu = build_matrix(run, phase, "mfu")
    elapsed_ms = build_matrix(run, phase, "ms")
    masked = np.ma.masked_invalid(tflops)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#d9d9d9")
    finite = tflops[np.isfinite(tflops)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    image = ax.imshow(
        masked,
        cmap=cmap,
        aspect="auto",
        origin="upper",
        vmin=0,
        vmax=vmax,
    )
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.02, fraction=0.046)
    colorbar.set_label("TFLOPS", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    ax.set_xticks(range(len(run["seqlens"])))
    ax.set_xticklabels(
        [str(value) for value in run["seqlens"]],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(range(len(run["batch_sizes"])))
    ax.set_yticklabels([str(value) for value in run["batch_sizes"]], fontsize=7)
    ax.set_xlabel("Sequence length" if show_x_label else "", fontsize=9)
    ax.set_ylabel(f"{PHASE_LABELS[phase]}\nBatch size", fontsize=9)

    midpoint = np.nanmedian(finite) if finite.size else 0.0
    for row in range(tflops.shape[0]):
        for column in range(tflops.shape[1]):
            if not np.isfinite(tflops[row, column]):
                text = "OOM"
                color = "#333333"
            else:
                text = (
                    f"{tflops[row, column]:.0f} TF\n"
                    f"{mfu[row, column]:.1f}%\n"
                    f"{elapsed_ms[row, column]:.3g} ms"
                )
                color = "white" if tflops[row, column] < midpoint else "#111111"
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=5.5,
                linespacing=1.05,
                color=color,
            )

    ax.set_title(
        f"heads={run['num_heads']}, head_dim={run['dim_per_head']}\n"
        f"hidden={run['num_heads'] * run['dim_per_head']}",
        fontsize=12,
    )
    return image


def plot_heatmap(
    runs: list[dict],
    output: Path,
    title: str | None,
    dpi: int,
) -> None:
    phases = runs[0]["phases"]
    figure_title = title or (
        f"{runs[0]['kernel_backend'].capitalize()} HSTU " "TFLOPS / MFU / Time"
    )
    fig, axes = plt.subplots(
        len(phases),
        len(runs),
        figsize=(6.0 * len(runs), 4.0 * len(phases) + 1.0),
        squeeze=False,
    )
    for row, phase in enumerate(phases):
        for column, run in enumerate(runs):
            draw_panel(
                axes[row, column],
                run,
                phase,
                row == len(phases) - 1,
            )

    device_name = runs[0]["device_name"]
    peak_tflops = runs[0]["peak_tflops"]
    fig.suptitle(
        f"{figure_title}\n{device_name} | Peak BF16: {peak_tflops:.0f} TFLOPS",
        fontsize=15,
    )
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.06,
        top=0.90,
        wspace=0.22,
        hspace=0.25,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, transparent=False)
    plt.close(fig)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read hstu_attn_mfu_results.json files from --output-dir and plot "
            "a combined HSTU attention TFLOPS/MFU/time heatmap."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=f"Directory containing one or more {RESULTS_FILE} files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <output-dir>/hstu_attn_mfu.png.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Device and peak TFLOPS are appended.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    runs = load_runs(args.output_dir)
    validate_runs(runs)
    output = args.output or args.output_dir / "hstu_attn_mfu.png"
    plot_heatmap(runs, output, args.title, args.dpi)


if __name__ == "__main__":
    main()
