#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSTU Benchmark Results Analyzer

Extract FLOPS and MFU metrics from experiment logs and visualize comparisons.

Usage:
    python analyze_results.py <results_dir> [options]

Examples:
    # Analyze all experiments in a batch result directory
    python analyze_results.py training/benchmark/results/20260205_072123
    
    # Save plot to file instead of displaying
    python analyze_results.py training/benchmark/results/20260205_072123 --output comparison.png
    
    # Use bar chart (default)
    python analyze_results.py training/benchmark/results/20260205_072123 --plot-type bar
    
    # Use line chart
    python analyze_results.py training/benchmark/results/20260205_072123 --plot-type line
    
    # Skip first N iterations (warmup)
    python analyze_results.py training/benchmark/results/20260205_072123 --skip-warmup 1
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Try to import matplotlib, provide helpful message if not available
try:
    import matplotlib

    # IMPORTANT: Must set backend BEFORE importing pyplot
    # Use non-interactive backend for headless servers (no display)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def extract_metrics_from_log(
    log_path: str, skip_warmup: int = 0
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract max FLOPS and MFU from a log file.

    Args:
        log_path: Path to log file
        skip_warmup: Number of initial iterations to skip (warmup)

    Returns:
        Tuple of (max_flops, max_mfu) or (None, None) if not found
    """
    # Regex patterns
    # Pattern: "achieved FLOPS 67.81 TFLOPS, MFU 10.87%"
    flops_pattern = r"achieved FLOPS\s+([\d.]+)\s+TFLOPS"
    mfu_pattern = r"MFU\s+([\d.]+)%"

    flops_values = []
    mfu_values = []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                flops_match = re.search(flops_pattern, line)
                mfu_match = re.search(mfu_pattern, line)

                if flops_match and mfu_match:
                    flops_values.append(float(flops_match.group(1)))
                    mfu_values.append(float(mfu_match.group(1)))
    except Exception as e:
        print(f"Warning: Error reading {log_path}: {e}")
        return None, None

    if not flops_values:
        return None, None

    # Skip warmup iterations
    if skip_warmup > 0 and len(flops_values) > skip_warmup:
        flops_values = flops_values[skip_warmup:]
        mfu_values = mfu_values[skip_warmup:]

    return max(flops_values), max(mfu_values)


def find_log_files(results_dir: str) -> Dict[str, str]:
    """
    Find all experiment log files in the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dict mapping experiment names to log file paths
    """
    experiments: Dict[str, str] = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Directory not found: {results_dir}")
        return experiments

    # Look for experiment subdirectories
    for exp_dir in sorted(results_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Skip summary.txt and other non-directory items
        if exp_dir.name in ["summary.txt"]:
            continue

        # Find .log files in the experiment directory
        log_files = list(exp_dir.glob("*.log"))
        if log_files:
            # Use the first (or only) log file
            experiments[exp_dir.name] = str(log_files[0])

    return experiments


def analyze_experiments(
    results_dir: str, skip_warmup: int = 0
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all experiments in the results directory.

    Args:
        results_dir: Path to results directory
        skip_warmup: Number of warmup iterations to skip

    Returns:
        Dict mapping experiment names to their metrics
    """
    experiments = find_log_files(results_dir)

    if not experiments:
        print(f"No experiments found in {results_dir}")
        return {}

    results = {}
    print(f"\nAnalyzing {len(experiments)} experiments...")
    print("-" * 60)

    for exp_name, log_path in experiments.items():
        max_flops, max_mfu = extract_metrics_from_log(log_path, skip_warmup)

        if max_flops is not None and max_mfu is not None:
            results[exp_name] = {
                "max_flops": max_flops,
                "max_mfu": max_mfu,
                "log_path": log_path,
            }
            print(
                f"  {exp_name:30s} | FLOPS: {max_flops:7.2f} TFLOPS | MFU: {max_mfu:6.2f}%"
            )
        else:
            print(f"  {exp_name:30s} | No metrics found")

    print("-" * 60)
    return results


FLAG_DISPLAY = {
    "kernel_backend=cutlass": "CUTLASS Kernel",
    "caching": "Caching",
    "recompute_layernorm": "Recompute LN",
    "balanced_shuffler": "Balanced Shuffler",
    "pipeline_type=prefetch": "Prefetch Pipeline",
    "tp_size": "TP",
}


def _parse_experiments_txt(path: str) -> Dict[str, set]:
    """Parse experiments.txt → {exp_name: set_of_normalised_flags}."""
    experiments: Dict[str, set] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            name = parts[0].strip()
            opts = parts[1].strip() if len(parts) > 1 else ""
            flags: set = set()
            tokens = opts.split()
            i = 0
            while i < len(tokens):
                if tokens[i].startswith("--"):
                    key = tokens[i].lstrip("-")
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                        flags.add(f"{key}={tokens[i + 1]}")
                        i += 2
                    else:
                        flags.add(key)
                        i += 1
                else:
                    i += 1
            experiments[name] = flags
    return experiments


def _flag_label(flag: str) -> str:
    """Convert a raw flag string to a human-readable label."""
    if flag in FLAG_DISPLAY:
        return FLAG_DISPLAY[flag]
    key_val = flag.split("=", 1)
    if len(key_val) == 2:
        key, val = key_val
        if key in FLAG_DISPLAY:
            return f"{FLAG_DISPLAY[key]}={val}"
        pretty_key = key.replace("_", " ").title()
        return f"{pretty_key}={val}"
    return flag.replace("_", " ").title()


def _compute_deltas(experiments_txt: Optional[str], sorted_exp_names: list) -> list:
    """Return a list of human-readable delta labels (one per experiment)."""
    if not experiments_txt or not os.path.isfile(experiments_txt):
        return [""] * len(sorted_exp_names)

    exp_flags = _parse_experiments_txt(experiments_txt)

    deltas: list = []
    for i, name in enumerate(sorted_exp_names):
        if i == 0:
            deltas.append("Baseline")
            continue
        prev = exp_flags.get(sorted_exp_names[i - 1], set())
        curr = exp_flags.get(name, set())
        new_flags = curr - prev
        if new_flags:
            labels = [f"+{_flag_label(f)}" for f in sorted(new_flags)]
            deltas.append("\n".join(labels))
        else:
            deltas.append("")
    return deltas


def _plot_single_metric(
    ax: plt.Axes,
    exp_names: list,
    values: list,
    ylabel: str,
    chart_title: str,
    plot_type: str,
    fmt_func,
    color: str,
    marker: str,
    delta_labels: Optional[list] = None,
) -> None:
    """Plot a single metric (TFLOPS or MFU) on the given axes."""
    colors = plt.cm.Set2(range(len(exp_names)))
    x = range(len(exp_names))
    val_range = max(values) - min(values) if len(values) > 1 else max(values)

    if plot_type == "bar":
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for idx, (bar, val) in enumerate(zip(bars, values)):
            cx = bar.get_x() + bar.get_width() / 2
            val_y = bar.get_height() + val_range * 0.01 + 0.1
            ax.text(
                cx,
                val_y,
                fmt_func(val),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            if delta_labels and delta_labels[idx]:
                ax.text(
                    cx,
                    val_y + val_range * 0.06,
                    delta_labels[idx],
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontstyle="italic",
                    color="#555555",
                )
    else:
        ax.plot(x, values, f"{marker}-", color=color, linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=10)
        ax.grid(alpha=0.3)
        for idx, val in enumerate(values):
            ax.annotate(
                fmt_func(val),
                (idx, val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
            if delta_labels and delta_labels[idx]:
                ax.annotate(
                    delta_labels[idx],
                    (idx, val),
                    textcoords="offset points",
                    xytext=(0, 22),
                    ha="center",
                    fontsize=7,
                    fontstyle="italic",
                    color="#555555",
                )

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(chart_title, fontsize=14, fontweight="bold")

    if delta_labels:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)


def plot_comparison(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    plot_type: str = "bar",
    title: str = "HSTU Benchmark Comparison",
    experiments_txt: Optional[str] = None,
) -> None:
    """
    Generate a single dual-axis comparison plot: bars for TFLOPS (left axis),
    line for MFU% (right axis).

    When *experiments_txt* is given, each bar is annotated with the
    optimisation switch that was additionally enabled compared to the
    previous experiment.
    """
    if not results:
        print("No results to plot")
        return

    exp_names = sorted(results.keys())
    flops_values = [results[exp]["max_flops"] for exp in exp_names]
    mfu_values = [results[exp]["max_mfu"] for exp in exp_names]

    delta_labels = _compute_deltas(experiments_txt, exp_names)

    bar_colors = plt.cm.Set2(range(len(exp_names)))
    x = list(range(len(exp_names)))

    fig, ax_flops = plt.subplots(figsize=(max(8, len(exp_names) * 1.5), 7))
    ax_mfu = ax_flops.twinx()

    # --- Left axis: TFLOPS bars ---
    bars = ax_flops.bar(
        x,
        flops_values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
        zorder=2,
        alpha=0.85,
        label="TFLOPS",
    )
    ax_flops.set_ylabel("Max TFLOPS", fontsize=12)
    ax_flops.set_xlabel("Experiment", fontsize=12)
    ax_flops.grid(axis="y", alpha=0.25, zorder=0)

    flops_range = max(flops_values) - min(flops_values)
    for idx, (bar, fv) in enumerate(zip(bars, flops_values)):
        cx = bar.get_x() + bar.get_width() / 2
        val_y = bar.get_height() + flops_range * 0.01 + 0.1
        ax_flops.text(
            cx,
            val_y,
            f"{fv:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        if delta_labels and delta_labels[idx]:
            ax_flops.text(
                cx,
                val_y + flops_range * 0.06,
                delta_labels[idx],
                ha="center",
                va="bottom",
                fontsize=7,
                fontstyle="italic",
                color="#555555",
            )

    if delta_labels:
        ymin, ymax = ax_flops.get_ylim()
        ax_flops.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)

    # --- Right axis: MFU line ---
    mfu_color = "#E53935"
    ax_mfu.plot(
        x,
        mfu_values,
        "s-",
        color=mfu_color,
        linewidth=2.5,
        markersize=7,
        zorder=3,
        label="MFU (%)",
    )
    for idx, mv in enumerate(mfu_values):
        ax_mfu.annotate(
            f"{mv:.2f}%",
            (idx, mv),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=8,
            color=mfu_color,
            fontweight="bold",
        )
    ax_mfu.set_ylabel("Max MFU (%)", fontsize=12, color=mfu_color)
    ax_mfu.tick_params(axis="y", labelcolor=mfu_color)
    ax_mfu.set_ylim(0, max(mfu_values) * 1.35)

    # --- X axis ---
    ax_flops.set_xticks(x)
    ax_flops.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=10)

    # --- Legend & title ---
    [bars]
    lines_line = ax_mfu.get_lines()
    ax_flops.legend(
        [bars, lines_line[0]],
        ["TFLOPS", "MFU (%)"],
        loc="upper left",
        fontsize=9,
    )
    ax_flops.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_path:
        stem, ext = os.path.splitext(output_path)
        if not ext:
            ext = ".png"
        fpath = f"{stem}{ext}"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {fpath}")
    else:
        plt.show()
    plt.close(fig)


def print_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a summary table of results."""
    if not results:
        return

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Experiment':<30} | {'Max FLOPS (TFLOPS)':>18} | {'Max MFU (%)':>12}")
    print("-" * 70)

    # Sort by FLOPS descending
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["max_flops"], reverse=True
    )

    best_flops = sorted_results[0][1]["max_flops"]
    best_mfu = max(r["max_mfu"] for r in results.values())

    for exp_name, metrics in sorted_results:
        flops_marker = " 🏆" if metrics["max_flops"] == best_flops else ""
        mfu_marker = " 🏆" if metrics["max_mfu"] == best_mfu else ""
        print(
            f"{exp_name:<30} | {metrics['max_flops']:>15.2f}{flops_marker:>3} | {metrics['max_mfu']:>9.2f}%{mfu_marker}"
        )

    print("=" * 70)

    # Calculate speedup relative to first experiment
    if len(sorted_results) > 1:
        baseline_name = sorted_results[-1][0]  # Worst performer as baseline
        baseline_flops = sorted_results[-1][1]["max_flops"]

        print(f"\nSpeedup relative to {baseline_name}:")
        for exp_name, metrics in sorted_results:
            if exp_name != baseline_name:
                speedup = metrics["max_flops"] / baseline_flops
                print(f"  {exp_name}: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze HSTU benchmark results and visualize comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        help="Path to results directory (e.g., training/benchmark/results/20260205_072123)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for the plot (e.g., comparison.png). If not specified, display interactively.",
    )
    parser.add_argument(
        "--plot-type",
        choices=["bar", "line"],
        default="bar",
        help="Type of plot: bar or line (default: bar)",
    )
    parser.add_argument(
        "--skip-warmup",
        type=int,
        default=1,
        help="Number of warmup iterations to skip (default: 1)",
    )
    parser.add_argument(
        "--title", default="HSTU Benchmark Comparison", help="Plot title"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print summary, do not generate plot",
    )
    parser.add_argument(
        "--experiments-txt",
        default=None,
        help="Path to experiments.txt for delta annotations on bars. "
        "Auto-detected from script location if omitted.",
    )

    args = parser.parse_args()

    # Auto-detect experiments.txt when not explicitly provided
    exp_txt: Optional[str] = args.experiments_txt
    if exp_txt is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, "..", "experiments.txt")
        if os.path.isfile(candidate):
            exp_txt = candidate

    # Analyze experiments
    results = analyze_experiments(args.results_dir, args.skip_warmup)

    if not results:
        print("No valid results found. Please check the log files.")
        sys.exit(1)

    # Print summary table
    print_summary_table(results)

    # Generate plot
    if not args.no_plot:
        plot_comparison(
            results,
            args.output,
            args.plot_type,
            args.title,
            experiments_txt=exp_txt,
        )


if __name__ == "__main__":
    main()
