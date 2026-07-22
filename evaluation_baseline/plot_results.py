#!/usr/bin/env python3
"""
Plot latency/throughput/quality charts from evaluation_baseline JSON reports.

For every (machine, device) combination found in the results directory (e.g.
"rpi+cpu", "mac+cpu", "mac+mps") and for every model, this produces one PNG
with a grid of bar-chart panels:

    rows    = available dtypes, in order float32 -> float16 -> bfloat16
              (only dtypes that actually have a matching result file are
              plotted; a device+model combination with zero matching files
              is skipped with an error message)
    columns = 3 metric panels, identical across all rows:
        1. Preprocessing / Total processing / TTFT   (3 bars, ms)
        2. System prompt / Tools list / User query / Decode  (4 bars, ms)
        3. Accuracy / Peak RAM / KV cache / Peak GPU  (4 bars, log scale)

All values are the *mean* aggregates from each run's JSON report. A shared
legend for each column is rendered once in a dedicated row at the top of the
figure so it never overlaps the chart area.

Only reports produced with ``--mode kv_cache`` or ``--mode prefix_cache`` have
the system-prompt/tools-list/user-query prefill split needed for columns 1-2
(see evaluation_baseline/run.py); "no_cache" reports are not usable here.

Usage
------
    python evaluation_baseline/plot_results.py
    python evaluation_baseline/plot_results.py --mode kv_cache
    python evaluation_baseline/plot_results.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import MODEL_DISPLAY_NAMES  # noqa: E402

_RESULTS_DIR = _REPO_ROOT / "evaluation_baseline" / "results"
_CHARTS_DIR = _RESULTS_DIR / "charts"

# Row order requested: fp32 first, then fp16, then bf16.
DTYPE_ORDER = ["float32", "float16", "bfloat16"]
DTYPE_LABELS = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot charts from evaluation_baseline JSON reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_RESULTS_DIR,
        help="Directory containing run JSON reports",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_CHARTS_DIR,
        help="Directory to write PNG charts",
    )
    p.add_argument(
        "--mode",
        choices=["kv_cache", "prefix_cache"],
        default="prefix_cache",
        help=(
            "Which mode's reports to chart (only kv_cache/prefix_cache carry "
            "the system-prompt/tools-list/user-query prefill split)"
        ),
    )
    return p.parse_args()


def _load_reports(results_dir: Path, mode: str) -> list[dict]:
    """Load all JSON reports under *results_dir* matching *mode*."""
    reports = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        run_config = doc.get("run_config", {})
        if run_config.get("mode") != mode:
            continue
        reports.append(doc)
    return reports


def _group_reports(reports: list[dict]) -> dict:
    """Group reports by (machine, device) -> model_key -> dtype -> latest report."""
    grouped: dict[tuple[str, str], dict[str, dict[str, dict]]] = {}
    for doc in reports:
        rc = doc["run_config"]
        machine = rc.get("machine", "unknown")
        device = rc.get("device", "unknown")
        model_key = rc.get("model_key", "unknown")
        dtype = rc.get("dtype", "unknown")
        ts = rc.get("timestamp_utc", "")

        device_group = (machine, device)
        grouped.setdefault(device_group, {}).setdefault(model_key, {})
        existing = grouped[device_group][model_key].get(dtype)
        if existing is None or ts > existing["run_config"].get("timestamp_utc", ""):
            grouped[device_group][model_key][dtype] = doc
    return grouped


def _annotate_bars(ax, bars, fmt: str = "{:.0f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _panel_preprocessing(ax, aggregate: dict, quality: dict) -> list:
    labels = ["Preprocessing", "Total Processing", "TTFT"]
    values = [
        aggregate.get("mean_preprocessing_latency_ms") or 0.0,
        aggregate.get("mean_e2e_latency_ms") or 0.0,
        aggregate.get("mean_ttft_ms") or 0.0,
    ]
    bars = ax.bar(labels, values, color=_COLORS[:3])
    _annotate_bars(ax, bars, "{:.0f} ms")
    ax.set_xticklabels([])
    ax.margins(y=0.15)
    return bars


def _panel_phase_breakdown(ax, aggregate: dict, quality: dict) -> list:
    labels = ["System Prompt", "Tools List", "User Query", "Decode"]
    values = [
        aggregate.get("mean_system_prefill_latency_ms") or 0.0,
        aggregate.get("mean_tools_prefill_latency_ms") or 0.0,
        aggregate.get("mean_query_prefill_latency_ms") or 0.0,
        aggregate.get("mean_decode_latency_ms") or 0.0,
    ]
    bars = ax.bar(labels, values, color=_COLORS[:4])
    _annotate_bars(ax, bars, "{:.0f} ms")
    ax.set_xticklabels([])
    ax.margins(y=0.15)
    return bars


def _panel_quality_memory(ax, aggregate: dict, quality: dict) -> list:
    labels = ["Accuracy %", "Peak RAM MB", "KV Cache MB", "Peak GPU MB"]
    accuracy_pct = (quality.get("tool_accuracy") or 0.0) * 100
    peak_ram = aggregate.get("peak_ram_mb") or 0.0
    kv_cache_mb = (aggregate.get("mean_kv_cache_kb") or 0.0) / 1024
    peak_gpu = aggregate.get("mean_peak_gpu_mb")
    # Log scale can't render a true 0 (N/A on CPU runs): use a small visible
    # placeholder height so the "N/A" label still shows up in the chart.
    peak_gpu_val = peak_gpu if peak_gpu is not None else 0.15

    values = [accuracy_pct, peak_ram, kv_cache_mb, peak_gpu_val]
    bars = ax.bar(labels, values, color=_COLORS[:4])
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.1, top=max(values) * 3)
    fmts = [
        "{:.1f}%".format(accuracy_pct),
        "{:.0f}".format(peak_ram),
        "{:.1f}".format(kv_cache_mb),
        "N/A" if peak_gpu is None else "{:.0f}".format(peak_gpu),
    ]
    for bar, label in zip(bars, fmts):
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticklabels([])
    return bars


_PANELS = [
    (
        "Preprocessing / Processing / TTFT (ms)",
        ["Preprocessing", "Total Processing", "TTFT"],
        _panel_preprocessing,
    ),
    (
        "Prefill Phase Breakdown (ms)",
        ["System Prompt", "Tools List", "User Query", "Decode"],
        _panel_phase_breakdown,
    ),
    (
        "Quality & Memory (log scale)",
        ["Accuracy %", "Peak RAM MB", "KV Cache MB", "Peak GPU MB"],
        _panel_quality_memory,
    ),
]


def _plot_device_model(
    machine: str,
    device: str,
    model_key: str,
    dtype_docs: dict[str, dict],
    mode: str,
    output_dir: Path,
) -> None:
    available_dtypes = [d for d in DTYPE_ORDER if d in dtype_docs]
    if not available_dtypes:
        print(
            f"[plot] ERROR: no precisions available for "
            f"machine={machine} device={device} model={model_key} mode={mode}. Skipping."
        )
        return

    n_rows = len(available_dtypes)
    fig = plt.figure(figsize=(15, 3.2 * n_rows + 1.4))
    gs = fig.add_gridspec(
        n_rows + 1, 3, height_ratios=[0.45] + [1] * n_rows, hspace=0.55, wspace=0.35
    )

    model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    fig.suptitle(
        f"{model_name} -- machine={machine}, device={device}, mode={mode}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # Dedicated legend row at the top -- one mini-legend per column, shared
    # across all dtype rows below it, so it never overlaps the chart area.
    for col, (title, labels, _) in enumerate(_PANELS):
        legend_ax = fig.add_subplot(gs[0, col])
        legend_ax.axis("off")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=_COLORS[i]) for i in range(len(labels))
        ]
        legend_ax.legend(
            handles,
            labels,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=8,
            title=title,
            title_fontsize=9,
        )

    for row, dtype in enumerate(available_dtypes, start=1):
        doc = dtype_docs[dtype]
        aggregate = doc.get("aggregate", {})
        quality = doc.get("quality", {})

        for col, (_, _, panel_fn) in enumerate(_PANELS):
            ax = fig.add_subplot(gs[row, col])
            panel_fn(ax, aggregate, quality)
            if col == 0:
                ax.set_ylabel(
                    DTYPE_LABELS.get(dtype, dtype), fontsize=12, fontweight="bold"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_key}_{machine}_{device}_{mode}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Wrote {out_path}")


def main() -> None:
    args = parse_args()

    reports = _load_reports(args.results_dir, args.mode)
    if not reports:
        print(
            f"[plot] ERROR: no reports found for mode={args.mode} in "
            f"{args.results_dir}."
        )
        return

    grouped = _group_reports(reports)

    for (machine, device), by_model in grouped.items():
        for model_key in MODEL_DISPLAY_NAMES:
            dtype_docs = by_model.get(model_key, {})
            _plot_device_model(
                machine, device, model_key, dtype_docs, args.mode, args.output_dir
            )


if __name__ == "__main__":
    main()
