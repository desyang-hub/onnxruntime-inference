#!/usr/bin/env python3
"""
Benchmark result visualization script.

Reads all benchmark_*.json files from the benchmark output directory,
aggregates results across different concurrent client counts, and plots:

  1. Latency vs Concurrent Clients  (P50, P90, P95, P99, avg)
  2. Throughput vs Concurrent Clients
  3. Latency percentiles heatmap table

Each scheduler mode (Sync / Async / Batch) is plotted with a distinct
color palette.  Lines are smoothed with a light spline curve.

Usage:
    python scripts/plot_benchmark.py                          # default output dir
    python scripts/plot_benchmark.py -d ./benchmark_output
    python scripts/plot_benchmark.py -d ./benchmark_output -o ./reports
    python scripts/plot_benchmark.py -d ./benchmark_output --latency p95 p99
    python scripts/plot_benchmark.py -d ./benchmark_output --format png
    python scripts/plot_benchmark.py -d ./benchmark_output --dpi 300

Dependencies:
    pip install matplotlib numpy
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Color / style palette
# ---------------------------------------------------------------------------

STYLE = {
    "Sync":  {"color": "#1f77b4", "marker": "o",  "label": "Sync"},
    "Async": {"color": "#ff7f0e", "marker": "s",  "label": "Async"},
    "Batch": {"color": "#2ca02c", "marker": "^",  "label": "Batch"},
    "dpi":   150,
    "figsize": (12, 7),
}

LATENCY_FIELDS = {
    "latency_avg_ms":  {"label": "Avg",   "color": "#666666"},
    "latency_p50_ms":  {"label": "P50",   "color": "#1f77b4"},
    "latency_p90_ms":  {"label": "P90",   "color": "#ff7f0e"},
    "latency_p95_ms":  {"label": "P95",   "color": "#d62728"},
    "latency_p99_ms":  {"label": "P99",   "color": "#9467bd"},
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_files(directory: str) -> list[dict]:
    """Load all benchmark_*.json files from *directory*."""
    pattern = os.path.join(directory, "benchmark_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No benchmark_*.json files found in {directory}")
    data = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
    return data


def aggregate_by_clients(data_list: list[dict]) -> dict:
    """
    Aggregate results keyed by (concurrent_clients, mode).

    Returns
    -------
    {
        (clients, mode): {
            "latency_avg_ms": [],
            "latency_p50_ms": [],
            ...
            "throughput_qps": [],
            "override_batch": [],
        }
    }
    """
    agg: dict = {}
    for entry in data_list:
        clients = entry["benchmark"]["concurrent_clients"]
        for res in entry["results"]:
            mode = res["mode"]
            key = (clients, mode)
            if key not in agg:
                agg[key] = {
                    "latency_avg_ms": [],
                    "latency_min_ms": [],
                    "latency_max_ms": [],
                    "latency_p50_ms": [],
                    "latency_p90_ms": [],
                    "latency_p95_ms": [],
                    "latency_p99_ms": [],
                    "throughput_qps": [],
                    "total_requests": [],
                    "samples_count": [],
                    "override_batch": [],
                }
            for field in agg[key]:
                agg[key][field].append(res.get(field, 0))
    return agg


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def smooth_line(x, y, sigma=0.3):
    """Return a smoothly interpolated (x_smooth, y_smooth) for markers."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return x, y
    # deduplicate by averaging duplicates, then smooth
    xs = np.sort(np.unique(x))
    ys = np.array([np.mean(y[x == v]) for v in xs])
    # simple moving-average smooth
    win = max(1, len(ys) // 3 + 1)
    ys_smooth = uniform_filter1d(ys, size=win)
    return xs, ys_smooth


def annotate_points(ax, x, y, xytext=(5, 5)):
    """Add data-label annotations at each point."""
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=xytext, fontsize=6, alpha=0.7)


# ---------------------------------------------------------------------------
# Latency chart
# ---------------------------------------------------------------------------

def plot_latency(agg: dict, all_modes: list[str], latency_keys: list[str],
                 output_path: str):
    """Latency (ms) vs concurrent clients — one subplot per percentile."""
    all_clients = sorted({k[0] for k in agg})
    if not all_clients:
        return

    n = len(latency_keys)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 10),
                             sharex=True)
    if n == 1:
        axes = [axes]
    elif n < 3:
        axes = axes.flatten()[:n]
    else:
        axes = axes.flatten()

    for idx, field in enumerate(latency_keys):
        ax = axes[idx]
        info = LATENCY_FIELDS.get(field, {"label": field, "color": "#333"})
        ax.set_title(f"{info['label']} Latency", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latency (ms)")

        for mode in all_modes:
            ys = []
            for c in all_clients:
                key = (c, mode)
                if key in agg and len(agg[key][field]):
                    ys.append(np.mean(agg[key][field]))
                else:
                    ys.append(np.nan)
            ys_arr = np.array(ys)
            style = STYLE.get(mode, {"color": "#333", "marker": "o"})
            ax.plot(all_clients, ys_arr, color=style["color"],
                    marker=style["marker"], markersize=5, linewidth=1.5,
                    label=style["label"], zorder=3)
            # smooth overlay
            xs_s, ys_s = smooth_line(all_clients, ys_arr)
            ax.plot(xs_s, ys_s, color=style["color"], linewidth=2.5,
                    alpha=0.4, zorder=1)
            annotate_points(ax, all_clients, ys_arr)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Concurrent Clients")
        ax.set_xticks(all_clients)
        ax.tick_params(axis="x", rotation=45)

    # remove unused subplot slots
    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("Latency vs Concurrent Clients", fontsize=14, fontweight="bold",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Latency chart -> {output_path}")


# ---------------------------------------------------------------------------
# Throughput chart
# ---------------------------------------------------------------------------

def plot_throughput(agg: dict, all_modes: list[str], output_path: str):
    """Throughput (QPS) vs concurrent clients."""
    all_clients = sorted({k[0] for k in agg})
    if not all_clients:
        return

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.set_title("Throughput vs Concurrent Clients", fontsize=14, fontweight="bold")
    ax.set_xlabel("Concurrent Clients")
    ax.set_ylabel("Throughput (QPS)")

    for mode in all_modes:
        ys = []
        for c in all_clients:
            key = (c, mode)
            if key in agg and len(agg[key]["throughput_qps"]):
                ys.append(np.mean(agg[key]["throughput_qps"]))
            else:
                ys.append(np.nan)
        ys_arr = np.array(ys)
        style = STYLE.get(mode, {"color": "#333", "marker": "o"})
        ax.plot(all_clients, ys_arr, color=style["color"],
                marker=style["marker"], markersize=5, linewidth=1.5,
                label=style["label"], zorder=3)
        xs_s, ys_s = smooth_line(all_clients, ys_arr)
        ax.plot(xs_s, ys_s, color=style["color"], linewidth=2.5,
                alpha=0.4, zorder=1)
        annotate_points(ax, all_clients, ys_arr)

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_clients)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Throughput chart -> {output_path}")


# ---------------------------------------------------------------------------
# Combined dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(agg: dict, all_modes: list[str], all_latency: list[str],
                   output_path: str):
    """Single dashboard: latency subplots (top) + throughput (bottom)."""
    all_clients = sorted({k[0] for k in agg})
    if not all_clients:
        return

    n_lat = len(all_latency)
    ncols = max(2, n_lat)
    nrows = (n_lat + ncols - 1) // ncols

    fig_height = 5 * nrows + 5
    fig = plt.figure(figsize=(6 * ncols, fig_height))
    gs = gridspec.GridSpec(nrows + 1, ncols)

    # Latency subplots — top rows
    for idx, field in enumerate(all_latency):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        info = LATENCY_FIELDS.get(field, {"label": field, "color": "#333"})
        ax.set_title(f"{info['label']} Latency (ms)", fontsize=11, fontweight="bold")
        for mode in all_modes:
            ys = []
            for c in all_clients:
                key = (c, mode)
                if key in agg and agg[key][field]:
                    ys.append(np.mean(agg[key][field]))
                else:
                    ys.append(np.nan)
            ys_arr = np.array(ys)
            style = STYLE.get(mode, {"color": "#333", "marker": "o"})
            ax.plot(all_clients, ys_arr, color=style["color"],
                    marker=style["marker"], markersize=4, linewidth=1.2,
                    label=style["label"], zorder=3)
            xs_s, ys_s = smooth_line(all_clients, ys_arr)
            ax.plot(xs_s, ys_s, color=style["color"], linewidth=2, alpha=0.35, zorder=1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Clients")
        ax.set_xticks(all_clients)
        ax.tick_params(axis="x", rotation=45)

    # Remove unused latency subplot slots
    for idx in range(n_lat, ncols * nrows):
        row = idx // ncols
        col = idx % ncols
        try:
            fig.delaxes(fig.axes[row * ncols + col - n_lat])
        except Exception:
            pass

    # Throughput subplot — bottom row, spanning all columns
    ax_tp = fig.add_subplot(gs[-1, :])
    ax_tp.set_title("Throughput (QPS)", fontsize=12, fontweight="bold")
    ax_tp.set_xlabel("Concurrent Clients")
    ax_tp.set_ylabel("QPS")
    for mode in all_modes:
        ys = []
        for c in all_clients:
            key = (c, mode)
            if key in agg and agg[key]["throughput_qps"]:
                ys.append(np.mean(agg[key]["throughput_qps"]))
            else:
                ys.append(np.nan)
        ys_arr = np.array(ys)
        style = STYLE.get(mode, {"color": "#333", "marker": "o"})
        ax_tp.plot(all_clients, ys_arr, color=style["color"],
                   marker=style["marker"], markersize=5, linewidth=1.5,
                   label=style["label"], zorder=3)
        xs_s, ys_s = smooth_line(all_clients, ys_arr)
        ax_tp.plot(xs_s, ys_s, color=style["color"], linewidth=2.5, alpha=0.4, zorder=1)
    ax_tp.legend(fontsize=8)
    ax_tp.grid(True, alpha=0.3)
    ax_tp.set_xticks(all_clients)
    ax_tp.tick_params(axis="x", rotation=45)

    fig.suptitle("Benchmark Dashboard", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Dashboard -> {output_path}")


# ---------------------------------------------------------------------------
# Summary table (console + optional HTML)
# ---------------------------------------------------------------------------

def print_summary(agg: dict, data_list: list[dict]):
    """Print a text summary table to stdout."""
    all_clients = sorted({k[0] for k in agg})
    all_modes = sorted({k[1] for k in agg}, key=lambda m: {"Sync": 0, "Async": 1, "Batch": 2}.get(m, 3))

    print("\n" + "=" * 90)
    print(f"{'Benchmark Summary':^90}")
    print("=" * 90)

    for mode in all_modes:
        print(f"\n--- {mode} (batch={agg.get((all_clients[0], mode), {}).get('override_batch', ['?'])[0] if agg.get((all_clients[0], mode)) else '?'}) ---")
        header = f"{'Clients':>8}  {'Batch':>6}  {'Reqs':>8}  {'Throughput':>11}  {'Avg':>9}  {'P50':>9}  {'P90':>9}  {'P95':>9}  {'P99':>9}"
        print(header)
        print("-" * len(header))
        for c in all_clients:
            key = (c, mode)
            if key not in agg:
                continue
            batch = int(np.mean(agg[key]["override_batch"])) if agg[key]["override_batch"] else 0
            reqs = int(np.mean(agg[key]["total_requests"])) if agg[key]["total_requests"] else 0
            tp = np.mean(agg[key]["throughput_qps"]) if agg[key]["throughput_qps"] else 0
            avg = np.mean(agg[key]["latency_avg_ms"]) if agg[key]["latency_avg_ms"] else 0
            p50 = np.mean(agg[key]["latency_p50_ms"]) if agg[key]["latency_p50_ms"] else 0
            p90 = np.mean(agg[key]["latency_p90_ms"]) if agg[key]["latency_p90_ms"] else 0
            p95 = np.mean(agg[key]["latency_p95_ms"]) if agg[key]["latency_p95_ms"] else 0
            p99 = np.mean(agg[key]["latency_p99_ms"]) if agg[key]["latency_p99_ms"] else 0
            print(f"{c:>8}  {batch:>6}  {reqs:>8}  {tp:>10.1f}  {avg:>8.2f}  {p50:>8.2f}  {p90:>8.2f}  {p95:>8.2f}  {p99:>8.2f}")

    print("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize benchmark_*.json results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-d", "--dir", default="./benchmark_output",
                   help="Directory containing benchmark_*.json files (default: ./benchmark_output)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory for plots (default: same as --dir)")
    p.add_argument("--latency", nargs="+", default=None,
                   choices=list(LATENCY_FIELDS.keys()),
                   help="Which latency fields to plot (default: all)")
    p.add_argument("--format", default="png", choices=["png", "svg", "pdf"],
                   help="Output image format (default: png)")
    p.add_argument("--dpi", type=int, default=150,
                   help="Image DPI (default: 150)")
    p.add_argument("--dashboard-only", action="store_true",
                   help="Only generate the combined dashboard, skip individual charts")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    data_dir = Path(args.dir)
    if not data_dir.is_dir():
        print(f"[ERROR] Directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = f".{args.format}"
    STYLE["dpi"] = args.dpi

    # Load & aggregate
    print(f"[INFO] Loading benchmark JSON files from {data_dir} ...")
    data_list = load_json_files(str(data_dir))
    if not data_list:
        print("[ERROR] No valid benchmark JSON files found. Exiting.")
        sys.exit(1)
    print(f"[INFO] Loaded {len(data_list)} file(s).")

    agg = aggregate_by_clients(data_list)
    if not agg:
        print("[ERROR] No result data found. Exiting.")
        sys.exit(1)

    all_modes = sorted({k[1] for k in agg},
                       key=lambda m: {"Sync": 0, "Async": 1, "Batch": 2}.get(m, 3))
    latency_keys = args.latency or list(LATENCY_FIELDS.keys())

    # Console summary
    print_summary(agg, data_list)

    # Plots
    print("[INFO] Generating plots ...")

    if not args.dashboard_only:
        # Individual latency chart
        plot_latency(agg, all_modes, latency_keys,
                     str(output_dir / f"latency{ext}"))
        # Throughput chart
        plot_throughput(agg, all_modes,
                        str(output_dir / f"throughput{ext}"))

    # Dashboard
    plot_dashboard(agg, all_modes, latency_keys,
                   str(output_dir / f"dashboard{ext}"))

    print(f"\n[INFO] Done. Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
