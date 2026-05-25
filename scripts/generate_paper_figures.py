"""Generate blog/paper-ready figures from scenario benchmark outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _theme() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#090d18",
            "axes.facecolor": "#111827",
            "savefig.facecolor": "#090d18",
            "axes.edgecolor": "#334155",
            "grid.color": "#334155",
            "font.size": 10,
        }
    )


def generate_figures(results_csv: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    _theme()
    results_csv = results_csv or ROOT / "benchmarks" / "results" / "scenario_results.csv"
    output_dir = output_dir or ROOT / "outputs" / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    outputs = [
        _memory_hierarchy(output_dir / "kv_memory_hierarchy.png"),
        _bar(df, "peak_hbm_gb", "Peak HBM Usage", output_dir / "peak_hbm_usage_comparison.png"),
        _bar(df, "bytes_moved_gb", "Bytes Moved", output_dir / "bytes_moved_comparison.png"),
        _bar(df, "dedup_saved_gb", "Prefix Reuse Savings", output_dir / "prefix_reuse_savings.png"),
        _bar(
            df,
            "cross_rack_traffic_gb",
            "Cross-Rack Traffic",
            output_dir / "cross_rack_traffic_comparison.png",
        ),
        _bar(
            df,
            "eviction_count",
            "Semantic Eviction Breakdown",
            output_dir / "semantic_eviction_breakdown.png",
        ),
        _bar(df, "prefetch_hit_rate", "Prefetch Hit Rate", output_dir / "prefetch_hit_rate.png"),
        _bar(
            df,
            "energy_per_token",
            "Energy Per Token Proxy",
            output_dir / "energy_per_token_comparison.png",
        ),
        _heatmap(df, "topology_congestion_score", output_dir / "topology_congestion_heatmap.png"),
        _summary_dashboard(df, output_dir / "end_to_end_summary_dashboard.png"),
    ]
    return outputs


def _bar(df: pd.DataFrame, metric: str, title: str, path: Path) -> Path:
    pivot = df.pivot(index="scenario", columns="policy", values=metric)
    ax = pivot.plot(kind="bar", figsize=(14, 7), width=0.82)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _heatmap(df: pd.DataFrame, metric: str, path: Path) -> Path:
    pivot = df.pivot(index="scenario", columns="policy", values=metric)
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title(metric.replace("_", " ").title())
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _memory_hierarchy(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    tiers = [
        ("GPU HBM", "active decode KV", "#38bdf8"),
        ("KV Appliance", "rack-local prefixes", "#22c55e"),
        ("CXL Pool", "cold session KV", "#a78bfa"),
        ("Persistent Tier", "object/cold KV", "#f97316"),
    ]
    for idx, (name, subtitle, color) in enumerate(tiers):
        y = 0.82 - idx * 0.2
        ax.add_patch(
            plt.Rectangle((0.18, y), 0.64, 0.11, edgecolor=color, facecolor="#111827", lw=2)
        )
        ax.text(0.22, y + 0.068, name, color="#f8fafc", fontsize=17, weight="bold")
        ax.text(0.22, y + 0.032, subtitle, color="#94a3b8", fontsize=11)
        if idx < len(tiers) - 1:
            ax.arrow(0.5, y - 0.01, 0, -0.055, color="#93c5fd", head_width=0.02, head_length=0.02)
    ax.text(0.08, 0.95, "Semantic KV Memory Hierarchy", color="#f8fafc", fontsize=22, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _summary_dashboard(df: pd.DataFrame, path: Path) -> Path:
    naive = df[df["policy"] == "Naive HBM + LRU"].groupby("scenario").first()
    dist = df[df["policy"] == "Distributed Semantic KV"].groupby("scenario").first()
    hbm_reduction = ((naive["peak_hbm_gb"] - dist["peak_hbm_gb"]) / naive["peak_hbm_gb"]).clip(
        lower=0
    ).mean() * 100
    movement_reduction = (
        (naive["bytes_moved_gb"] - dist["bytes_moved_gb"]) / naive["bytes_moved_gb"]
    ).clip(lower=0).mean() * 100
    dedup = dist["dedup_saved_gb"].sum()
    energy = dist["energy_per_token"].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.text(0.04, 0.88, "Synthetic Simulation Summary", fontsize=24, weight="bold", color="#f8fafc")
    cards = [
        ("Avg HBM reduction", f"{hbm_reduction:.1f}%"),
        ("Avg movement reduction", f"{movement_reduction:.1f}%"),
        ("Total dedup savings", f"{dedup:.1f} GB"),
        ("Mean energy/token proxy", f"{energy:.2e} J"),
    ]
    for i, (label, value) in enumerate(cards):
        x = 0.05 + (i % 2) * 0.45
        y = 0.56 - (i // 2) * 0.28
        ax.add_patch(
            plt.Rectangle((x, y), 0.38, 0.18, facecolor="#111827", edgecolor="#334155", lw=1.5)
        )
        ax.text(x + 0.03, y + 0.11, value, fontsize=22, weight="bold", color="#93c5fd")
        ax.text(x + 0.03, y + 0.055, label, fontsize=12, color="#cbd5e1")
    ax.text(
        0.05,
        0.08,
        "All numbers are synthetic simulation outputs, not real hardware measurements.",
        color="#94a3b8",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


if __name__ == "__main__":
    paths = generate_figures()
    print("Generated figures:")
    for path in paths:
        print(path)
