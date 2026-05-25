"""Metrics and plotting helpers for simulator observability."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class SimulationMetrics:
    total_kv_created_bytes: int = 0
    total_stored_bytes: int = 0
    hbm_used_peak: int = 0
    appliance_used_peak: int = 0
    cxl_used_peak: int = 0
    nvme_used_peak: int = 0
    hbm_miss_count: int = 0
    simulated_stall_us: float = 0.0
    transfer_penalty_us: float = 0.0
    queue_delay_us: float = 0.0
    bandwidth_saturation_events: int = 0
    bytes_moved: int = 0
    bytes_avoided: int = 0
    multicast_saved_bytes: int = 0
    avoided_cross_rack_bytes: int = 0
    movement_energy_j: float = 0.0
    energy_per_token: float = 0.0
    topology_congestion_score: float = 0.0
    dedup_saved_bytes: int = 0
    compression_saved_bytes: int = 0
    eviction_count: int = 0
    prefix_hit_rate: float = 0.0
    prefetch_success_rate: float = 0.0
    estimated_ttft_delta_us: float = 0.0
    estimated_throughput_score: float = 1.0
    occupancy_history: list[dict[str, float]] = field(default_factory=list)
    movement_history: list[dict[str, float]] = field(default_factory=list)
    latency_history: list[dict[str, float]] = field(default_factory=list)
    dedup_history: list[dict[str, float]] = field(default_factory=list)
    eviction_class_counts: dict[str, int] = field(default_factory=dict)

    def as_row(self, policy: str) -> dict[str, float | int | str]:
        row = asdict(self)
        row.pop("occupancy_history", None)
        row.pop("movement_history", None)
        row.pop("latency_history", None)
        row.pop("dedup_history", None)
        row.pop("eviction_class_counts", None)
        row["policy"] = policy
        return row


def bytes_to_gb(value: float) -> float:
    return value / (1024**3)


def write_results_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _use_dark_theme() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1020",
            "axes.facecolor": "#111827",
            "savefig.facecolor": "#0b1020",
            "axes.edgecolor": "#334155",
            "grid.color": "#334155",
            "text.color": "#e5e7eb",
            "axes.labelcolor": "#cbd5e1",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def plot_tier_occupancy(history: list[dict[str, float]], output: Path) -> None:
    if not history:
        return
    _use_dark_theme()
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 5.8))
    for column in ["GPU_HBM", "KV_APPLIANCE", "CXL_POOL", "NVME_OBJECT"]:
        if column in df:
            plt.plot(df["step"], df[column].map(bytes_to_gb), linewidth=2.2, label=column)
    plt.xlabel("Simulation step")
    plt.ylabel("Used capacity (GB)")
    plt.title("Tier Occupancy Over Time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_hbm_pressure(history: list[dict[str, float]], output: Path) -> None:
    if not history:
        return
    _use_dark_theme()
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 4.8))
    plt.fill_between(df["step"], df["GPU_HBM"].map(bytes_to_gb), color="#38bdf8", alpha=0.32)
    plt.plot(df["step"], df["GPU_HBM"].map(bytes_to_gb), color="#38bdf8", linewidth=2.4)
    plt.title("HBM Pressure Over Time")
    plt.xlabel("Simulation step")
    plt.ylabel("HBM used (GB)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_timeseries(history: list[dict[str, float]], column: str, title: str, ylabel: str, output: Path) -> None:
    if not history:
        return
    _use_dark_theme()
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    if column not in df:
        return
    plt.figure(figsize=(10, 4.8))
    plt.plot(df["step"], df[column], color="#a78bfa", linewidth=2.4)
    plt.fill_between(df["step"], df[column], color="#a78bfa", alpha=0.2)
    plt.title(title)
    plt.xlabel("Simulation step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_eviction_classes(counts: dict[str, int], output: Path) -> None:
    _use_dark_theme()
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = list(counts.keys()) or ["none"]
    values = list(counts.values()) or [0]
    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, values, color="#f97316")
    plt.title("Eviction Class Histogram")
    plt.ylabel("Evictions")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def generate_observability_plots(metrics: SimulationMetrics, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        output_dir / "tier_occupancy.png",
        output_dir / "hbm_pressure_over_time.png",
        output_dir / "bytes_moved_over_time.png",
        output_dir / "dedup_savings_over_time.png",
        output_dir / "decode_stall_timeline.png",
        output_dir / "eviction_class_histogram.png",
    ]
    plot_tier_occupancy(metrics.occupancy_history, plots[0])
    plot_hbm_pressure(metrics.occupancy_history, plots[1])
    plot_timeseries(
        metrics.movement_history,
        "bytes_moved_gb",
        "Bytes Moved Over Time",
        "Cumulative bytes moved (GB)",
        plots[2],
    )
    plot_timeseries(
        metrics.dedup_history,
        "dedup_saved_gb",
        "Dedup Savings Over Time",
        "Cumulative dedup savings (GB)",
        plots[3],
    )
    plot_timeseries(
        metrics.latency_history,
        "stall_us",
        "Decode Stall Accumulation",
        "Cumulative stall (us)",
        plots[4],
    )
    plot_eviction_classes(metrics.eviction_class_counts, plots[5])
    return plots


def plot_compare(results_csv: Path, output_dir: Path) -> list[Path]:
    _use_dark_theme()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    saved: list[Path] = []
    plots = [
        ("hbm_used_peak", "Peak HBM Usage", "peak_hbm_comparison.png"),
        ("bytes_moved", "Bytes Moved", "bytes_moved_comparison.png"),
        ("simulated_stall_us", "Simulated Stall", "stall_comparison.png"),
        ("dedup_saved_bytes", "Dedup Savings", "dedup_savings_comparison.png"),
        ("eviction_count", "Evictions", "eviction_count_comparison.png"),
    ]
    for column, title, filename in plots:
        plt.figure(figsize=(9, 5))
        values = df[column].map(bytes_to_gb) if "bytes" in column or "hbm" in column else df[column]
        plt.bar(df["policy"], values, color=["#38bdf8", "#a78bfa", "#22c55e"])
        plt.title(title)
        plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path)
        plt.close()
        saved.append(path)
    return saved
