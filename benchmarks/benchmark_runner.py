"""Benchmark harness for Semantic KV policy comparisons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from semantic_kv.eviction import DistributedSemanticEvictionPolicy, LRUEviction, SemanticEviction
from semantic_kv.metrics import bytes_to_gb, plot_compare, write_results_csv
from semantic_kv.models import MODEL_PRESETS
from semantic_kv.placement import (
    CXLSpillPolicy,
    DistributedSemanticKVPolicy,
    NaiveHBMPolicy,
    PlacementPolicy,
    SemanticKVPolicy,
    TopologyAwareSemanticPolicy,
)
from semantic_kv.simulator import SimulationEngine
from semantic_kv.tiers import default_tier_profiles
from semantic_kv.workloads import make_workload

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkCase:
    """Describe a benchmark workload configuration."""

    workload: str
    sessions: int
    context: int
    decode_steps: int


CASES = [
    BenchmarkCase("shared-prefix", 100, 32768, 128),
    BenchmarkCase("long-context", 8, 65536, 64),
    BenchmarkCase("agentic-workflow", 64, 16384, 96),
    BenchmarkCase("multi-tenant-inference", 64, 8192, 96),
    BenchmarkCase("shared-enterprise-prompt", 128, 16384, 96),
    BenchmarkCase("multi-agent-collaboration", 64, 12288, 96),
]

POLICIES: list[tuple[str, PlacementPolicy, object]] = [
    ("NaiveHBMPolicy", NaiveHBMPolicy(), LRUEviction()),
    ("CXLSpillPolicy", CXLSpillPolicy(), LRUEviction()),
    ("SemanticSingleNode", SemanticKVPolicy(), SemanticEviction()),
    ("TopologyAwareSemantic", TopologyAwareSemanticPolicy(), SemanticEviction()),
    ("DistributedSemanticKV", DistributedSemanticKVPolicy(), DistributedSemanticEvictionPolicy()),
]


def run_benchmarks(output_dir: Path | None = None) -> pd.DataFrame:
    output_dir = output_dir or ROOT / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = MODEL_PRESETS["llama70b-gqa"]
    rows: list[dict[str, float | int | str]] = []
    for case in CASES:
        workload = make_workload(
            case.workload, profile, case.sessions, case.context, case.decode_steps
        )
        for policy_name, placement, eviction in POLICIES:
            metrics = SimulationEngine(
                profile,
                deepcopy(workload),
                placement,
                eviction,
                default_tier_profiles(),
            ).run()
            row = metrics.as_row(policy_name)
            row["workload"] = case.workload
            row["sessions"] = case.sessions
            row["context"] = case.context
            row["decode_steps"] = case.decode_steps
            row["stall_overhead_ms"] = float(row["simulated_stall_us"]) / 1000
            rows.append(row)
    results = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark_results.csv"
    results.to_csv(csv_path, index=False)
    _write_markdown_summary(results, output_dir / "benchmark_summary.md")

    # Plot aggregate policy averages for quick visual inspection.
    aggregate = results.groupby("policy", as_index=False)[
        [
            "hbm_used_peak",
            "bytes_moved",
            "dedup_saved_bytes",
            "compression_saved_bytes",
            "simulated_stall_us",
            "prefetch_success_rate",
            "eviction_count",
            "bytes_avoided",
            "multicast_saved_bytes",
            "avoided_cross_rack_bytes",
            "energy_per_token",
            "topology_congestion_score",
            "estimated_throughput_score",
        ]
    ].mean()
    aggregate_path = output_dir / "benchmark_policy_averages.csv"
    write_results_csv(aggregate.to_dict("records"), aggregate_path)
    plot_compare(aggregate_path, output_dir / "plots")
    _plot_heatmap(results, "hbm_used_peak", output_dir / "plots" / "hbm_pressure_heatmap.png")
    _plot_heatmap(
        results,
        "topology_congestion_score",
        output_dir / "plots" / "topology_congestion_heatmap.png",
    )
    _plot_heatmap(results, "bytes_moved", output_dir / "plots" / "movement_heatmap.png")
    return results


def _write_markdown_summary(results: pd.DataFrame, path: Path) -> None:
    columns = [
        "workload",
        "policy",
        "hbm_used_peak",
        "bytes_moved",
        "dedup_saved_bytes",
        "compression_saved_bytes",
        "simulated_stall_us",
        "prefetch_success_rate",
        "bytes_avoided",
        "multicast_saved_bytes",
        "avoided_cross_rack_bytes",
        "energy_per_token",
        "topology_congestion_score",
        "estimated_throughput_score",
    ]
    pretty = results[columns].copy()
    for column in [
        "hbm_used_peak",
        "bytes_moved",
        "dedup_saved_bytes",
        "compression_saved_bytes",
        "bytes_avoided",
        "multicast_saved_bytes",
        "avoided_cross_rack_bytes",
    ]:
        pretty[column] = pretty[column].map(lambda value: f"{bytes_to_gb(value):.2f} GB")
    pretty["simulated_stall_us"] = pretty["simulated_stall_us"].map(lambda value: f"{value:.0f} us")
    pretty["prefetch_success_rate"] = pretty["prefetch_success_rate"].map(
        lambda value: f"{value:.0%}"
    )
    pretty["energy_per_token"] = pretty["energy_per_token"].map(lambda value: f"{value:.3e} J")
    pretty["topology_congestion_score"] = pretty["topology_congestion_score"].map(
        lambda value: f"{value:.2f}"
    )
    pretty["estimated_throughput_score"] = pretty["estimated_throughput_score"].map(
        lambda value: f"{value:.2f}"
    )
    table = _to_markdown(pretty)
    path.write_text(
        "# Benchmark Summary\n\n"
        "Synthetic simulator results for policy comparison. "
        "These are not hardware measurements.\n\n"
        + table
        + "\n",
        encoding="utf-8",
    )


def _to_markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _plot_heatmap(results: pd.DataFrame, metric: str, path: Path) -> None:
    pivot = results.pivot(index="workload", columns="policy", values=metric)
    values = pivot.to_numpy(dtype=float)
    if "bytes" in metric or "hbm" in metric:
        values = values / (1024**3)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("#090d18")
    ax.set_facecolor("#111827")
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title(metric.replace("_", " ").title())
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax.text(
                x, y, f"{values[y, x]:.1f}", ha="center", va="center", color="#f8fafc", fontsize=8
            )
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    frame = run_benchmarks()
    print(f"Wrote {len(frame)} benchmark rows to {ROOT / 'benchmarks' / 'results'}")
