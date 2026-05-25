"""Named reproducible benchmark scenarios for trace replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from semantic_kv.analysis import ResultInterpreter
from semantic_kv.metrics import bytes_to_gb
from semantic_kv.trace_generators import (
    AgenticLoopTrace,
    LongContextTrace,
    MixedProductionTrace,
    MultiTenantRackTrace,
    SharedEnterprisePromptTrace,
)
from semantic_kv.traces import POLICY_MATRIX, Trace, TraceReplayEngine


@dataclass(frozen=True)
class Scenario:
    name: str
    trace: Trace
    description: str


def build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            "scenario_01_shared_prefix_100_sessions",
            SharedEnterprisePromptTrace().generate(
                sessions=100,
                shared_prefix_tokens=8192,
                unique_tokens_per_session=1024,
                decode_steps=64,
            ),
            "100 sessions share a large enterprise policy/system prompt.",
        ),
        Scenario(
            "scenario_02_shared_prefix_1000_sessions",
            SharedEnterprisePromptTrace().generate(
                sessions=1000,
                shared_prefix_tokens=4096,
                unique_tokens_per_session=512,
                decode_steps=24,
            ),
            "1000 sessions share an enterprise prefix; fanout dominates.",
        ),
        Scenario(
            "scenario_03_agentic_tool_loop",
            AgenticLoopTrace().generate(
                sessions=64, tools_per_session=4, reflection_steps=8, decode_steps=64
            ),
            "Agentic tool loops mix ephemeral tool KV and persistent memory.",
        ),
        Scenario(
            "scenario_04_long_context_128k",
            LongContextTrace().generate(sessions=4, context_tokens=131072, decode_steps=64),
            "Few very long-context sessions stress HBM residency.",
        ),
        Scenario(
            "scenario_05_multi_tenant_rack",
            MultiTenantRackTrace().generate(
                tenants=8, sessions_per_tenant=32, shared_prefix_probability=0.7, decode_steps=48
            ),
            "Multi-tenant rack-scale serving with isolated tenant prefixes.",
        ),
        Scenario(
            "scenario_06_mixed_production",
            MixedProductionTrace().generate(),
            "Mixed enterprise, agentic, and long-context pressure.",
        ),
    ]


def run_scenario(scenario: Scenario) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replay = TraceReplayEngine(scenario.trace)
    for policy in POLICY_MATRIX:
        label, metrics = replay.replay(policy)
        row = {
            "scenario": scenario.name,
            "policy": label,
            "peak_hbm_gb": bytes_to_gb(metrics.hbm_used_peak),
            "peak_appliance_gb": bytes_to_gb(metrics.appliance_used_peak),
            "peak_cxl_gb": bytes_to_gb(metrics.cxl_used_peak),
            "total_stored_gb": bytes_to_gb(metrics.total_stored_bytes),
            "bytes_moved_gb": bytes_to_gb(metrics.bytes_moved),
            "avoided_movement_gb": bytes_to_gb(metrics.bytes_avoided),
            "dedup_saved_gb": bytes_to_gb(metrics.dedup_saved_bytes),
            "compression_saved_gb": bytes_to_gb(metrics.compression_saved_bytes),
            "multicast_saved_gb": bytes_to_gb(metrics.multicast_saved_bytes),
            "cross_rack_traffic_gb": max(
                0.0, bytes_to_gb(metrics.bytes_moved - metrics.avoided_cross_rack_bytes)
            ),
            "avoided_cross_rack_gb": bytes_to_gb(metrics.avoided_cross_rack_bytes),
            "prefetch_hit_rate": metrics.prefetch_success_rate,
            "prefix_hit_rate": metrics.prefix_hit_rate,
            "eviction_count": metrics.eviction_count,
            "late_prefetch_count": 0,
            "simulated_stall_ms": metrics.simulated_stall_us / 1000,
            "simulated_energy_j": metrics.movement_energy_j,
            "energy_per_token": metrics.energy_per_token,
            "throughput_score": metrics.estimated_throughput_score,
            "topology_congestion_score": metrics.topology_congestion_score,
        }
        rows.append(row)
    return rows


def run_all_scenarios(output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = Path("examples/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scenario in build_scenarios():
        scenario.trace.to_jsonl(trace_dir / f"{scenario.name}.jsonl")
        (trace_dir / f"{scenario.name}.md").write_text(
            f"# {scenario.name}\n\n{scenario.description}\n\n"
            f"{scenario.trace.description}\n\n"
            "All results are synthetic simulation inputs, not production traces.\n",
            encoding="utf-8",
        )
        rows.extend(run_scenario(scenario))
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "scenario_results.csv", index=False)
    (output_dir / "scenario_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "scenario_summary.md").write_text(_summary_markdown(frame), encoding="utf-8")
    (output_dir / "scenario_findings.md").write_text(
        ResultInterpreter(frame).to_markdown(), encoding="utf-8"
    )
    _plot_scenario_bars(frame, output_dir / "plots")
    return frame


def _summary_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in [col for col in display.columns if col.endswith("_gb")]:
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    display["simulated_stall_ms"] = display["simulated_stall_ms"].map(lambda value: f"{value:.1f}")
    lines = [
        "# Scenario Summary",
        "",
        "Synthetic simulation results. These are not real GPU or network measurements.",
        "",
        _to_markdown(display),
        "",
    ]
    return "\n".join(lines)


def _to_markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _plot_scenario_bars(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    for metric in [
        "peak_hbm_gb",
        "bytes_moved_gb",
        "dedup_saved_gb",
        "avoided_cross_rack_gb",
        "energy_per_token",
        "topology_congestion_score",
    ]:
        pivot = frame.pivot(index="scenario", columns="policy", values=metric)
        ax = pivot.plot(kind="bar", figsize=(13, 6), width=0.82)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=180)
        plt.close()
