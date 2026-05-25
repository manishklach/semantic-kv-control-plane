"""Tests for generators, plotting helpers, dashboards, and topology visuals."""

from __future__ import annotations

import pandas as pd

import scripts.generate_blog_assets as blog_assets
import scripts.generate_paper_figures as paper_figures
from semantic_kv.dashboard import generate_static_dashboard
from semantic_kv.metrics import (
    SimulationMetrics,
    generate_observability_plots,
    plot_compare,
)
from semantic_kv.models import MODEL_PRESETS
from semantic_kv.topology import TopologyVisualizer, default_rack_topology
from semantic_kv.trace_generators import (
    AgenticLoopTrace,
    LongContextTrace,
    MixedProductionTrace,
    MultiTenantRackTrace,
    SharedEnterprisePromptTrace,
    export_sample_traces,
)
from semantic_kv.workloads import make_workload


def _sample_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "scenario_a",
                "policy": "Naive HBM + LRU",
                "peak_hbm_gb": 8.0,
                "peak_appliance_gb": 1.0,
                "peak_cxl_gb": 2.0,
                "total_stored_gb": 9.0,
                "bytes_moved_gb": 7.0,
                "avoided_movement_gb": 0.5,
                "dedup_saved_gb": 0.0,
                "compression_saved_gb": 0.0,
                "multicast_saved_gb": 0.0,
                "cross_rack_traffic_gb": 3.0,
                "avoided_cross_rack_gb": 0.0,
                "prefetch_hit_rate": 0.10,
                "prefix_hit_rate": 0.0,
                "eviction_count": 12,
                "late_prefetch_count": 4,
                "simulated_stall_ms": 120.0,
                "stall_p99_ms": 160.0,
                "stall_p999_ms": 240.0,
                "simulated_energy_j": 1.8,
                "energy_per_token": 0.002,
                "active_hbm_gb": 2.2,
                "throughput_score": 0.9,
                "topology_congestion_score": 0.3,
            },
            {
                "scenario": "scenario_a",
                "policy": "Distributed Semantic KV",
                "peak_hbm_gb": 4.0,
                "peak_appliance_gb": 3.0,
                "peak_cxl_gb": 1.0,
                "total_stored_gb": 5.5,
                "bytes_moved_gb": 3.0,
                "avoided_movement_gb": 2.0,
                "dedup_saved_gb": 2.5,
                "compression_saved_gb": 0.8,
                "multicast_saved_gb": 1.2,
                "cross_rack_traffic_gb": 1.0,
                "avoided_cross_rack_gb": 2.0,
                "prefetch_hit_rate": 0.70,
                "prefix_hit_rate": 0.8,
                "eviction_count": 4,
                "late_prefetch_count": 1,
                "simulated_stall_ms": 40.0,
                "stall_p99_ms": 55.0,
                "stall_p999_ms": 80.0,
                "simulated_energy_j": 0.9,
                "energy_per_token": 0.001,
                "active_hbm_gb": 1.4,
                "throughput_score": 1.6,
                "topology_congestion_score": 0.1,
            },
        ]
    )


def _sample_compare_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy": "Naive HBM + LRU",
                "hbm_used_peak": 8 * 1024**3,
                "bytes_moved": 7 * 1024**3,
                "simulated_stall_us": 120_000.0,
                "stall_p99_us": 150_000.0,
                "dedup_saved_bytes": 0,
                "eviction_count": 12,
            },
            {
                "policy": "Distributed Semantic KV",
                "hbm_used_peak": 4 * 1024**3,
                "bytes_moved": 3 * 1024**3,
                "simulated_stall_us": 40_000.0,
                "stall_p99_us": 55_000.0,
                "dedup_saved_bytes": int(2.5 * 1024**3),
                "eviction_count": 4,
            },
        ]
    )


def test_trace_generators_and_sample_export(tmp_path) -> None:
    traces = [
        SharedEnterprisePromptTrace().generate(
            sessions=4,
            shared_prefix_tokens=256,
            unique_tokens_per_session=128,
            decode_steps=2,
        ),
        AgenticLoopTrace().generate(sessions=2, tools_per_session=1, decode_steps=2),
        LongContextTrace().generate(sessions=1, context_tokens=1024, decode_steps=2),
        MultiTenantRackTrace().generate(tenants=2, sessions_per_tenant=2, decode_steps=2),
        MixedProductionTrace().generate(),
    ]
    assert all(trace.events for trace in traces)
    exported = export_sample_traces(tmp_path / "traces")
    assert exported
    assert any(path.suffix == ".jsonl" for path in exported)
    assert any(path.suffix == ".md" for path in exported)


def test_metrics_plots_dashboard_and_blog_assets(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    plot_dir = output_dir / "plots"
    scenario_results_csv = tmp_path / "scenario_results.csv"
    compare_results_csv = tmp_path / "compare_results.csv"
    _sample_results_frame().to_csv(scenario_results_csv, index=False)
    _sample_compare_frame().to_csv(compare_results_csv, index=False)

    metrics = SimulationMetrics(
        occupancy_history=[
            {
                "step": 0,
                "GPU_HBM": 100,
                "KV_APPLIANCE": 50,
                "CXL_POOL": 25,
                "NVME_OBJECT": 10,
            },
            {
                "step": 1,
                "GPU_HBM": 120,
                "KV_APPLIANCE": 60,
                "CXL_POOL": 30,
                "NVME_OBJECT": 12,
            },
        ],
        movement_history=[{"step": 0, "bytes_moved_gb": 1.0}, {"step": 1, "bytes_moved_gb": 1.5}],
        latency_history=[{"step": 0, "stall_us": 10.0}, {"step": 1, "stall_us": 15.0}],
        dedup_history=[{"step": 0, "dedup_saved_gb": 0.2}, {"step": 1, "dedup_saved_gb": 0.4}],
        heat_history=[{"step": 0, "avg_heat": 0.6}, {"step": 1, "avg_heat": 0.7}],
        active_hbm_history=[
            {"step": 0, "active_hbm_gb": 0.1, "hbm_floor_gb": 0.2},
            {"step": 1, "active_hbm_gb": 0.15, "hbm_floor_gb": 0.2},
        ],
        eviction_class_counts={"SESSION_COLD": 2, "LOW_ATTENTION": 1},
    )
    observability = generate_observability_plots(metrics, plot_dir)
    compare_plots = plot_compare(compare_results_csv, plot_dir)
    assert observability
    assert compare_plots

    dashboard_path = generate_static_dashboard(
        scenario_results_csv, plot_dir, output_dir / "dashboard.html"
    )
    assert dashboard_path.exists()
    assert "Executive Summary" in dashboard_path.read_text(encoding="utf-8")

    figures = paper_figures.generate_figures(scenario_results_csv, output_dir / "paper_figures")
    assert len(figures) == 14
    assert all(path.exists() for path in figures)

    topology_svg = TopologyVisualizer(default_rack_topology()).render_svg(
        output_dir / "topology.svg"
    )
    assert topology_svg.exists()

    benchmark_results = tmp_path / "benchmarks" / "results"
    benchmark_results.mkdir(parents=True, exist_ok=True)
    (_sample_results_frame()).to_csv(benchmark_results / "scenario_results.csv", index=False)
    original_blog_root = blog_assets.ROOT
    original_fig_root = paper_figures.ROOT
    blog_assets.ROOT = tmp_path
    paper_figures.ROOT = tmp_path
    try:
        blog_dir = blog_assets.generate_blog_assets(output_dir / "blog")
    finally:
        blog_assets.ROOT = original_blog_root
        paper_figures.ROOT = original_fig_root
    assert (blog_dir / "blog_outline.md").exists()
    assert (blog_dir / "social_thread.md").exists()


def test_workload_factory_covers_named_workloads() -> None:
    profile = MODEL_PRESETS["llama8b"]
    names = [
        "basic",
        "shared-prefix",
        "agentic-tool",
        "long-context",
        "mixed-tenant",
        "agentic-workflow",
        "multi-tenant-inference",
        "shared-enterprise-prompt",
        "multi-agent-collaboration",
    ]
    for name in names:
        events = make_workload(name, profile, sessions=2, context=512, decode_steps=4)
        assert events
