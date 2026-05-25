"""Command line interface for the Semantic KV Control Plane simulator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from semantic_kv.dashboard import generate_static_dashboard
from semantic_kv.metrics import (
    bytes_to_gb,
    generate_observability_plots,
    plot_compare,
    plot_tier_occupancy,
    write_results_csv,
)
from semantic_kv.models import MODEL_PRESETS
from semantic_kv.placement import CXLSpillPolicy, NaiveHBMPolicy, SemanticKVPolicy, make_policy
from semantic_kv.simulator import SimulationEngine, make_eviction_policy
from semantic_kv.tiers import default_tier_profiles
from semantic_kv.trace_generators import export_sample_traces, generate_named_trace
from semantic_kv.traces import Trace, TraceReplayEngine
from semantic_kv.workloads import make_workload

app = typer.Typer(help="Semantic KV Control Plane simulator")
console = Console()
ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
PLOTS = OUTPUTS / "plots"


def _run(policy_name: str, workload_name: str, sessions: int, context: int, decode_steps: int):
    profile = MODEL_PRESETS["llama70b-gqa"]
    workload = make_workload(workload_name, profile, sessions, context, decode_steps)
    placement = make_policy(policy_name)
    eviction_name = "semantic" if isinstance(placement, SemanticKVPolicy) else "lru"
    engine = SimulationEngine(
        model_profile=profile,
        workload=workload,
        placement_policy=placement,
        eviction_policy=make_eviction_policy(eviction_name),
        tier_config=default_tier_profiles(),
    )
    return placement.name, engine.run()


@app.command()
def simulate(
    workload: str = typer.Option("shared-prefix"),
    sessions: int = typer.Option(100),
    context: int = typer.Option(32768),
    decode_steps: int = typer.Option(128),
    policy: str = typer.Option("semantic"),
) -> None:
    """Run one policy on one synthetic workload."""

    policy_name, metrics = _run(policy, workload, sessions, context, decode_steps)
    OUTPUTS.mkdir(exist_ok=True)
    generate_observability_plots(metrics, PLOTS)
    plot_tier_occupancy(metrics.occupancy_history, PLOTS / f"{policy_name}_tier_occupancy.png")
    write_results_csv([metrics.as_row(policy_name)], OUTPUTS / "last_run.csv")
    table = _comparison_table()
    _add_row(table, policy_name, metrics)
    console.print(table)
    console.print(f"Saved results to {OUTPUTS / 'last_run.csv'}")
    console.print(f"Saved plots to {PLOTS}")


@app.command()
def compare(
    workload: str = typer.Option("shared-prefix"),
    sessions: int = typer.Option(100),
    context: int = typer.Option(32768),
    decode_steps: int = typer.Option(128),
) -> None:
    """Compare naive HBM, CXL spill, and semantic KV policies."""

    runs = [
        ("Naive HBM+LRU", NaiveHBMPolicy(), "lru"),
        ("CXL Spill+LRU", CXLSpillPolicy(), "lru"),
        ("Semantic KV", SemanticKVPolicy(), "semantic"),
    ]
    profile = MODEL_PRESETS["llama70b-gqa"]
    workload_events = make_workload(workload, profile, sessions, context, decode_steps)
    rows = []
    table = _comparison_table()
    semantic_history = []
    semantic_metrics = None
    for label, placement, eviction in runs:
        engine = SimulationEngine(
            profile,
            deepcopy(workload_events),
            placement,
            make_eviction_policy(eviction),
            default_tier_profiles(),
        )
        metrics = engine.run()
        rows.append(metrics.as_row(label))
        _add_row(table, label, metrics)
        if isinstance(placement, SemanticKVPolicy):
            semantic_history = metrics.occupancy_history
            semantic_metrics = metrics
    OUTPUTS.mkdir(exist_ok=True)
    results_path = OUTPUTS / "results.csv"
    write_results_csv(rows, results_path)
    if semantic_metrics is not None:
        generate_observability_plots(semantic_metrics, PLOTS)
    plot_tier_occupancy(semantic_history, PLOTS / "semantic_tier_occupancy.png")
    saved = plot_compare(results_path, PLOTS)
    dashboard_path = generate_static_dashboard(results_path, PLOTS, OUTPUTS / "dashboard.html")
    console.print(table)
    console.print(f"Saved results to {results_path}")
    console.print("Saved plots: " + ", ".join(path.name for path in saved[:5]))
    console.print(f"Saved dashboard to {dashboard_path}")


@app.command("kv-math")
def kv_math(
    model: str = typer.Option("llama70b-gqa"),
    context: int = typer.Option(32768),
    sessions: int = typer.Option(100),
) -> None:
    """Estimate KV bytes for a model/context/session count."""

    profile = MODEL_PRESETS[model]
    per_session = profile.estimate_session_kv_bytes(context)
    table = Table(title="KV Cache Estimate")
    table.add_column("Model")
    table.add_column("Context")
    table.add_column("Sessions")
    table.add_column("Per Session")
    table.add_column("Total")
    table.add_row(
        model,
        str(context),
        str(sessions),
        f"{bytes_to_gb(per_session):.2f} GB",
        f"{bytes_to_gb(per_session * sessions):.2f} GB",
    )
    console.print(table)


@app.command()
def plot(last_run: Path = typer.Option(OUTPUTS / "results.csv")) -> None:
    """Generate comparison plots from a previous results CSV."""

    saved = plot_compare(last_run, PLOTS)
    console.print("Saved plots: " + ", ".join(str(path) for path in saved))


@app.command("generate-trace")
def generate_trace(
    scenario: str = typer.Option("shared-enterprise"),
    sessions: int = typer.Option(1000),
    output: Path = typer.Option(ROOT / "examples" / "traces"),
) -> None:
    """Generate a synthetic JSONL trace."""

    kwargs = (
        {"sessions": sessions}
        if scenario in {"shared-enterprise", "agentic-loop", "long-context"}
        else {}
    )
    trace = generate_named_trace(scenario, **kwargs)
    path = output / f"{trace.workload_name}.jsonl"
    trace.to_jsonl(path)
    console.print(f"Saved trace to {path}")
    console.print(trace.summary())


@app.command("replay-trace")
def replay_trace(
    trace: Path = typer.Option(ROOT / "examples" / "traces" / "shared_enterprise.jsonl"),
    policy: str = typer.Option("distributed-semantic"),
) -> None:
    """Replay a JSONL trace through one policy."""

    loaded = Trace.from_jsonl(trace)
    errors = loaded.validate()
    if errors:
        raise typer.BadParameter("; ".join(errors))
    label, metrics = TraceReplayEngine(loaded).replay(policy)
    table = _comparison_table()
    _add_row(table, label, metrics)
    console.print(table)


@app.command("benchmark-suite")
def benchmark_suite(all: bool = typer.Option(False, "--all")) -> None:
    """Run the trace replay scenario benchmark suite."""

    from benchmarks.scenarios import run_all_scenarios

    output = ROOT / "benchmarks" / "results"
    frame = run_all_scenarios(output)
    console.print(f"Saved {len(frame)} scenario rows to {output}")


@app.command("generate-figures")
def generate_figures_command() -> None:
    """Generate paper/blog-ready figures."""

    from scripts.generate_paper_figures import generate_figures

    paths = generate_figures()
    console.print("Generated figures:")
    for path in paths:
        console.print(f"- {path}")


@app.command("generate-blog-assets")
def generate_blog_assets_command() -> None:
    """Generate blog outline, findings, figures, and social thread draft."""

    from scripts.generate_blog_assets import generate_blog_assets

    console.print(f"Generated blog assets in {generate_blog_assets()}")


@app.command("dashboard")
def dashboard_command() -> None:
    """Generate the static HTML dashboard."""

    scenario_results = ROOT / "benchmarks" / "results" / "scenario_results.csv"
    results = scenario_results if scenario_results.exists() else OUTPUTS / "results.csv"
    plots = OUTPUTS / "paper_figures" if (OUTPUTS / "paper_figures").exists() else PLOTS
    path = generate_static_dashboard(results, plots, OUTPUTS / "dashboard.html")
    console.print(f"Generated dashboard at {path}")


@app.command("export-sample-traces")
def export_sample_traces_command() -> None:
    """Export all small sample traces."""

    paths = export_sample_traces(ROOT / "examples" / "traces")
    console.print(f"Generated {len(paths)} trace files")


def _comparison_table() -> Table:
    table = Table(title="Semantic KV Policy Comparison")
    for column in [
        "Policy",
        "Peak HBM",
        "Stored",
        "Bytes Moved",
        "Stall us",
        "Prefix Hit",
        "Dedup Saved",
        "Compression Saved",
        "Evictions",
        "Throughput",
    ]:
        table.add_column(column)
    return table


def _add_row(table: Table, policy: str, metrics) -> None:
    table.add_row(
        policy,
        f"{bytes_to_gb(metrics.hbm_used_peak):.2f} GB",
        f"{bytes_to_gb(metrics.total_stored_bytes):.2f} GB",
        f"{bytes_to_gb(metrics.bytes_moved):.2f} GB",
        f"{metrics.simulated_stall_us:.0f}",
        f"{metrics.prefix_hit_rate:.0%}",
        f"{bytes_to_gb(metrics.dedup_saved_bytes):.2f} GB",
        f"{bytes_to_gb(metrics.compression_saved_bytes):.2f} GB",
        str(metrics.eviction_count),
        f"{metrics.estimated_throughput_score:.2f}",
    )


if __name__ == "__main__":
    app()
