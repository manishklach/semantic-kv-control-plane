"""CLI integration tests for generated outputs and policy comparisons."""

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

import semantic_kv.cli as cli

runner = CliRunner()


def test_compare_cli_writes_results_and_semantic_beats_cxl(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(cli, "OUTPUTS", output_dir)
    monkeypatch.setattr(cli, "PLOTS", output_dir / "plots")
    result = runner.invoke(
        cli.app,
        [
            "compare",
            "--workload",
            "shared-prefix",
            "--sessions",
            "10",
            "--context",
            "1024",
            "--decode-steps",
            "8",
        ],
    )
    assert result.exit_code == 0, result.stdout
    results_csv = output_dir / "results.csv"
    assert results_csv.exists()
    frame = pd.read_csv(results_csv)
    semantic = frame.loc[
        frame["policy"] == "Semantic KV", "estimated_throughput_score"
    ].iloc[0]
    cxl = frame.loc[
        frame["policy"] == "CXL Spill+LRU", "estimated_throughput_score"
    ].iloc[0]
    assert semantic > cxl
    assert (output_dir / "dashboard.html").exists()


def test_cli_supports_simulate_trace_export_replay_and_plot(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    traces_dir = tmp_path / "traces"
    monkeypatch.setattr(cli, "OUTPUTS", output_dir)
    monkeypatch.setattr(cli, "PLOTS", output_dir / "plots")

    simulate = runner.invoke(
        cli.app,
        [
            "simulate",
            "--workload",
            "basic",
            "--sessions",
            "2",
            "--context",
            "512",
            "--decode-steps",
            "4",
            "--policy",
            "semantic",
        ],
    )
    assert simulate.exit_code == 0, simulate.stdout
    assert (output_dir / "last_run.csv").exists()

    kv_math = runner.invoke(cli.app, ["kv-math", "--model", "llama8b", "--context", "1024"])
    assert kv_math.exit_code == 0
    assert "KV Cache Estimate" in kv_math.stdout

    generate_trace = runner.invoke(
        cli.app,
        [
            "generate-trace",
            "--scenario",
            "shared-enterprise",
            "--sessions",
            "8",
            "--output",
            str(traces_dir),
        ],
    )
    assert generate_trace.exit_code == 0, generate_trace.stdout
    trace_path = traces_dir / "shared_enterprise.jsonl"
    assert trace_path.exists()

    replay = runner.invoke(
        cli.app,
        [
            "replay-trace",
            "--trace",
            str(trace_path),
            "--policy",
            "distributed-semantic",
        ],
    )
    assert replay.exit_code == 0, replay.stdout

    plot = runner.invoke(cli.app, ["plot", "--last-run", str(output_dir / "last_run.csv")])
    assert plot.exit_code == 0, plot.stdout
    assert any((output_dir / "plots").glob("*.png"))

    dashboard = runner.invoke(cli.app, ["dashboard"])
    assert dashboard.exit_code == 0, dashboard.stdout
    assert (output_dir / "dashboard.html").exists()
