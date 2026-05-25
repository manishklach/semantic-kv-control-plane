"""Tests for approximate structural prefix matching."""

from __future__ import annotations

from typer.testing import CliRunner

import semantic_kv.cli as cli
from semantic_kv.approx_prefix import (
    ApproxPrefixMatcher,
    MinHashPrefixIndex,
    compare_exact_and_approx_prefix_hits,
)
from semantic_kv.models import MODEL_PRESETS
from semantic_kv.workloads import shared_prefix_workload

runner = CliRunner()


def test_minhash_prefix_matcher_returns_similar_prefix() -> None:
    matcher = ApproxPrefixMatcher(threshold=0.7)
    matcher.add("prefix-a", [1, 2, 3, 4, 5, 6])
    match = matcher.match([1, 2, 3, 4, 5, 99])
    assert match is not None
    assert match.prefix_id == "prefix-a"
    assert match.similarity >= 0.7


def test_approximate_matching_improves_over_exact_for_variants() -> None:
    result = compare_exact_and_approx_prefix_hits(
        sessions=100,
        prefix_tokens=128,
        variation_rate=0.05,
        threshold=0.85,
    )
    assert result["approx_hit_rate"] >= result["exact_hit_rate"]
    assert result["approx_hit_count"] >= result["exact_hit_count"]


def test_index_similarity_is_high_for_small_structural_changes() -> None:
    index = MinHashPrefixIndex(num_perm=64)
    score = index.similarity([10, 11, 12, 13, 14], [10, 11, 12, 13, 99])
    assert 0.6 <= score <= 1.0


def test_simulate_cli_supports_approx_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(cli, "PLOTS", tmp_path / "outputs" / "plots")
    result = runner.invoke(
        cli.app,
        [
            "simulate",
            "--workload",
            "shared-prefix",
            "--sessions",
            "8",
            "--context",
            "1024",
            "--decode-steps",
            "4",
            "--policy",
            "semantic",
            "--approx-prefix",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Approximate Prefix Matching" in result.stdout


def test_shared_prefix_workload_remains_hash_based_even_with_variants() -> None:
    profile = MODEL_PRESETS["llama8b"]
    events = shared_prefix_workload(profile, sessions=3, context=512, decode_steps=2)
    prefix_hashes = {
        event.block.prefix_hash
        for event in events
        if event.block is not None and event.block.prefix_hash is not None
    }
    assert "shared-system-prompt-v1" in prefix_hashes
    assert all(isinstance(prefix_hash, str) for prefix_hash in prefix_hashes)
