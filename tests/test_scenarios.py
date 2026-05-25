from benchmarks.scenarios import Scenario, run_scenario
from semantic_kv.trace_generators import SharedEnterprisePromptTrace


def test_small_scenario_runs_all_policies():
    trace = SharedEnterprisePromptTrace().generate(
        sessions=3, shared_prefix_tokens=128, unique_tokens_per_session=128, decode_steps=1
    )
    rows = run_scenario(Scenario("tiny", trace, "tiny test scenario"))
    assert len(rows) == 5
    assert any(row["dedup_saved_gb"] > 0 for row in rows)
