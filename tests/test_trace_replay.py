from semantic_kv.trace_generators import SharedEnterprisePromptTrace
from semantic_kv.traces import TraceReplayEngine


def test_trace_replay_produces_metrics():
    trace = SharedEnterprisePromptTrace().generate(
        sessions=4, shared_prefix_tokens=256, unique_tokens_per_session=128, decode_steps=2
    )
    label, metrics = TraceReplayEngine(trace).replay("distributed-semantic")
    assert "Distributed" in label
    assert metrics.total_kv_created_bytes > 0
    assert metrics.hbm_used_peak <= metrics.total_kv_created_bytes
