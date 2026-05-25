from semantic_kv.traces import Trace, TraceEventType, synthetic_trace_from_steps


def test_synthetic_trace_roundtrip(tmp_path):
    trace = synthetic_trace_from_steps("tiny", steps=2, sessions=1)
    path = tmp_path / "trace.jsonl"
    trace.to_jsonl(path)
    loaded = Trace.from_jsonl(path)
    assert loaded.validate() == []
    assert loaded.summary()["events"] > 0
    assert loaded.events[0].event_type is TraceEventType.SESSION_START
    assert any(event.event_type is TraceEventType.KV_ALLOC for event in loaded.events)
