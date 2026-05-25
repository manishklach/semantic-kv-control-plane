# Trace Support

`src/semantic_kv/traces.py` defines a runtime-neutral trace envelope for future replay and import flows.

Supported v0.1 trace event types:

- `KV_ALLOC`
- `KV_ACCESS`
- `KV_PREFETCH`
- `KV_EVICT`
- `SESSION_START`
- `SESSION_END`

The simulator still runs on synthetic `WorkloadEvent` objects. The trace layer exists so future importers can translate vLLM block allocation logs, TensorRT-LLM reuse events, or LMCache-style cache events into a common schema without coupling this repo to a production runtime.

Future vLLM trace replay should map:

- block allocation to `KV_ALLOC`
- block table lookup or decode access to `KV_ACCESS`
- swap/prefetch hints to `KV_PREFETCH`
- block free or eviction to `KV_EVICT`
- request lifecycle to `SESSION_START` and `SESSION_END`

This is intentionally simulation-only. No CUDA memory, vLLM runtime objects, or real tensor payloads are imported in v0.1.
