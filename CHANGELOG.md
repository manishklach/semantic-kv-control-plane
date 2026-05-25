# Changelog

## v0.1.0 - 2026-05-25

Initial public research prototype release.

### Added

- Semantic KV block model with tiered placement across GPU HBM, KV appliance,
  CXL pool, and persistent storage
- Placement policies for naive HBM spill, CXL spill, metadata-aware single-node
  KV, topology-aware KV, and distributed semantic KV
- LRU, semantic, and distributed semantic eviction policies
- Exact prefix reuse directories plus approximate structural prefix matching with
  MinHash
- Synthetic workloads, synthetic traces, and trace replay support
- Benchmark harness, scenario suite, paper/blog figure generation, and static
  dashboard output
- Rack-scale topology model, fabric congestion model, movement accounting, and
  energy-per-token proxy
- CLI, notebook quickstart, contributor documentation, and issue templates

### Notes

- All results are synthetic simulation outputs under workload assumptions.
- No CUDA kernels, no real vLLM integration, and no hardware benchmark claims.
