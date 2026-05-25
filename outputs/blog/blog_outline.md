# When KV Cache Becomes a Distributed Systems Problem

## 1. The Bottleneck Shift

Inference serving is increasingly constrained by memory residency and movement,
not only by math throughput.

## 2. Why KV Cache Stops Being A Runtime Detail

KV carries session, prefix, tenant, and reuse structure. Treating it as
anonymous memory hides useful intent.

## 3. Why CXL Alone Is Insufficient

CXL exposes capacity. It does not decide which KV should move, stay, compress, or be recomputed.

## 4. Semantic KV Orchestration

Metadata-aware placement, prefix reuse, semantic eviction, and prefetching form
a control-plane layer.

## 5. Rack-Scale Memory Fabric Model

The simulator models HBM, appliance tiers, CXL-like pools, uplinks, and congestion.

## 6. Simulation Setup

Trace replay scenarios are synthetic and reproducible. They are not real GPU benchmarks.

## 7. Results

Report HBM pressure, bytes moved, prefix savings, cross-rack traffic, stall proxy, and energy proxy.

## 8. What This Does Not Solve

No CUDA kernels, no real vLLM integration, no measured network performance, no quality validation.

## 9. Open Questions

What metadata should real runtimes expose? When is recompute cheaper than
movement? Where should prefix caches live?

## 10. Future Work

Trace import, connector mocks, topology calibration, DPU/NIC offload
simulation, and rack-scale policy search.
