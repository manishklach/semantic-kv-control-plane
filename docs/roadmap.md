# Roadmap

## v0.1 Simulator

Build synthetic workloads, memory tiers, placement policies, eviction policies, prefix dedup, prefetch accounting, CLI comparisons, tests, and plots.

## v0.2 vLLM Trace Import

Add an importer for vLLM-style block allocation traces without linking to vLLM as a runtime dependency.

## v0.3 Real Trace Replay

Replay production-like request traces with prompt sharing, tenant IDs, session lifetimes, and token-level access.

## v0.4 LMCache/vLLM Connector Mock

Prototype connector boundaries and metadata exchange without taking a serving dependency.

## v0.5 FPGA/DPU Offload Concept

Sketch which metadata and movement tasks could be offloaded and define simulator hooks for offload latency/throughput assumptions.

## Later Research Directions

- TensorRT-LLM integration experiments
- topology-aware multi-GPU placement
- semantic multicast for shared prefixes
- rack-scale KV fabrics
- KV-aware NIC scheduling concepts
- tenant-aware isolation and fairness policies
- distributed eviction with migration before eviction
- memory-vs-recompute policy experiments
