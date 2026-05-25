# Research Notes: Memory-Orchestrated Inference Infrastructure

## Why Inference Is Becoming Memory-Bound

Long-context serving turns KV cache into one of the dominant memory consumers in LLM inference. Decode may be compute-light compared with prefill, but it repeatedly depends on resident KV state. As concurrency, context length, agent traces, and RAG payloads grow, HBM pressure becomes an infrastructure problem rather than a local allocator detail.

## KV As An Infrastructure Primitive

KV cache has structure: sessions, tenants, token ranges, prefix hashes, fanout, reuse probability, attention value, recompute cost, and decode deadlines. Treating KV as opaque pages loses information that could guide placement and movement.

## Movement Versus Compute Economics

Moving KV through PCIe, CXL, or rack uplinks is not free. A future runtime may sometimes choose to recompute cold KV rather than fetch it from a congested remote tier. The simulator includes a lightweight energy model to explore this tradeoff without pretending to be a power model for real silicon.

## Topology-Aware Serving

Once KV spans multiple GPUs, appliances, CXL pools, and racks, locality matters. The same prefix may be cheap for one GPU and expensive for another. Topology-aware placement tries to keep active decode local, anchor shared prefixes near high-fanout clusters, and avoid cross-rack motion unless reuse justifies it.

## Semantic Memory Orchestration

Semantic orchestration means routing memory based on intent:

- hot active KV stays close to decode
- reusable prefixes are protected and multicast
- low-attention history can be compressed
- ephemeral tool calls can be demoted quickly
- recomputable state may be dropped under pressure

## Why CXL Alone Is Insufficient

CXL can expose more memory, but it does not answer which KV should move, where it should live, or whether it should be recomputed. A KV-aware control plane can use CXL as one tier in a broader intent-aware memory hierarchy.

## Future KV Fabrics

A rack-scale KV fabric could combine GPU HBM, appliance memory, CXL pools, object storage, NIC/DPU metadata offload, and multicast distribution for shared prefixes. The hard questions are routing, congestion, fairness, isolation, and when to avoid movement entirely.

## KV Multicast

If many sessions share the same prefix, naive duplication wastes HBM and fabric bandwidth. Prefix multicast models one canonical copy plus rack-local references or replicated anchors near fanout clusters.

## Memory-Centric Inference

The central question is no longer only "how fast is the model kernel?" It becomes: where is the working set, how much moves, who will reuse it, and what did movement cost?
