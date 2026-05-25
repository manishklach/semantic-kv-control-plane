# Design Notes

## Why Generic LRU Is Insufficient

LRU observes recency but not intent. A shared system prompt that has not been touched recently can still be valuable because future sessions may reuse it. An ephemeral tool-call block may be recent but safe to discard. Semantic classes make these differences visible.

In the code, semantic eviction scores class, reuse score, fanout, prefix presence, last access, recompute cost, and tier. This is not meant to be a final scoring function; it is a research knob for asking which metadata dimensions are useful.

## Why CXL Alone Is Not Enough

CXL exposes capacity, but capacity alone does not decide what should move, when to prefetch it, or what should be protected. The control plane layer modeled here uses KV-specific metadata to steer tiering decisions.

## Relationship To PagedAttention

PagedAttention gives serving systems an efficient block abstraction for KV cache. This simulator assumes a block abstraction exists and asks what a higher-level policy plane could do with semantic metadata around those blocks.

## Hardware Mapping Later

A future DPU, FPGA, or ASIC concept could offload metadata lookups, prefix reference tracking, compression accounting, or movement scheduling. v0.1 intentionally stays in Python simulation so policy behavior is easy to inspect.

## Latency Model

The simulator includes tier latency penalties, topology-aware transfer penalties, queue-depth effects, bandwidth saturation counters, and decode stall accumulation. These are deliberately simple models: useful for comparative policy experiments, not cycle-level performance prediction.
