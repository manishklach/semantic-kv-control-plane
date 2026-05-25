# Architecture

Semantic KV Control Plane models KV cache as intent-rich infrastructure instead of anonymous memory pages. The implementation is intentionally simulator-first: policies operate on metadata-rich `KVBlock` objects and simulated memory tiers, not real tensors.

## Tiers

GPU HBM is the active decode tier with the lowest simulated latency and tightest capacity. KV appliance memory represents a dedicated service tier for reusable prefixes and recent session state. CXL pool memory represents large pooled expansion memory with higher latency. NVMe object storage represents a persistent tier for cold or cheap-to-recompute KV.

![KV memory hierarchy](diagrams/kv_memory_hierarchy.svg)

## Metadata Directory

The metadata directory tracks prefix hashes, canonical KV blocks, attached sessions, fanout, and reuse savings. This lets a shared system prompt be modeled as one canonical set of blocks plus references.

![Prefix reuse flow](diagrams/prefix_reuse_flow.svg)

## Prefix Dedup

Prefix dedup is exact-match only in v0.1. The simulator stores canonical prefix KV once, then turns later matching blocks into `DEDUP_REF` entries with a small simulated storage footprint.

## Prefetch Scheduler

The prefetch scheduler predicts a simple next-token window and tracks requests, hits, late prefetches, success rate, and avoided stall time.

## Semantic Eviction

Semantic eviction scores blocks using eviction class, reuse score, fanout, prefix presence, last access, recompute cost, and tier. It protects hot active and reusable prefix blocks while preferring ephemeral and cheap-to-recompute blocks as victims.

![Semantic eviction flow](diagrams/semantic_eviction_flow.svg)

## Control Plane And Data Plane

The control plane chooses where KV should live and why. The data plane simulation records bytes moved, stall accumulation, bandwidth saturation events, and occupancy over time.

![KV control plane](diagrams/kv_control_plane.svg)

![KV data plane](diagrams/kv_data_plane.svg)

## Rack-Scale Fabric

The rack-scale model adds GPU nodes, appliance affinity, rack IDs, CXL pools, uplinks, cross-rack links, congestion penalties, and topology-aware placement decisions.

![Rack-scale KV fabric](diagrams/rack_scale_kv_fabric.svg)

![Topology-aware placement](diagrams/topology_aware_placement.svg)

## Movement Pipeline

Movement accounting tracks bytes moved, avoided movement, multicast savings, avoided cross-rack traffic, and an illustrative movement-energy estimate.

![KV movement pipeline](diagrams/kv_movement_pipeline.svg)
