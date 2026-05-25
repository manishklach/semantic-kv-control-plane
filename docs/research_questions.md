# Research Questions

`semantic-kv-control-plane` is a synthetic simulation platform for policy
research. It is designed to explore what happens when KV cache becomes a
distributed systems problem.

## What this repo now explores

1. **KV cache as distributed state**
   - When does KV stop looking like per-request runtime state and start looking
     like shared cluster state?
2. **Why page granularity is insufficient**
   - Runtime pages are useful, but they do not capture reuse fanout, decode
     criticality, recompute worthiness, or movement cost.
3. **Why CXL alone is insufficient**
   - CXL exposes capacity, but not the policy intent needed to decide which KV
     should move or stay hot.
4. **Why semantic movement matters**
   - Movement cost depends on who will consume the KV next, where they are, and
     whether the bytes are hot, cold, or shared.
5. **Why multicast matters**
   - Shared prefixes become a delivery problem as much as a storage problem.
6. **Why attention-aware eviction matters**
   - Some KV is cold because it is old. Other KV is cold because it is rarely
     attended and cheap to recompute.
7. **Why active decode windows change everything**
   - A distributed policy still needs a decode-hot HBM working set floor.
8. **Why memory orchestration may matter more than FLOPS**
   - In some workloads, movement and residency pressure dominate arithmetic.

## What this repo proves

- That adding synthetic metadata can change placement and eviction behavior in
  measurable ways under controlled assumptions.
- That active HBM floors and decode-hot reservations reduce implausible
  “everything moved out of HBM” outcomes.
- That rack-scale prefix reuse, multicast, and topology-aware placement can be
  compared as policy choices rather than hand-wavy ideas.

## What this repo does not prove

- Real production latency
- Real GPU throughput
- Real vLLM or TensorRT-LLM behavior
- Real CXL or NVLink hardware performance
- Real quality impact from compression or recompute

## Open Problems

- How should production runtimes expose KV heat, attention, and decode-window
  metadata?
- When is remote KV movement more expensive than recomputation?
- How should prefix caches be partitioned across tenants?
- What is the right control plane for rack-scale KV multicast?
- Which parts of this policy stack belong in software versus NIC/DPU/FPGA/ASIC
  offload?

## Future ASIC / DPU / CXL Implications

- **ASICs** may want KV-local scheduling signals rather than blind DMA.
- **DPUs/NICs** may be able to own multicast trees and remote-KV fetch
  scheduling.
- **CXL devices** add capacity, but still need orchestration that understands
  decode windows, heat, and reuse intent.
- **KV appliances** become more plausible when they expose prefix fanout,
  attention-aware retention, and locality-aware delivery semantics.
