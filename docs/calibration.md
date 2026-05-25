# Calibration Notes

This repository is a **synthetic simulation** and **architectural exploration**.
Its numbers are comparative policy outputs, not hardware benchmark results.

## What is synthetic here

- Tier latencies and bandwidth
- Decode-window pinning and HBM floor behavior
- Congestion and queue-depth penalties
- Attention density estimates
- Recompute worthiness
- Multicast savings
- Failure and degradation triggers
- Energy-per-token proxy

## Why these numbers are not hardware benchmarks

The simulator does not run CUDA kernels, does not overlap real NCCL traffic,
does not issue PCIe DMA, and does not replay production serving traces from
actual runtimes. That means it cannot tell you real p99 latency or real GPU
utilization.

## What would be needed for more realism

1. Runtime-shaped traces from vLLM, TensorRT-LLM, LMCache, or similar systems
2. Calibrated transfer timing against real NVLink / PCIe / CXL platforms
3. Measured decode critical-path overlap and prefetch timing
4. Better attention and recompute models from real workloads
5. Real failure and degraded-mode telemetry from distributed serving systems

## Why keep the synthetic model anyway

Because policy research often needs a controllable environment first.
This repo is useful for comparing:

- active HBM floors vs unrestricted spill
- exact reuse vs approximate structural reuse
- point-to-point delivery vs multicast
- naive spill vs topology-aware placement
- LRU vs semantic + attention-aware eviction

Those are legitimate research questions even when the numeric outputs are
explicitly labeled as synthetic.
