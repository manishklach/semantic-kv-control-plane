# Results Confidence

These benchmark artifacts are **synthetic simulation outputs**, not hardware
measurements. They are useful for comparing policy behavior inside this repo's
assumptions, but they should not be read as claims about real GPU throughput,
real vLLM performance, or production network latency.

## What the numbers mean

- `peak_hbm_gb`: simulated peak HBM residency under the chosen policy.
- `bytes_moved_gb`: modeled KV traffic moved between tiers.
- `dedup_saved_gb`: duplicate KV residency avoided through exact or distributed
  prefix reuse.
- `compression_saved_gb`: storage bytes avoided through simulated compression
  modes.
- `stall_overhead_ms`: decode-stall proxy derived from simulated tier latency,
  bandwidth pressure, and transfer penalties.
- `throughput_score`: a relative synthetic score that combines reuse benefit and
  movement/stall penalties.
- `energy_per_token`: lightweight movement-energy proxy, not a power draw
  measurement.

## Main assumptions

- Tier capacities, latency, and bandwidth are illustrative.
- Prefix reuse is driven by synthetic trace structure and exact hash or MinHash
  token similarity.
- 5-15% deterministic prefix variance is injected into shared-prefix scenarios
  to avoid best-case hit rates.
- Congestion, queue depth, and energy are simplified comparative models.

## What these results do not tell you

- Real CUDA execution cost
- Real kernel overlap behavior
- Real vLLM, TensorRT-LLM, LMCache, or SGLang runtime performance
- Actual NIC, NVLink, PCIe, or CXL hardware throughput
- End-user latency on production traffic

Use these outputs as **policy evidence under synthetic workload assumptions**,
not as hardware benchmark claims.
