"""Generate blog outline, findings, figures, and social thread draft."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BLOG_TITLE = "When KV Cache Becomes a Distributed Systems Problem"


def generate_blog_assets(output_dir: Path | None = None) -> Path:
    """Build blog-ready markdown and figure assets from benchmark outputs."""

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    from scripts.generate_paper_figures import generate_figures
    from semantic_kv.analysis import ResultInterpreter

    output_dir = output_dir or ROOT / "outputs" / "blog"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_csv = ROOT / "benchmarks" / "results" / "scenario_results.csv"
    df = pd.read_csv(results_csv)
    interpreter = ResultInterpreter(df)
    (output_dir / "blog_outline.md").write_text(_outline(), encoding="utf-8")
    (output_dir / "key_findings.md").write_text(interpreter.to_markdown(), encoding="utf-8")
    (output_dir / "social_thread.md").write_text(_social_thread(), encoding="utf-8")
    for figure in generate_figures(results_csv, ROOT / "outputs" / "paper_figures"):
        shutil.copy2(figure, figures_dir / figure.name)
    return output_dir


def _outline() -> str:
    return f"""# {BLOG_TITLE}

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
"""


def _social_thread() -> str:
    return f"""# Social Thread Draft: {BLOG_TITLE}

1. KV cache is starting to look less like a runtime detail and more like distributed infrastructure.

2. Long context, shared prompts, RAG payloads, and agent loops all create
memory pressure that generic spill policies cannot fully understand.

3. I built a synthetic simulation platform to ask a narrow question: what
changes if KV metadata includes prefix hashes, fanout, eviction class,
topology, and movement cost?

4. The project does not run CUDA and does not claim real hardware speedups.
It is a policy sandbox for memory-orchestrated inference.

5. The interesting result is not a single number. It is the shape of the
problem: HBM pressure, movement, prefix reuse, congestion, and energy proxy
interact.

6. CXL adds capacity, but semantic orchestration asks what should move, where
it should live, and whether movement is worth it.

7. Next step: import real serving traces and compare simulated policies against
runtime-observed KV behavior.
"""


if __name__ == "__main__":
    print(generate_blog_assets())
