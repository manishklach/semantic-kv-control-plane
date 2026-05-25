"""Reproduce the complete synthetic evidence package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Generate traces, benchmark results, figures, and blog assets."""

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    from benchmarks.scenarios import run_all_scenarios
    from scripts.generate_blog_assets import generate_blog_assets
    from scripts.generate_paper_figures import generate_figures
    from semantic_kv.trace_generators import export_sample_traces

    trace_paths = export_sample_traces(ROOT / "examples" / "traces")
    frame = run_all_scenarios(ROOT / "benchmarks" / "results")
    figures = generate_figures()
    blog_dir = generate_blog_assets()
    print("Reproduction complete.")
    print(f"Traces generated: {len(trace_paths)} files")
    print(f"Benchmark rows: {len(frame)}")
    print(f"Figures generated: {len(figures)}")
    print(f"Blog assets: {blog_dir}")


if __name__ == "__main__":
    main()
