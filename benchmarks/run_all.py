"""Run the full trace replay benchmark suite."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.scenarios import run_all_scenarios


if __name__ == "__main__":
    frame = run_all_scenarios(ROOT / "benchmarks" / "results")
    print(f"Wrote {len(frame)} scenario rows to {ROOT / 'benchmarks' / 'results'}")
