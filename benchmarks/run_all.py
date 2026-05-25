"""Run the full trace replay benchmark suite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Execute every named benchmark scenario and write result artifacts."""

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from benchmarks.scenarios import run_all_scenarios

    frame = run_all_scenarios(ROOT / "benchmarks" / "results")
    print(f"Wrote {len(frame)} scenario rows to {ROOT / 'benchmarks' / 'results'}")


if __name__ == "__main__":
    main()
