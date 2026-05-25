"""Run the full trace replay benchmark suite."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Execute every named benchmark scenario and write result artifacts."""

    from benchmarks.scenarios import run_all_scenarios

    frame = run_all_scenarios(ROOT / "benchmarks" / "results")
    print(f"Wrote {len(frame)} scenario rows to {ROOT / 'benchmarks' / 'results'}")


if __name__ == "__main__":
    main()
