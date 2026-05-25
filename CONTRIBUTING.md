# Contributing

Thanks for contributing to `semantic-kv-control-plane`.

This repository is a systems research simulator. The goal is to make policy and
architecture experiments easy to reproduce, compare, and extend without
pretending to be a production inference runtime.

## Development Setup

1. Clone the repository.
2. Create and activate a Python 3.11+ environment.
3. Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

4. Run the checks:

```bash
python -m ruff check
python -m pytest --tb=short
```

## Code Style

- Python 3.11+
- `ruff` for linting and import organization
- Google-style docstrings for public classes and functions
- Type hints on public APIs and helper functions where practical
- Keep the project lightweight: avoid heavy runtime or ML dependencies

## Adding a New Policy Class

1. Add a new class in [src/semantic_kv/placement.py](/C:/Users/ManishKL/Documents/Playground/semantic-kv-control-plane/src/semantic_kv/placement.py).
2. Subclass `PlacementPolicy`.
3. Implement `choose_tier(block, tiers) -> PlacementDecision`.
4. Give the policy a stable `name`.
5. Register the policy in `make_policy(...)`.
6. Add targeted tests for expected placement behavior.
7. If the policy needs special demotion rules, pair it with a matching
   `EvictionPolicy`.

Minimal sketch:

```python
class MyPolicy(PlacementPolicy):
    name = "my-policy"

    def choose_tier(
        self,
        block: KVBlock,
        tiers: dict[MemoryTier, MemoryTierState],
    ) -> PlacementDecision:
        return PlacementDecision(
            target_tier=MemoryTier.KV_APPLIANCE,
            reason="example policy",
            expected_latency_us=tiers[MemoryTier.KV_APPLIANCE].latency_us,
            moved_bytes=block.bytes_stored,
        )
```

## Adding a New Workload Scenario

1. Add a generator in [src/semantic_kv/workloads.py](/C:/Users/ManishKL/Documents/Playground/semantic-kv-control-plane/src/semantic_kv/workloads.py)
   or a trace generator in
   [src/semantic_kv/trace_generators.py](/C:/Users/ManishKL/Documents/Playground/semantic-kv-control-plane/src/semantic_kv/trace_generators.py).
2. Keep it synthetic, deterministic, and small enough for tests.
3. Register the workload in `make_workload(...)` or `GENERATORS`.
4. Add at least one test covering the new scenario.
5. If it belongs in the benchmark suite, add it to
   [benchmarks/scenarios.py](/C:/Users/ManishKL/Documents/Playground/semantic-kv-control-plane/benchmarks/scenarios.py).

## Pull Request Checklist

- [ ] `python -m ruff check` passes
- [ ] `python -m pytest --tb=short` passes
- [ ] New public code has docstrings and type hints
- [ ] New policy/workload behavior has tests
- [ ] README or docs are updated if the change affects user-facing behavior
- [ ] Synthetic results remain clearly labeled as simulation-only
