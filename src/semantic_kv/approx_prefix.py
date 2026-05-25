"""Approximate structural prefix matching for KV reuse experiments.

This module models token-level structural similarity with MinHash/Jaccard, not
embedding-based semantic similarity. It is intentionally lightweight so the
simulator can compare exact-hash reuse against fuzzy structural reuse without
pulling in a vector search stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from datasketch import MinHash


def _normalized_items(tokens: list[int] | tuple[int, ...]) -> tuple[str, ...]:
    """Convert a token prefix into stable set-like items for MinHash."""

    return tuple(sorted({str(token) for token in tokens}))


def exact_prefix_key(tokens: list[int] | tuple[int, ...]) -> str:
    """Return a deterministic exact-match key for a token prefix."""

    return ",".join(str(token) for token in tokens)


def _minhash(tokens: list[int] | tuple[int, ...], num_perm: int) -> MinHash:
    """Build a MinHash sketch for a token prefix."""

    sketch = MinHash(num_perm=num_perm)
    for item in _normalized_items(tokens):
        sketch.update(item.encode("utf-8"))
    return sketch


@dataclass(frozen=True)
class PrefixMatch:
    """Represent one approximate structural prefix match."""

    prefix_id: str
    similarity: float
    tokens: tuple[int, ...]


@dataclass
class MinHashPrefixIndex:
    """Index token prefixes with MinHash for approximate Jaccard lookup."""

    num_perm: int = 128
    prefixes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    sketches: dict[str, MinHash] = field(default_factory=dict)

    def add(self, prefix_id: str, tokens: list[int] | tuple[int, ...]) -> None:
        """Insert a token prefix into the approximate index."""

        normalized = tuple(tokens)
        self.prefixes[prefix_id] = normalized
        self.sketches[prefix_id] = _minhash(normalized, self.num_perm)

    def similarity(
        self,
        tokens_a: list[int] | tuple[int, ...],
        tokens_b: list[int] | tuple[int, ...],
    ) -> float:
        """Estimate Jaccard similarity between two token prefixes."""

        return _minhash(tokens_a, self.num_perm).jaccard(_minhash(tokens_b, self.num_perm))

    def query(
        self,
        tokens: list[int] | tuple[int, ...],
        *,
        top_k: int = 1,
    ) -> list[PrefixMatch]:
        """Return the closest stored prefixes by approximate Jaccard score."""

        if not self.sketches:
            return []
        query_sketch = _minhash(tokens, self.num_perm)
        scored = sorted(
            (
                PrefixMatch(prefix_id, query_sketch.jaccard(sketch), self.prefixes[prefix_id])
                for prefix_id, sketch in self.sketches.items()
            ),
            key=lambda match: match.similarity,
            reverse=True,
        )
        return scored[:top_k]


@dataclass
class ApproxPrefixMatcher:
    """Match new token prefixes against stored structural prefixes."""

    threshold: float = 0.85
    index: MinHashPrefixIndex = field(default_factory=MinHashPrefixIndex)

    def add(self, prefix_id: str, tokens: list[int] | tuple[int, ...]) -> None:
        """Insert a reusable prefix into the matcher index."""

        self.index.add(prefix_id, tokens)

    def match(self, tokens: list[int] | tuple[int, ...]) -> PrefixMatch | None:
        """Return the closest stored prefix if it meets the threshold."""

        matches = self.index.query(tokens, top_k=1)
        if not matches:
            return None
        best = matches[0]
        return best if best.similarity >= self.threshold else None


def generate_prefix_variants(
    *,
    sessions: int,
    prefix_tokens: int = 512,
    variation_rate: float = 0.05,
    seed: int = 7,
) -> list[list[int]]:
    """Generate token prefixes with small random structural variation."""

    rng = Random(seed)
    base = [10_000 + idx for idx in range(prefix_tokens)]
    variants: list[list[int]] = []
    mutations = max(1, int(prefix_tokens * variation_rate))
    for session in range(sessions):
        tokens = list(base)
        if session:
            for _ in range(mutations):
                offset = rng.randrange(prefix_tokens)
                tokens[offset] = tokens[offset] + rng.randint(1, 97)
        variants.append(tokens)
    return variants


def compare_exact_and_approx_prefix_hits(
    *,
    sessions: int = 500,
    prefix_tokens: int = 512,
    variation_rate: float = 0.05,
    threshold: float = 0.85,
    seed: int = 7,
) -> dict[str, float | int]:
    """Compare exact-hash and approximate structural prefix hit rates."""

    sequences = generate_prefix_variants(
        sessions=sessions,
        prefix_tokens=prefix_tokens,
        variation_rate=variation_rate,
        seed=seed,
    )
    exact_seen: dict[str, int] = {}
    approx = ApproxPrefixMatcher(threshold=threshold)
    exact_hits = 0
    approx_hits = 0
    for index, tokens in enumerate(sequences):
        key = exact_prefix_key(tokens)
        if key in exact_seen:
            exact_hits += 1
        else:
            exact_seen[key] = index

        match = approx.match(tokens)
        if match is not None:
            approx_hits += 1
        else:
            approx.add(f"prefix-{index}", tokens)

    return {
        "sessions": sessions,
        "prefix_tokens": prefix_tokens,
        "variation_rate": variation_rate,
        "threshold": threshold,
        "exact_hit_count": exact_hits,
        "approx_hit_count": approx_hits,
        "exact_hit_rate": exact_hits / sessions if sessions else 0.0,
        "approx_hit_rate": approx_hits / sessions if sessions else 0.0,
    }
