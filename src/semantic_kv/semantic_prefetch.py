"""Semantic prefetch engine for future KV movement simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from semantic_kv.models import MemoryTier


class PrefetchPriority(StrEnum):
    """Priority buckets for semantic prefetch requests."""

    URGENT = "URGENT"
    HOT_PATH = "HOT_PATH"
    WARM_PATH = "WARM_PATH"
    BEST_EFFORT = "BEST_EFFORT"


@dataclass(frozen=True)
class SemanticPrefetchPlan:
    """Describe a predicted KV movement needed before a decode stall."""

    session_id: str
    target_tier: MemoryTier
    token_start: int
    token_count: int
    priority: PrefetchPriority
    reason: str
    deadline_us: float


@dataclass
class PredictivePrefetchEngine:
    """Track simple semantic prefetch planning and hit accounting."""

    prefetch_requests: int = 0
    prefetch_hits: int = 0
    avoided_decode_stalls: float = 0.0
    late_prefetch_penalty: float = 0.0
    wasted_prefetch_bytes: int = 0

    def plan(
        self,
        session_id: str,
        current_token: int,
        decode_velocity_tps: float,
        prefix_reuse_probability: float,
        active_tool_loop: bool = False,
        beam_divergence: float = 0.0,
    ) -> SemanticPrefetchPlan:
        """Create a simulated prefetch plan from lightweight session signals."""

        if active_tool_loop or decode_velocity_tps > 120:
            priority = PrefetchPriority.URGENT
            reason = "fast decode or active tool loop"
        elif prefix_reuse_probability > 0.7:
            priority = PrefetchPriority.HOT_PATH
            reason = "high prefix reuse probability"
        elif beam_divergence > 0.5:
            priority = PrefetchPriority.WARM_PATH
            reason = "beam divergence may branch future KV"
        else:
            priority = PrefetchPriority.BEST_EFFORT
            reason = "low-pressure speculative prefetch"
        window = 256 if priority in {PrefetchPriority.URGENT, PrefetchPriority.HOT_PATH} else 128
        deadline = max(25.0, window / max(decode_velocity_tps, 1.0) * 1_000)
        return SemanticPrefetchPlan(
            session_id=session_id,
            target_tier=MemoryTier.GPU_HBM,
            token_start=current_token + window,
            token_count=window,
            priority=priority,
            reason=reason,
            deadline_us=deadline,
        )

    def mark_result(self, hit: bool, avoided_stall_us: float = 0.0, wasted_bytes: int = 0) -> None:
        """Record whether a scheduled prefetch arrived before use."""

        self.prefetch_requests += 1
        if hit:
            self.prefetch_hits += 1
            self.avoided_decode_stalls += avoided_stall_us
        else:
            self.late_prefetch_penalty += avoided_stall_us
            self.wasted_prefetch_bytes += wasted_bytes

    @property
    def prefetch_hit_rate(self) -> float:
        """Return the fraction of scheduled prefetches that were hits."""

        return self.prefetch_hits / self.prefetch_requests if self.prefetch_requests else 0.0
