"""Simple prefetch scheduling model."""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_kv.models import MemoryTier


@dataclass(frozen=True)
class PrefetchRequest:
    session_id: str
    next_token_start: int
    next_token_count: int
    target_tier: MemoryTier
    deadline_us: float


@dataclass
class PrefetchScheduler:
    prefetch_requests: int = 0
    prefetch_hits: int = 0
    late_prefetches: int = 0
    avoided_stall_us: float = 0.0
    scheduled: dict[str, PrefetchRequest] = field(default_factory=dict)

    def predict_next_blocks(
        self, session_id: str, current_token: int, window: int, target_tier: MemoryTier
    ) -> list[PrefetchRequest]:
        return [
            PrefetchRequest(session_id, current_token + window, window, target_tier, deadline_us=50)
        ]

    def schedule_prefetch(self, request: PrefetchRequest) -> None:
        key = f"{request.session_id}:{request.next_token_start}"
        self.scheduled[key] = request
        self.prefetch_requests += 1

    def mark_success(self, avoided_stall_us: float) -> None:
        self.prefetch_hits += 1
        self.avoided_stall_us += avoided_stall_us

    def mark_failure(self) -> None:
        self.late_prefetches += 1

    @property
    def prefetch_success_rate(self) -> float:
        return self.prefetch_hits / self.prefetch_requests if self.prefetch_requests else 0.0
