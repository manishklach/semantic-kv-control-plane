"""Stall and backpressure modeling for synthetic inference memory experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


def _percentile(values: list[float], percentile: float) -> float:
    """Return a simple percentile without external numeric dependencies."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


@dataclass(frozen=True)
class StallBreakdown:
    """Represent one decode stall observation with component penalties."""

    queue_delay_us: float
    fabric_wait_us: float
    dma_transfer_us: float
    decode_pause_us: float
    cache_miss_penalty_us: float
    serialization_penalty_us: float
    prefetch_lateness_penalty_us: float
    overlap_discount_us: float = 0.0

    @property
    def total_us(self) -> float:
        """Return the net stall after subtracting transfer overlap."""

        gross = (
            self.queue_delay_us
            + self.fabric_wait_us
            + self.dma_transfer_us
            + self.decode_pause_us
            + self.cache_miss_penalty_us
            + self.serialization_penalty_us
            + self.prefetch_lateness_penalty_us
        )
        return max(0.0, gross - self.overlap_discount_us)


@dataclass
class StallModel:
    """Accumulate simulated stall events and compute latency percentiles."""

    events: list[StallBreakdown] = field(default_factory=list)

    def record(self, breakdown: StallBreakdown) -> float:
        """Record a stall event and return its total penalty."""

        self.events.append(breakdown)
        return breakdown.total_us

    def queueing_delay_us(
        self,
        occupancy: float,
        queue_depth: float,
        base_latency_us: float,
    ) -> float:
        """Estimate queueing delay from occupancy and depth."""

        return base_latency_us * min(12.0, occupancy * 3.2 + queue_depth * 1.5)

    def fabric_wait_us(self, congestion: float, serialization_penalty_us: float) -> float:
        """Estimate fabric wait from congestion and serialization."""

        return max(0.0, serialization_penalty_us * (0.4 + congestion))

    def transfer_overlap_discount_us(self, prefetch_ready: bool, dma_transfer_us: float) -> float:
        """Estimate how much DMA latency overlaps with useful decode work."""

        if prefetch_ready:
            return dma_transfer_us * 0.7
        return dma_transfer_us * 0.18

    def summary(self) -> dict[str, float]:
        """Return percentile and average stall metrics."""

        totals = [event.total_us for event in self.events]
        return {
            "count": float(len(totals)),
            "mean_us": sum(totals) / len(totals) if totals else 0.0,
            "p50_us": _percentile(totals, 0.50),
            "p95_us": _percentile(totals, 0.95),
            "p99_us": _percentile(totals, 0.99),
            "p999_us": _percentile(totals, 0.999),
        }
