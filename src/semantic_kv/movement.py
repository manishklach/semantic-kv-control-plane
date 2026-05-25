"""Memory movement and energy accounting for distributed KV simulation."""

from __future__ import annotations

from dataclasses import dataclass

from semantic_kv.models import MemoryTier

ENERGY_PJ_PER_BYTE = {
    "hbm": 3.5,
    "pcie": 18.0,
    "cxl": 24.0,
    "cross_rack": 85.0,
    "recompute": 140.0,
}


@dataclass
class MovementStats:
    """Accumulate movement, avoidance, and energy counters."""

    bytes_moved: int = 0
    bytes_avoided: int = 0
    multicast_saved_bytes: int = 0
    avoided_recompute_bytes: int = 0
    avoided_hbm_residency_bytes: int = 0
    avoided_cross_rack_bytes: int = 0
    movement_energy_j: float = 0.0

    @property
    def energy_per_token(self) -> float:
        """Estimate movement energy normalized by moved megabytes."""

        moved_mb = self.bytes_moved / (1024**2)
        return self.movement_energy_j / moved_mb if moved_mb else 0.0


class MovementAnalyzer:
    """Collect movement and estimate relative energy costs.

    Energy constants are illustrative. The goal is to compare movement choices,
    including cases where recomputation may be cheaper than remote KV movement.
    """

    def __init__(self) -> None:
        self.stats = MovementStats()

    def record_move(
        self, bytes_moved: int, source: MemoryTier, target: MemoryTier, cross_rack: bool = False
    ) -> None:
        self.stats.bytes_moved += bytes_moved
        mode = "cross_rack" if cross_rack else self._mode(source, target)
        self.stats.movement_energy_j += bytes_moved * ENERGY_PJ_PER_BYTE[mode] * 1e-12

    def record_avoided(
        self,
        bytes_avoided: int,
        multicast: bool = False,
        recompute: bool = False,
        hbm_residency: bool = False,
        cross_rack: bool = False,
    ) -> None:
        self.stats.bytes_avoided += bytes_avoided
        if multicast:
            self.stats.multicast_saved_bytes += bytes_avoided
        if recompute:
            self.stats.avoided_recompute_bytes += bytes_avoided
        if hbm_residency:
            self.stats.avoided_hbm_residency_bytes += bytes_avoided
        if cross_rack:
            self.stats.avoided_cross_rack_bytes += bytes_avoided

    def recompute_energy_j(self, bytes_to_recompute: int) -> float:
        return bytes_to_recompute * ENERGY_PJ_PER_BYTE["recompute"] * 1e-12

    def _mode(self, source: MemoryTier, target: MemoryTier) -> str:
        if source is MemoryTier.GPU_HBM or target is MemoryTier.GPU_HBM:
            return "hbm"
        if MemoryTier.CXL_POOL in {source, target}:
            return "cxl"
        return "pcie"
