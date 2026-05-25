"""Simulated memory tiers and capacity accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_kv.models import KVBlock, MemoryTier

GB = 1024**3


@dataclass
class MemoryTierState:
    name: MemoryTier
    capacity_bytes: int
    bandwidth_gbps: float
    latency_us: float
    used_bytes: int = 0
    stored_block_ids: set[str] = field(default_factory=set)

    def can_fit(self, block: KVBlock | int) -> bool:
        required = block if isinstance(block, int) else block.bytes_stored
        return self.used_bytes + required <= self.capacity_bytes

    def add_block(self, block: KVBlock) -> None:
        if block.block_id in self.stored_block_ids:
            return
        if not self.can_fit(block):
            raise ValueError(f"{self.name.value} cannot fit block {block.block_id}")
        self.stored_block_ids.add(block.block_id)
        self.used_bytes += block.bytes_stored
        block.tier = self.name

    def remove_block(self, block: KVBlock) -> None:
        if block.block_id not in self.stored_block_ids:
            return
        self.stored_block_ids.remove(block.block_id)
        self.used_bytes = max(0, self.used_bytes - block.bytes_stored)

    def occupancy_pct(self) -> float:
        return 100.0 * self.used_bytes / self.capacity_bytes if self.capacity_bytes else 0.0


def default_tier_profiles(scale: float = 1.0) -> dict[MemoryTier, MemoryTierState]:
    """Return illustrative simulation tiers. Capacities can be scaled for tests."""

    return {
        MemoryTier.GPU_HBM: MemoryTierState(MemoryTier.GPU_HBM, int(80 * GB * scale), 3000, 1),
        MemoryTier.KV_APPLIANCE: MemoryTierState(
            MemoryTier.KV_APPLIANCE, int(512 * GB * scale), 800, 8
        ),
        MemoryTier.CXL_POOL: MemoryTierState(MemoryTier.CXL_POOL, int(2048 * GB * scale), 256, 40),
        MemoryTier.NVME_OBJECT: MemoryTierState(
            MemoryTier.NVME_OBJECT, int(16384 * GB * scale), 32, 300
        ),
    }
