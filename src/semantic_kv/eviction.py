"""Eviction policies for simulated KV placement."""

from __future__ import annotations

from dataclasses import dataclass

from semantic_kv.compression import apply_compression
from semantic_kv.models import CompressionMode, EvictionClass, KVBlock, MemoryTier
from semantic_kv.topology import RackTopology, default_rack_topology


@dataclass(frozen=True)
class EvictionResult:
    """Capture victims, compressed blocks, and total bytes freed."""

    victims: list[KVBlock]
    compressed: list[KVBlock]
    freed_bytes: int


class EvictionPolicy:
    """Interface for eviction strategies used by the simulator."""

    name = "base"

    def select_victim(
        self, blocks: list[KVBlock], required_bytes: int, current_step: int
    ) -> EvictionResult:
        raise NotImplementedError


class LRUEviction(EvictionPolicy):
    """Evict the least recently accessed blocks first."""

    name = "lru"

    def select_victim(
        self, blocks: list[KVBlock], required_bytes: int, current_step: int
    ) -> EvictionResult:
        victims: list[KVBlock] = []
        freed = 0
        for block in sorted(blocks, key=lambda item: item.last_access_step):
            victims.append(block)
            freed += block.bytes_stored
            if freed >= required_bytes:
                break
        return EvictionResult(victims=victims, compressed=[], freed_bytes=freed)


class SemanticEviction(EvictionPolicy):
    """Evict using semantic metadata rather than recency alone."""

    name = "semantic"

    CLASS_WEIGHTS = {
        EvictionClass.EPHEMERAL_TOOL_CALL: 0,
        EvictionClass.SAFE_TO_RECOMPUTE: 10,
        EvictionClass.LOW_ATTENTION: 25,
        EvictionClass.SESSION_COLD: 40,
        EvictionClass.SESSION_RECENT: 60,
        EvictionClass.REUSABLE_PREFIX: 500,
        EvictionClass.HOT_ACTIVE: 1000,
    }

    TIER_PENALTY = {
        MemoryTier.GPU_HBM: -20,
        MemoryTier.KV_APPLIANCE: 0,
        MemoryTier.CXL_POOL: 8,
        MemoryTier.NVME_OBJECT: 20,
    }

    def score(self, block: KVBlock, current_step: int) -> float:
        """Return a keep score; lower-scored blocks are better eviction victims.

        LRU only knows when a block was touched. Semantic eviction uses KV intent:
        shared prefixes and active decode state are protected, while ephemeral or
        cheap-to-recompute blocks are allowed to leave fast tiers earlier.
        """
        age = max(0, current_step - block.last_access_step)
        protected = self.CLASS_WEIGHTS[block.eviction_class]
        reuse = block.reuse_score * 100
        fanout = min(block.fanout_count, 100) * 5
        prefix_bonus = 75 if block.prefix_hash else 0
        recompute = min(block.recompute_cost / 1_000_000, 100)
        heat = block.heat_score * 120
        attention = block.attention_importance * 90
        decode_priority = block.decode_priority * 80
        recompute_credit = block.recompute_worthiness * -75
        tier_penalty = self.TIER_PENALTY[block.tier]
        return (
            protected
            + reuse
            + fanout
            + prefix_bonus
            + recompute
            + heat
            + attention
            + decode_priority
            + recompute_credit
            - age
            + tier_penalty
        )

    def select_victim(
        self, blocks: list[KVBlock], required_bytes: int, current_step: int
    ) -> EvictionResult:
        victims: list[KVBlock] = []
        compressed: list[KVBlock] = []
        freed = 0

        for block in blocks:
            if block.eviction_class is EvictionClass.LOW_ATTENTION and not block.compressed:
                before = block.bytes_stored
                apply_compression(block, CompressionMode.BLOCK_QUANT_SIM)
                compressed.append(block)
                freed += before - block.bytes_stored
                if freed >= required_bytes:
                    return EvictionResult(victims=[], compressed=compressed, freed_bytes=freed)

        candidates = sorted(blocks, key=lambda item: self.score(item, current_step))
        for block in candidates:
            if block.eviction_class in {EvictionClass.HOT_ACTIVE, EvictionClass.REUSABLE_PREFIX}:
                continue
            victims.append(block)
            freed += block.bytes_stored
            if freed >= required_bytes:
                break
        return EvictionResult(victims=victims, compressed=compressed, freed_bytes=freed)


class DistributedSemanticEvictionPolicy(SemanticEviction):
    """Prefer migration to nearby tiers before evicting reusable prefixes."""

    name = "distributed-semantic"

    def __init__(self, topology: RackTopology | None = None) -> None:
        self.topology = topology or default_rack_topology()

    def select_victim(
        self, blocks: list[KVBlock], required_bytes: int, current_step: int
    ) -> EvictionResult:
        migration_candidates = [
            block
            for block in blocks
            if block.eviction_class is EvictionClass.REUSABLE_PREFIX
            and block.tier is MemoryTier.GPU_HBM
        ]
        if migration_candidates:
            migrated: list[KVBlock] = []
            freed = 0
            for block in sorted(migration_candidates, key=lambda item: -item.fanout_count):
                # Migration-before-eviction: preserve hot prefixes by demoting
                # them to an appliance tier, freeing HBM without losing reuse.
                block.tier = MemoryTier.KV_APPLIANCE
                migrated.append(block)
                freed += block.bytes_stored
                if freed >= required_bytes:
                    return EvictionResult(victims=[], compressed=migrated, freed_bytes=freed)
        return super().select_victim(blocks, required_bytes, current_step)
