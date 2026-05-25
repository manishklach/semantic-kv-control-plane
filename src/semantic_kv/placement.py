"""Placement policies for semantic KV memory tiers."""

from __future__ import annotations

from dataclasses import dataclass

from semantic_kv.models import EvictionClass, KVBlock, MemoryTier
from semantic_kv.tiers import MemoryTierState
from semantic_kv.topology import RackTopology, default_rack_topology


@dataclass(frozen=True)
class PlacementDecision:
    target_tier: MemoryTier
    reason: str
    expected_latency_us: float
    moved_bytes: int


class PlacementPolicy:
    name = "base"

    def choose_tier(
        self, block: KVBlock, tiers: dict[MemoryTier, MemoryTierState]
    ) -> PlacementDecision:
        raise NotImplementedError


class NaiveHBMPolicy(PlacementPolicy):
    name = "naive-hbm"

    def choose_tier(
        self, block: KVBlock, tiers: dict[MemoryTier, MemoryTierState]
    ) -> PlacementDecision:
        target = MemoryTier.GPU_HBM
        reason = "HBM first; LRU spill makes room when full"
        return PlacementDecision(target, reason, tiers[target].latency_us, block.bytes_stored)


class CXLSpillPolicy(PlacementPolicy):
    name = "cxl-spill"

    def choose_tier(
        self, block: KVBlock, tiers: dict[MemoryTier, MemoryTierState]
    ) -> PlacementDecision:
        target = (
            MemoryTier.GPU_HBM
            if block.eviction_class is EvictionClass.HOT_ACTIVE
            else MemoryTier.CXL_POOL
        )
        if not tiers[target].can_fit(block):
            target = MemoryTier.NVME_OBJECT
        reason = "hot active stays in HBM, other KV spills to pooled memory"
        return PlacementDecision(target, reason, tiers[target].latency_us, block.bytes_stored)


class SemanticKVPolicy(PlacementPolicy):
    name = "semantic"

    def choose_tier(
        self, block: KVBlock, tiers: dict[MemoryTier, MemoryTierState]
    ) -> PlacementDecision:
        if block.eviction_class is EvictionClass.HOT_ACTIVE:
            target, reason = MemoryTier.GPU_HBM, "active decode block"
        elif block.eviction_class is EvictionClass.REUSABLE_PREFIX and block.fanout_count >= 2:
            target, reason = MemoryTier.KV_APPLIANCE, "shared prefix with fanout"
        elif block.eviction_class is EvictionClass.SESSION_RECENT:
            target, reason = MemoryTier.KV_APPLIANCE, "recent session state"
        elif block.eviction_class is EvictionClass.SESSION_COLD:
            target, reason = MemoryTier.CXL_POOL, "cold session state"
        elif block.eviction_class is EvictionClass.SAFE_TO_RECOMPUTE:
            target, reason = MemoryTier.NVME_OBJECT, "cheap to recompute"
        elif block.eviction_class is EvictionClass.LOW_ATTENTION:
            target, reason = MemoryTier.CXL_POOL, "low-attention block can be compressed"
        elif block.eviction_class is EvictionClass.EPHEMERAL_TOOL_CALL:
            target, reason = MemoryTier.NVME_OBJECT, "short-lived tool-call KV"
        else:
            target, reason = MemoryTier.CXL_POOL, "default semantic placement"

        for fallback in [target, MemoryTier.CXL_POOL, MemoryTier.NVME_OBJECT]:
            if tiers[fallback].can_fit(block):
                return PlacementDecision(fallback, reason, tiers[fallback].latency_us, block.bytes_stored)
        return PlacementDecision(MemoryTier.NVME_OBJECT, "oversubscribed fallback", tiers[MemoryTier.NVME_OBJECT].latency_us, block.bytes_stored)


class TopologyAwareSemanticPolicy(SemanticKVPolicy):
    name = "topology-aware-semantic"

    def __init__(self, topology: RackTopology | None = None) -> None:
        self.topology = topology or default_rack_topology()

    def choose_tier(
        self, block: KVBlock, tiers: dict[MemoryTier, MemoryTierState]
    ) -> PlacementDecision:
        gpu = self.topology.gpu_for_session(block.session_id)
        appliance = self.topology.preferred_appliance(gpu.gpu_id)
        congestion = self.topology.congestion_penalty(gpu.gpu_id, appliance.appliance_id)
        if block.eviction_class is EvictionClass.HOT_ACTIVE:
            reason = f"active decode near {gpu.gpu_id}; avoid fabric hop"
            return PlacementDecision(MemoryTier.GPU_HBM, reason, tiers[MemoryTier.GPU_HBM].latency_us, block.bytes_stored)
        if block.eviction_class is EvictionClass.REUSABLE_PREFIX and block.fanout_count >= 4:
            self.topology.reserve_appliance(appliance.appliance_id, min(0.1, block.fanout_count / 10_000))
            latency = tiers[MemoryTier.KV_APPLIANCE].latency_us + congestion * 10
            reason = f"rack-local prefix cache on {appliance.appliance_id}; fanout={block.fanout_count}"
            return PlacementDecision(MemoryTier.KV_APPLIANCE, reason, latency, block.bytes_stored)
        if block.eviction_class in {EvictionClass.SESSION_RECENT, EvictionClass.LOW_ATTENTION}:
            target = MemoryTier.KV_APPLIANCE if congestion < 0.75 else MemoryTier.CXL_POOL
            reason = f"locality-aware recent KV; congestion={congestion:.2f}"
            return PlacementDecision(target, reason, tiers[target].latency_us + congestion * 8, block.bytes_stored)
        return super().choose_tier(block, tiers)


class DistributedSemanticKVPolicy(TopologyAwareSemanticPolicy):
    name = "distributed-semantic-kv"

    def choose_tier(
        self, block: KVBlock, tiers: dict[MemoryTier, MemoryTierState]
    ) -> PlacementDecision:
        decision = super().choose_tier(block, tiers)
        if block.eviction_class is EvictionClass.REUSABLE_PREFIX and block.fanout_count >= 16:
            gpu = self.topology.gpu_for_session(block.session_id)
            appliance = self.topology.preferred_appliance(gpu.gpu_id)
            return PlacementDecision(
                MemoryTier.KV_APPLIANCE,
                f"distributed prefix multicast anchor on {appliance.appliance_id}; fanout={block.fanout_count}",
                max(1.0, decision.expected_latency_us * 0.8),
                block.bytes_stored,
            )
        return decision


def make_policy(name: str) -> PlacementPolicy:
    normalized = name.lower()
    if normalized in {"naive", "naive-hbm", "hbm"}:
        return NaiveHBMPolicy()
    if normalized in {"cxl", "cxl-spill"}:
        return CXLSpillPolicy()
    if normalized in {"semantic", "semantic-kv"}:
        return SemanticKVPolicy()
    if normalized in {"topology", "topology-aware", "topology-aware-semantic"}:
        return TopologyAwareSemanticPolicy()
    if normalized in {"distributed", "distributed-semantic", "distributed-semantic-kv"}:
        return DistributedSemanticKVPolicy()
    raise ValueError(f"Unknown placement policy: {name}")
