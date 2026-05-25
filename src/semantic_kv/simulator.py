"""Simulation engine for KV-aware placement experiments.

The engine intentionally models control-plane effects rather than CUDA kernels:
placement choices, transfer cost, tier latency, queueing, dedup references, and
eviction behavior. The numbers are simulation assumptions for policy comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_kv.attention import AttentionImportanceEstimator
from semantic_kv.compression import apply_compression
from semantic_kv.eviction import (
    DistributedSemanticEvictionPolicy,
    EvictionPolicy,
    LRUEviction,
    SemanticEviction,
)
from semantic_kv.failures import FailureInjector
from semantic_kv.heat import HeatModel
from semantic_kv.metadata import PrefixDirectory
from semantic_kv.metrics import SimulationMetrics
from semantic_kv.models import CompressionMode, EvictionClass, KVBlock, MemoryTier, ModelProfile
from semantic_kv.movement import MovementAnalyzer
from semantic_kv.multicast import MulticastPlanner
from semantic_kv.placement import PlacementPolicy, SemanticKVPolicy
from semantic_kv.prefetch import PrefetchScheduler
from semantic_kv.stalls import StallBreakdown, StallModel
from semantic_kv.tiers import MemoryTierState, default_tier_profiles
from semantic_kv.topology import RackTopology, default_rack_topology
from semantic_kv.working_set import HBMReservationManager
from semantic_kv.workloads import EventType, WorkloadEvent

TOPOLOGY_HOPS: dict[tuple[MemoryTier, MemoryTier], float] = {
    (MemoryTier.GPU_HBM, MemoryTier.KV_APPLIANCE): 1.4,
    (MemoryTier.GPU_HBM, MemoryTier.CXL_POOL): 1.9,
    (MemoryTier.GPU_HBM, MemoryTier.NVME_OBJECT): 4.5,
    (MemoryTier.KV_APPLIANCE, MemoryTier.CXL_POOL): 1.3,
    (MemoryTier.CXL_POOL, MemoryTier.NVME_OBJECT): 2.4,
}


@dataclass
class SimulationEngine:
    """Replay workload events through placement, eviction, and tier models."""

    model_profile: ModelProfile
    workload: list[WorkloadEvent]
    placement_policy: PlacementPolicy
    eviction_policy: EvictionPolicy
    tier_config: dict[MemoryTier, MemoryTierState] | None = None
    steps: int | None = None
    active_hbm_floor: float = 0.15
    topology: RackTopology | None = None
    tiers: dict[MemoryTier, MemoryTierState] = field(init=False)
    blocks: dict[str, KVBlock] = field(default_factory=dict)
    prefix_directory: PrefixDirectory = field(default_factory=PrefixDirectory)
    prefetch_scheduler: PrefetchScheduler = field(default_factory=PrefetchScheduler)
    movement_analyzer: MovementAnalyzer = field(default_factory=MovementAnalyzer)
    metrics: SimulationMetrics = field(default_factory=SimulationMetrics)
    reservation_manager: HBMReservationManager = field(init=False)
    stall_model: StallModel = field(default_factory=StallModel)
    heat_model: HeatModel = field(default_factory=HeatModel)
    attention_estimator: AttentionImportanceEstimator = field(
        default_factory=AttentionImportanceEstimator
    )
    multicast_planner: MulticastPlanner = field(default_factory=MulticastPlanner)
    failure_injector: FailureInjector = field(default_factory=FailureInjector)
    _step_bytes_moved: int = 0
    _step_stall_us: float = 0.0

    def __post_init__(self) -> None:
        """Initialize tier state from defaults or caller-provided profiles."""

        self.tiers = self.tier_config or default_tier_profiles()
        self.metrics.active_hbm_floor = self.active_hbm_floor
        self.topology = self.topology or getattr(self.placement_policy, "topology", None)
        self.topology = self.topology or default_rack_topology()
        self.reservation_manager = HBMReservationManager(active_hbm_floor=self.active_hbm_floor)

    def run(self) -> SimulationMetrics:
        """Execute the configured workload and return aggregated metrics."""

        events_by_step: dict[int, list[WorkloadEvent]] = {}
        for event in self.workload:
            events_by_step.setdefault(event.step, []).append(event)
        max_step = self.steps if self.steps is not None else max(events_by_step.keys(), default=0)
        for step in range(max_step + 1):
            self._step_bytes_moved = 0
            self._step_stall_us = 0.0
            self._cool_blocks()
            for event in events_by_step.get(step, []):
                self._process_event(event, step)
            self._check_failures(step)
            self._record_occupancy(step)
        self.metrics.total_stored_bytes = sum(block.bytes_stored for block in self.blocks.values())
        self.metrics.dedup_saved_bytes = self.prefix_directory.dedup_saved_bytes
        self.metrics.prefix_hit_rate = self.prefix_directory.prefix_hit_rate
        self.metrics.prefetch_success_rate = self.prefetch_scheduler.prefetch_success_rate
        self.metrics.bytes_avoided = self.movement_analyzer.stats.bytes_avoided
        self.metrics.multicast_saved_bytes = self.movement_analyzer.stats.multicast_saved_bytes
        self.metrics.avoided_cross_rack_bytes = (
            self.movement_analyzer.stats.avoided_cross_rack_bytes
        )
        self.metrics.movement_energy_j = self.movement_analyzer.stats.movement_energy_j
        created_tokens = max(
            1,
            self.metrics.total_kv_created_bytes
            // max(self.model_profile.estimate_kv_block_bytes(), 1)
            * self.model_profile.block_tokens,
        )
        self.metrics.energy_per_token = self.metrics.movement_energy_j / created_tokens
        self.metrics.topology_congestion_score = self.metrics.bandwidth_saturation_events / max(
            1, len(self.metrics.occupancy_history)
        )
        self.metrics.estimated_ttft_delta_us = -self.metrics.dedup_saved_bytes / 1_000_000
        stall_summary = self.stall_model.summary()
        self.metrics.stall_p50_us = stall_summary["p50_us"]
        self.metrics.stall_p95_us = stall_summary["p95_us"]
        self.metrics.stall_p99_us = stall_summary["p99_us"]
        self.metrics.stall_p999_us = stall_summary["p999_us"]
        self.metrics.estimated_throughput_score = self._throughput_score()
        return self.metrics

    def _process_event(self, event: WorkloadEvent, step: int) -> None:
        if event.event_type is EventType.CREATE_BLOCK and event.block:
            self._create_block(event.block, step)
        elif event.event_type is EventType.ACCESS_BLOCK and event.block_id:
            self._access_block(event.block_id, step)
        elif event.event_type is EventType.PREFETCH_REQUEST and event.session_id:
            request = self.prefetch_scheduler.predict_next_blocks(
                event.session_id, step, self.model_profile.block_tokens, MemoryTier.GPU_HBM
            )[0]
            self.prefetch_scheduler.schedule_prefetch(request)
            self.prefetch_scheduler.mark_success(avoided_stall_us=5)
        elif event.event_type is EventType.RELEASE_BLOCK and event.block_id:
            self._remove_block(event.block_id)
        elif event.event_type is EventType.SESSION_END and event.session_id:
            self.reservation_manager.tracker.release_session(event.session_id, self.blocks)
            for block_id in [
                bid for bid, block in self.blocks.items() if block.session_id == event.session_id
            ]:
                self._remove_block(block_id)

    def _create_block(self, block: KVBlock, step: int) -> None:
        self.metrics.total_kv_created_bytes += block.bytes_uncompressed
        self._apply_attention_and_heat(block, step)
        if isinstance(self.placement_policy, SemanticKVPolicy):
            self._handle_semantic_prefix(block)
            if block.eviction_class is EvictionClass.LOW_ATTENTION:
                self.metrics.compression_saved_bytes += apply_compression(
                    block, CompressionMode.BLOCK_QUANT_SIM
                )
            if block.eviction_class is EvictionClass.EPHEMERAL_TOOL_CALL:
                self.metrics.compression_saved_bytes += apply_compression(
                    block, CompressionMode.INT8_SIM
                )

        original_decision = self.placement_policy.choose_tier(block, self.tiers)
        decision = original_decision
        if self.reservation_manager.should_force_hbm(block):
            decision = decision.__class__(
                MemoryTier.GPU_HBM,
                f"{decision.reason}; forced by active HBM floor",
                self.tiers[MemoryTier.GPU_HBM].latency_us,
                block.bytes_stored,
            )
        self._ensure_capacity(decision.target_tier, block.bytes_stored, step)
        if not self.tiers[decision.target_tier].can_fit(block):
            fallback = self._spill_target_for_pressure(block, original_decision.target_tier)
            self.metrics.failure_events.append(
                {
                    "step": step,
                    "component": decision.target_tier.value,
                    "severity": "warning",
                    "retry_penalty_us": 0.0,
                    "reason": "admission spill under active working-set pressure",
                }
            )
            decision = decision.__class__(
                fallback,
                f"{decision.reason}; degraded spill because hot tier was saturated",
                self.tiers[fallback].latency_us,
                block.bytes_stored,
            )
            self._ensure_capacity(decision.target_tier, block.bytes_stored, step)
        self.tiers[decision.target_tier].add_block(block)
        self.blocks[block.block_id] = block
        self._record_transfer(MemoryTier.GPU_HBM, decision.target_tier, decision.moved_bytes, step)

    def _apply_attention_and_heat(self, block: KVBlock, step: int) -> None:
        """Apply attention-aware and heat-aware metadata updates."""

        session_type = "shared-prefix" if block.prefix_hash else "tool-call"
        if block.eviction_class in {EvictionClass.HOT_ACTIVE, EvictionClass.SESSION_RECENT}:
            session_type = "agentic"
        elif block.eviction_class is EvictionClass.LOW_ATTENTION:
            session_type = "long-context"
        attention_density = max(0.1, min(1.0, block.reuse_score + block.fanout_count / 128))
        estimate = self.attention_estimator.estimate(
            layer_id=block.layer_id,
            token_age=max(0, step - block.created_step) * block.token_count,
            attention_density=attention_density,
            session_type=session_type,
        )
        block.attention_importance = estimate.importance_score
        block.recompute_worthiness = estimate.recompute_worthiness
        if block.recompute_worthiness > 0.85 and block.eviction_class is EvictionClass.SESSION_COLD:
            block.eviction_class = EvictionClass.SAFE_TO_RECOMPUTE
        if (
            estimate.importance_score < 0.25
            and block.eviction_class is EvictionClass.SESSION_RECENT
        ):
            block.eviction_class = EvictionClass.LOW_ATTENTION
        decode_priority = 1.4 if block.eviction_class is EvictionClass.HOT_ACTIVE else 0.9
        self.heat_model.apply(block, current_step=step, decode_priority=decode_priority)

    def _handle_semantic_prefix(self, block: KVBlock) -> None:
        if not block.prefix_hash:
            return
        canonical = self.prefix_directory.lookup_prefix(block.prefix_hash)
        if canonical is None:
            self.prefix_directory.register_prefix(block.prefix_hash, [block])
            return
        if canonical and block.session_id == canonical[0].session_id:
            canonical.append(block)
            return
        self.prefix_directory.attach_session_to_prefix(
            block.session_id, block.prefix_hash, count_saved=False
        )
        self.prefix_directory.dedup_saved_bytes += block.bytes_uncompressed
        self.movement_analyzer.record_avoided(
            block.bytes_uncompressed,
            multicast=block.fanout_count > 1,
            hbm_residency=True,
            cross_rack=self.placement_policy.name == "distributed-semantic-kv"
            and block.fanout_count >= 16,
        )
        if block.fanout_count > 2:
            gpu = self.topology.gpu_for_session(block.session_id)
            appliance = self.topology.preferred_appliance(gpu.gpu_id)
            cross_rack_ratio = 0.5 if gpu.rack_id != appliance.rack_id else 0.15
            multicast = self.multicast_planner.plan(
                fanout=block.fanout_count,
                bytes_per_replica=block.bytes_uncompressed,
                cross_rack_ratio=cross_rack_ratio,
            )
            self.metrics.multicast_saved_bytes += multicast.avoided_bytes
            self.metrics.avoided_cross_rack_bytes += multicast.avoided_cross_rack_bytes
        self.metrics.compression_saved_bytes += self.prefix_directory.make_dedup_reference(block)

    def _access_block(self, block_id: str, step: int) -> None:
        block = self.blocks.get(block_id)
        if not block:
            return
        block.last_access_step = step
        self.heat_model.apply(block, current_step=step, decode_priority=1.5)
        tier = self.tiers[block.tier]
        latency = tier.latency_us
        if block.tier is not MemoryTier.GPU_HBM:
            self.metrics.hbm_miss_count += 1
            occupancy = tier.used_bytes / tier.capacity_bytes if tier.capacity_bytes else 0.0
            queue_depth = len(tier.stored_block_ids) / 2048
            queue_delay = self.stall_model.queueing_delay_us(
                occupancy,
                queue_depth,
                tier.latency_us,
            )
            dma_transfer = max(0.0, latency - self.tiers[MemoryTier.GPU_HBM].latency_us)
            serialization_penalty = dma_transfer * (1.0 + queue_depth * 0.2)
            fabric_wait = self.stall_model.fabric_wait_us(
                self.topology.congestion_penalty(
                    self.topology.gpu_for_session(block.session_id).gpu_id,
                    self.topology.preferred_appliance(
                        self.topology.gpu_for_session(block.session_id).gpu_id
                    ).appliance_id,
                ),
                serialization_penalty,
            )
            decode_pause = latency * max(0.25, block.decode_priority)
            cache_miss_penalty = latency * max(0.1, 1 - block.attention_importance)
            prefetch_ready = block_id in self.prefetch_scheduler.scheduled
            prefetch_lateness = 0.0 if prefetch_ready else dma_transfer * 0.35
            overlap_discount = self.stall_model.transfer_overlap_discount_us(
                prefetch_ready,
                dma_transfer,
            )
            breakdown = StallBreakdown(
                queue_delay_us=queue_delay,
                fabric_wait_us=fabric_wait,
                dma_transfer_us=dma_transfer,
                decode_pause_us=decode_pause,
                cache_miss_penalty_us=cache_miss_penalty,
                serialization_penalty_us=serialization_penalty,
                prefetch_lateness_penalty_us=prefetch_lateness,
                overlap_discount_us=overlap_discount,
            )
            stall = self.stall_model.record(breakdown)
            self.metrics.queue_delay_us += queue_delay
            self.metrics.fabric_wait_us += fabric_wait
            self.metrics.decode_pause_us += decode_pause
            self.metrics.serialization_penalty_us += serialization_penalty
            self.metrics.prefetch_lateness_penalty_us += prefetch_lateness
            self.metrics.simulated_stall_us += stall
            self._step_stall_us += stall
            self._promote_hot_block_to_hbm(block, step)
        self.reservation_manager.reserve_for_access(block)

    def _promote_hot_block_to_hbm(self, block: KVBlock, step: int) -> None:
        """Promote a decode-hot remote block into HBM when feasible."""

        if not self.reservation_manager.should_force_hbm(block):
            return
        hbm = self.tiers[MemoryTier.GPU_HBM]
        source_tier = block.tier
        self._ensure_capacity(MemoryTier.GPU_HBM, block.bytes_stored, step)
        if not hbm.can_fit(block):
            self.metrics.failure_events.append(
                {
                    "step": step,
                    "component": MemoryTier.GPU_HBM.value,
                    "severity": "warning",
                    "retry_penalty_us": 0.0,
                    "reason": "promotion skipped because HBM remained saturated",
                }
            )
            return
        self.tiers[source_tier].remove_block(block)
        hbm.add_block(block)
        self._record_transfer(source_tier, MemoryTier.GPU_HBM, block.bytes_stored, step)

    def _ensure_capacity(self, tier_name: MemoryTier, required_bytes: int, step: int) -> None:
        tier = self.tiers[tier_name]
        if tier.can_fit(required_bytes):
            return
        candidates = [self.blocks[bid] for bid in tier.stored_block_ids if bid in self.blocks]
        if tier_name is MemoryTier.GPU_HBM:
            candidates = [
                block
                for block in candidates
                if self.reservation_manager.can_demote(
                    block,
                    hbm_used_bytes=tier.used_bytes,
                    hbm_capacity_bytes=tier.capacity_bytes,
                )
            ]
        result = self.eviction_policy.select_victim(candidates, required_bytes, step)
        self.metrics.compression_saved_bytes += sum(
            max(0, block.bytes_uncompressed - block.bytes_stored) for block in result.compressed
        )
        for victim in result.victims:
            tier.remove_block(victim)
            self.metrics.eviction_count += 1
            self.metrics.eviction_class_counts[victim.eviction_class.value] = (
                self.metrics.eviction_class_counts.get(victim.eviction_class.value, 0) + 1
            )
            fallback = (
                MemoryTier.CXL_POOL if tier_name is MemoryTier.GPU_HBM else MemoryTier.NVME_OBJECT
            )
            if victim.pinned_in_hbm and tier_name is MemoryTier.GPU_HBM:
                continue
            if self.tiers[fallback].can_fit(victim):
                self.tiers[fallback].add_block(victim)
                self._record_transfer(tier_name, fallback, victim.bytes_stored, step)
            else:
                self.blocks.pop(victim.block_id, None)

    def _remove_block(self, block_id: str) -> None:
        block = self.blocks.pop(block_id, None)
        if block:
            self.tiers[block.tier].remove_block(block)

    def _spill_target_for_pressure(
        self,
        block: KVBlock,
        preferred_tier: MemoryTier,
    ) -> MemoryTier:
        """Select a degraded-mode target when the preferred hot tier is saturated."""

        candidates = [
            preferred_tier,
            MemoryTier.KV_APPLIANCE,
            MemoryTier.CXL_POOL,
            MemoryTier.NVME_OBJECT,
        ]
        for tier_name in candidates:
            tier = self.tiers[tier_name]
            if tier_name is MemoryTier.GPU_HBM:
                continue
            if tier.can_fit(block):
                return tier_name
        return MemoryTier.NVME_OBJECT

    def _record_occupancy(self, step: int) -> None:
        self.reservation_manager.tracker.refresh(self.blocks)
        self.metrics.hbm_used_peak = max(
            self.metrics.hbm_used_peak, self.tiers[MemoryTier.GPU_HBM].used_bytes
        )
        self.metrics.appliance_used_peak = max(
            self.metrics.appliance_used_peak, self.tiers[MemoryTier.KV_APPLIANCE].used_bytes
        )
        self.metrics.cxl_used_peak = max(
            self.metrics.cxl_used_peak, self.tiers[MemoryTier.CXL_POOL].used_bytes
        )
        self.metrics.nvme_used_peak = max(
            self.metrics.nvme_used_peak, self.tiers[MemoryTier.NVME_OBJECT].used_bytes
        )
        self.metrics.reserved_active_hbm_bytes = (
            self.reservation_manager.tracker.active_decode_bytes
        )
        self.metrics.occupancy_history.append(
            {
                "step": step,
                "GPU_HBM": self.tiers[MemoryTier.GPU_HBM].used_bytes,
                "KV_APPLIANCE": self.tiers[MemoryTier.KV_APPLIANCE].used_bytes,
                "CXL_POOL": self.tiers[MemoryTier.CXL_POOL].used_bytes,
                "NVME_OBJECT": self.tiers[MemoryTier.NVME_OBJECT].used_bytes,
            }
        )
        self.metrics.movement_history.append(
            {"step": step, "bytes_moved_gb": self.metrics.bytes_moved / (1024**3)}
        )
        self.metrics.latency_history.append(
            {
                "step": step,
                "stall_us": self.metrics.simulated_stall_us,
                "p99_us": self.stall_model.summary()["p99_us"],
            }
        )
        self.metrics.dedup_history.append(
            {"step": step, "dedup_saved_gb": self.prefix_directory.dedup_saved_bytes / (1024**3)}
        )
        avg_heat = (
            sum(block.heat_score for block in self.blocks.values()) / len(self.blocks)
            if self.blocks
            else 0.0
        )
        self.metrics.heat_history.append({"step": step, "avg_heat": avg_heat})
        self.metrics.active_hbm_history.append(
            {
                "step": step,
                "active_hbm_gb": self.reservation_manager.tracker.active_decode_bytes / (1024**3),
                "hbm_floor_gb": self.reservation_manager.floor_bytes(
                    self.tiers[MemoryTier.GPU_HBM].capacity_bytes
                )
                / (1024**3),
            }
        )

    def _record_transfer(
        self, source: MemoryTier, target: MemoryTier, moved_bytes: int, step: int
    ) -> None:
        if moved_bytes <= 0:
            return
        cross_rack = source is MemoryTier.NVME_OBJECT or target is MemoryTier.NVME_OBJECT
        self.movement_analyzer.record_move(moved_bytes, source, target, cross_rack=cross_rack)
        self.metrics.bytes_moved += moved_bytes
        self._step_bytes_moved += moved_bytes
        bandwidth = max(self.tiers[target].bandwidth_gbps, 1)
        transfer_us = (moved_bytes / (1024**3)) / bandwidth * 1_000_000
        hop_penalty = TOPOLOGY_HOPS.get((source, target), TOPOLOGY_HOPS.get((target, source), 1.0))
        queue_depth = max(1, len(self.tiers[target].stored_block_ids) // 1024)
        topology_penalty = 0.0
        if target is MemoryTier.KV_APPLIANCE:
            gpu = next(iter(self.topology.gpus))
            appliance = self.topology.preferred_appliance(gpu).appliance_id
            path = self.topology.path_between(gpu, appliance)
            topology_penalty = path.routing_cost(moved_bytes, self.topology.path_utilization(path))
            self.topology.reserve_path(
                path,
                min(0.25, moved_bytes / max(1, self.tiers[target].capacity_bytes)),
            )
        penalty = (
            transfer_us * hop_penalty
            + queue_depth * self.tiers[target].latency_us * 0.02
            + topology_penalty * 0.1
        )
        self.metrics.transfer_penalty_us += penalty
        if self._step_bytes_moved > self.tiers[target].bandwidth_gbps * 1024**3 * 0.001:
            self.metrics.bandwidth_saturation_events += 1
            penalty *= 1.15
        # Background demotion/promotion traffic affects decode less than direct
        # HBM demand misses, but it still contributes some queueing pressure.
        stall_component = penalty if target is MemoryTier.GPU_HBM else penalty * 0.005
        self.metrics.simulated_stall_us += stall_component
        self._step_stall_us += stall_component

    def _queue_delay_us(self, tier_name: MemoryTier) -> float:
        tier = self.tiers[tier_name]
        occupancy = tier.used_bytes / tier.capacity_bytes if tier.capacity_bytes else 0
        queue_depth = len(tier.stored_block_ids) / 2048
        return tier.latency_us * min(8.0, occupancy * 2 + queue_depth)

    def _cool_blocks(self) -> None:
        """Cool blocks that were not accessed this step."""

        for block in self.blocks.values():
            self.heat_model.cool(block)

    def _check_failures(self, step: int) -> None:
        """Apply deterministic degraded-mode events under extreme pressure."""

        hbm = self.tiers[MemoryTier.GPU_HBM]
        hbm_occupancy = hbm.used_bytes / hbm.capacity_bytes if hbm.capacity_bytes else 0.0
        appliance_load = max(
            (appliance.current_load for appliance in self.topology.appliances.values()),
            default=0.0,
        )
        event = self.failure_injector.maybe_trigger(
            step=step,
            hbm_occupancy=hbm_occupancy,
            appliance_load=appliance_load,
            topology_congestion=self.metrics.topology_congestion_score,
        )
        if event is None:
            return
        self.metrics.failure_events.append(
            {
                "step": event.step,
                "component": event.component,
                "severity": event.severity,
                "retry_penalty_us": event.retry_penalty_us,
            }
        )
        self.metrics.simulated_stall_us += event.retry_penalty_us
        if event.emergency_spill:
            self.metrics.bytes_moved += int(hbm.used_bytes * 0.02)

    def _throughput_score(self) -> float:
        stall_penalty = (
            self.metrics.stall_p95_us + self.metrics.stall_p99_us + self.metrics.stall_p999_us
        ) / 3_000_000
        movement_penalty = self.metrics.bytes_moved / (1024**4)
        saturation_penalty = self.metrics.bandwidth_saturation_events / 10_000
        reuse_bonus = self.metrics.prefix_hit_rate + self.metrics.prefetch_success_rate
        active_hbm_bonus = min(
            0.5,
            self.metrics.reserved_active_hbm_bytes
            / max(1, self.tiers[MemoryTier.GPU_HBM].used_bytes + 1),
        )
        return max(
            0.01,
            1.0
            + reuse_bonus
            + active_hbm_bonus
            - stall_penalty
            - movement_penalty
            - saturation_penalty,
        )


def make_eviction_policy(name: str) -> EvictionPolicy:
    """Construct an eviction policy from a user-facing name."""

    normalized = name.lower()
    if normalized == "lru":
        return LRUEviction()
    if normalized == "semantic":
        return SemanticEviction()
    if normalized in {"distributed", "distributed-semantic"}:
        return DistributedSemanticEvictionPolicy()
    raise ValueError(f"Unknown eviction policy: {name}")
