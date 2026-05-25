"""Property-style invariants for simulator capacity accounting."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from semantic_kv.eviction import SemanticEviction
from semantic_kv.models import EvictionClass, KVBlock, MemoryTier, ModelProfile
from semantic_kv.placement import SemanticKVPolicy
from semantic_kv.simulator import SimulationEngine
from semantic_kv.tiers import default_tier_profiles
from semantic_kv.workloads import EventType, WorkloadEvent


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=4_096),
            st.sampled_from(list(EvictionClass)),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=25, deadline=None)
def test_total_bytes_never_exceed_total_capacity(allocations) -> None:
    profile = ModelProfile("tiny", 2, 2, 8, 2, 16)
    events: list[WorkloadEvent] = []
    for step, (block_bytes, eviction_class) in enumerate(allocations):
        block = KVBlock(
            block_id=f"b{step}",
            session_id=f"s{step % 3}",
            model_id=profile.model_name,
            layer_id=0,
            head_id=0,
            token_start=step * profile.block_tokens,
            token_count=profile.block_tokens,
            bytes_uncompressed=block_bytes,
            bytes_stored=block_bytes,
            tier=MemoryTier.GPU_HBM,
            prefix_hash="shared" if eviction_class is EvictionClass.REUSABLE_PREFIX else None,
            reuse_score=0.8 if eviction_class is EvictionClass.REUSABLE_PREFIX else 0.1,
            eviction_class=eviction_class,
            last_access_step=step,
            created_step=step,
            fanout_count=4 if eviction_class is EvictionClass.REUSABLE_PREFIX else 0,
        )
        events.append(WorkloadEvent(step=step, event_type=EventType.CREATE_BLOCK, block=block))

    tiers = default_tier_profiles(scale=1e-6)
    engine = SimulationEngine(
        model_profile=profile,
        workload=events,
        placement_policy=SemanticKVPolicy(),
        eviction_policy=SemanticEviction(),
        tier_config=tiers,
    )
    engine.run()

    total_used = sum(tier.used_bytes for tier in engine.tiers.values())
    total_capacity = sum(tier.capacity_bytes for tier in engine.tiers.values())
    assert total_used <= total_capacity
    assert all(tier.used_bytes <= tier.capacity_bytes for tier in engine.tiers.values())
