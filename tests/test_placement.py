import pytest

from semantic_kv.models import EvictionClass, KVBlock, MemoryTier
from semantic_kv.placement import CXLSpillPolicy, SemanticKVPolicy
from semantic_kv.tiers import MemoryTierState, default_tier_profiles


def _block(size=100, klass=EvictionClass.SESSION_RECENT, fanout=0):
    return KVBlock(
        "b",
        "s",
        "m",
        0,
        0,
        0,
        1,
        size,
        size,
        MemoryTier.GPU_HBM,
        eviction_class=klass,
        fanout_count=fanout,
    )


def test_memory_tier_capacity():
    tier = MemoryTierState(MemoryTier.GPU_HBM, 100, 1, 1)
    block = _block(size=80)
    assert tier.can_fit(block)
    tier.add_block(block)
    assert tier.used_bytes == 80
    assert tier.occupancy_pct() == pytest.approx(80)
    assert not tier.can_fit(_block(size=30))


def test_semantic_policy_places_reusable_prefix_on_appliance():
    block = _block(klass=EvictionClass.REUSABLE_PREFIX, fanout=10)
    decision = SemanticKVPolicy().choose_tier(block, default_tier_profiles(scale=0.001))
    assert decision.target_tier is MemoryTier.KV_APPLIANCE


def test_cxl_spill_keeps_hot_active_in_hbm():
    block = _block(klass=EvictionClass.HOT_ACTIVE)
    decision = CXLSpillPolicy().choose_tier(block, default_tier_profiles(scale=0.001))
    assert decision.target_tier is MemoryTier.GPU_HBM
