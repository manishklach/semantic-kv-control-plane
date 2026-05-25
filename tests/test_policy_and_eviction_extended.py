"""Extended policy, eviction, and prefix directory behavior tests."""

from __future__ import annotations

from semantic_kv.eviction import DistributedSemanticEvictionPolicy, LRUEviction, SemanticEviction
from semantic_kv.metadata import PrefixDirectory
from semantic_kv.models import CompressionMode, EvictionClass, KVBlock, MemoryTier
from semantic_kv.placement import (
    CXLSpillPolicy,
    DistributedSemanticKVPolicy,
    NaiveHBMPolicy,
    SemanticKVPolicy,
    TopologyAwareSemanticPolicy,
)
from semantic_kv.tiers import default_tier_profiles


def _block(
    block_id: str,
    *,
    bytes_stored: int = 256,
    klass: EvictionClass = EvictionClass.SESSION_RECENT,
    last_access: int = 0,
    fanout: int = 0,
    prefix_hash: str | None = None,
    tier: MemoryTier = MemoryTier.GPU_HBM,
) -> KVBlock:
    return KVBlock(
        block_id=block_id,
        session_id="session-1",
        model_id="model",
        layer_id=0,
        head_id=0,
        token_start=0,
        token_count=128,
        bytes_uncompressed=bytes_stored,
        bytes_stored=bytes_stored,
        tier=tier,
        prefix_hash=prefix_hash,
        reuse_score=0.9 if prefix_hash else 0.2,
        eviction_class=klass,
        last_access_step=last_access,
        created_step=0,
        fanout_count=fanout,
    )


def test_naive_hbm_policy_places_everything_in_hbm() -> None:
    decision = NaiveHBMPolicy().choose_tier(_block("b0"), default_tier_profiles(scale=0.001))
    assert decision.target_tier is MemoryTier.GPU_HBM


def test_cxl_spill_places_non_hot_blocks_in_cxl() -> None:
    decision = CXLSpillPolicy().choose_tier(
        _block("b0", klass=EvictionClass.SESSION_RECENT),
        default_tier_profiles(scale=0.001),
    )
    assert decision.target_tier is MemoryTier.CXL_POOL


def test_semantic_policy_places_safe_to_recompute_in_nvme() -> None:
    decision = SemanticKVPolicy().choose_tier(
        _block("b0", klass=EvictionClass.SAFE_TO_RECOMPUTE),
        default_tier_profiles(scale=0.001),
    )
    assert decision.target_tier is MemoryTier.NVME_OBJECT


def test_topology_policy_keeps_hot_active_in_hbm() -> None:
    decision = TopologyAwareSemanticPolicy().choose_tier(
        _block("b0", klass=EvictionClass.HOT_ACTIVE),
        default_tier_profiles(scale=0.001),
    )
    assert decision.target_tier is MemoryTier.GPU_HBM
    assert "avoid fabric hop" in decision.reason


def test_distributed_policy_anchors_high_fanout_prefixes() -> None:
    decision = DistributedSemanticKVPolicy().choose_tier(
        _block(
            "b0",
            klass=EvictionClass.REUSABLE_PREFIX,
            fanout=32,
            prefix_hash="shared",
        ),
        default_tier_profiles(scale=0.001),
    )
    assert decision.target_tier is MemoryTier.KV_APPLIANCE
    assert "multicast anchor" in decision.reason


def test_semantic_eviction_compresses_low_attention_before_evicting() -> None:
    low_attention = _block("cold", klass=EvictionClass.LOW_ATTENTION, bytes_stored=1_000)
    result = SemanticEviction().select_victim([low_attention], required_bytes=500, current_step=10)
    assert result.victims == []
    assert result.compressed == [low_attention]
    assert low_attention.compression_mode is CompressionMode.BLOCK_QUANT_SIM
    assert low_attention.compressed is True


def test_lru_orders_victims_by_access_time() -> None:
    blocks = [
        _block("oldest", last_access=1),
        _block("middle", last_access=2),
        _block("newest", last_access=3),
    ]
    result = LRUEviction().select_victim(blocks, required_bytes=512, current_step=10)
    assert [victim.block_id for victim in result.victims] == ["oldest", "middle"]


def test_distributed_eviction_migrates_reusable_prefix_before_eviction() -> None:
    prefix = _block(
        "prefix",
        klass=EvictionClass.REUSABLE_PREFIX,
        fanout=64,
        prefix_hash="shared",
        tier=MemoryTier.GPU_HBM,
    )
    result = DistributedSemanticEvictionPolicy().select_victim([prefix], 128, current_step=10)
    assert result.victims == []
    assert result.compressed == [prefix]
    assert prefix.tier is MemoryTier.KV_APPLIANCE


def test_prefix_directory_tracks_hits_saved_bytes_and_fanout() -> None:
    canonical = _block(
        "canonical",
        klass=EvictionClass.REUSABLE_PREFIX,
        fanout=1,
        prefix_hash="shared",
        bytes_stored=512,
    )
    directory = PrefixDirectory()
    directory.register_prefix("shared", [canonical])
    assert directory.lookup_prefix("shared") == [canonical]
    saved = directory.attach_session_to_prefix("session-2", "shared")
    assert saved == canonical.bytes_uncompressed
    assert directory.compute_saved_bytes() == canonical.bytes_uncompressed
    assert canonical.fanout_count >= 1
    assert directory.prefix_hit_rate == 1.0
