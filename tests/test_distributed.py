from semantic_kv.fabric import FabricLink
from semantic_kv.metadata import DistributedPrefixDirectory
from semantic_kv.models import EvictionClass, KVBlock, MemoryTier
from semantic_kv.movement import MovementAnalyzer
from semantic_kv.placement import DistributedSemanticKVPolicy, TopologyAwareSemanticPolicy
from semantic_kv.semantic_prefetch import PredictivePrefetchEngine, PrefetchPriority
from semantic_kv.tiers import default_tier_profiles
from semantic_kv.topology import default_rack_topology


def _block(klass=EvictionClass.REUSABLE_PREFIX, fanout=32):
    return KVBlock(
        "b",
        "s1",
        "m",
        0,
        0,
        0,
        128,
        1024,
        1024,
        MemoryTier.GPU_HBM,
        prefix_hash="p",
        eviction_class=klass,
        fanout_count=fanout,
    )


def test_fabric_routing_cost_increases_with_congestion():
    link = FabricLink("rack", "a", "b", bandwidth_gbps=100, latency_us=10)
    base = link.routing_cost(1024**3)
    link.utilization = 0.9
    link.queue_depth = 100
    assert link.routing_cost(1024**3) > base


def test_topology_prefers_local_appliance():
    topology = default_rack_topology()
    gpu = topology.gpu_for_session("session-a")
    appliance = topology.preferred_appliance(gpu.gpu_id)
    assert gpu.gpu_id in appliance.connected_gpus


def test_topology_aware_policy_places_prefix_on_appliance():
    decision = TopologyAwareSemanticPolicy().choose_tier(
        _block(), default_tier_profiles(scale=0.01)
    )
    assert decision.target_tier is MemoryTier.KV_APPLIANCE
    assert "prefix" in decision.reason


def test_distributed_policy_mentions_multicast_anchor():
    decision = DistributedSemanticKVPolicy().choose_tier(
        _block(fanout=64), default_tier_profiles(scale=0.01)
    )
    assert decision.target_tier is MemoryTier.KV_APPLIANCE
    assert "multicast" in decision.reason


def test_distributed_prefix_directory_tracks_avoided_bytes():
    directory = DistributedPrefixDirectory()
    directory.register_prefix("rack-0", "p", [_block()])
    saved = directory.attach_session("rack-1", "s2", "p", 2048)
    assert saved == 2048
    assert directory.duplicate_kv_eliminated == 2048
    assert directory.cross_rack_avoided_bytes == 2048


def test_predictive_prefetch_prioritizes_tool_loop():
    engine = PredictivePrefetchEngine()
    plan = engine.plan(
        "s", 128, decode_velocity_tps=20, prefix_reuse_probability=0.1, active_tool_loop=True
    )
    assert plan.priority is PrefetchPriority.URGENT


def test_movement_energy_accounting():
    analyzer = MovementAnalyzer()
    analyzer.record_move(1024, MemoryTier.GPU_HBM, MemoryTier.CXL_POOL)
    analyzer.record_avoided(2048, multicast=True, cross_rack=True)
    assert analyzer.stats.movement_energy_j > 0
    assert analyzer.stats.multicast_saved_bytes == 2048
