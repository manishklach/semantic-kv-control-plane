from semantic_kv.eviction import LRUEviction
from semantic_kv.models import MODEL_PRESETS
from semantic_kv.placement import NaiveHBMPolicy
from semantic_kv.simulator import SimulationEngine
from semantic_kv.tiers import default_tier_profiles
from semantic_kv.workloads import basic_decode_workload


def test_simulator_produces_metrics():
    profile = MODEL_PRESETS["llama8b"]
    workload = basic_decode_workload(profile, sessions=2, context=512, decode_steps=4)
    engine = SimulationEngine(
        profile, workload, NaiveHBMPolicy(), LRUEviction(), default_tier_profiles(scale=0.01)
    )
    metrics = engine.run()
    assert metrics.total_kv_created_bytes > 0
    assert metrics.hbm_used_peak > 0
    assert metrics.estimated_throughput_score > 0
    assert metrics.occupancy_history
