from semantic_kv.eviction import SemanticEviction
from semantic_kv.models import MODEL_PRESETS
from semantic_kv.placement import SemanticKVPolicy
from semantic_kv.simulator import SimulationEngine
from semantic_kv.tiers import default_tier_profiles
from semantic_kv.workloads import shared_prefix_workload


def test_shared_prefix_workload_records_dedup_savings():
    profile = MODEL_PRESETS["llama8b"]
    workload = shared_prefix_workload(profile, sessions=4, context=1024, decode_steps=2)
    engine = SimulationEngine(profile, workload, SemanticKVPolicy(), SemanticEviction(), default_tier_profiles(scale=0.01))
    metrics = engine.run()
    assert metrics.dedup_saved_bytes > 0
    assert metrics.prefix_hit_rate > 0
