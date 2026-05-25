from semantic_kv.metadata import PrefixDirectory
from semantic_kv.models import EvictionClass, KVBlock, MemoryTier, ModelProfile


def test_kv_byte_estimation():
    profile = ModelProfile("tiny", 2, 4, 8, 2, 16)
    assert profile.estimate_kv_block_bytes() == 2 * 2 * 4 * 8 * 16 * 2
    assert profile.estimate_session_kv_bytes(32) == 2 * 2 * 4 * 8 * 32 * 2


def test_prefix_dedup_savings():
    block = KVBlock(
        "b1",
        "s1",
        "m",
        0,
        0,
        0,
        16,
        1024,
        1024,
        MemoryTier.GPU_HBM,
        prefix_hash="p",
        eviction_class=EvictionClass.REUSABLE_PREFIX,
    )
    directory = PrefixDirectory()
    directory.register_prefix("p", [block])
    assert directory.lookup_prefix("p") == [block]
    assert directory.attach_session_to_prefix("s2", "p") == 1024
    assert directory.compute_saved_bytes() == 1024
    assert directory.prefix_hit_rate == 1.0
