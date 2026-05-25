from semantic_kv.compression import apply_compression
from semantic_kv.eviction import LRUEviction, SemanticEviction
from semantic_kv.models import CompressionMode, EvictionClass, KVBlock, MemoryTier


def _block(block_id, last_access, klass):
    return KVBlock(
        block_id,
        "s",
        "m",
        0,
        0,
        0,
        1,
        100,
        100,
        MemoryTier.GPU_HBM,
        prefix_hash="p" if klass is EvictionClass.REUSABLE_PREFIX else None,
        eviction_class=klass,
        last_access_step=last_access,
        fanout_count=20 if klass is EvictionClass.REUSABLE_PREFIX else 0,
    )


def test_lru_evicts_oldest():
    blocks = [_block("old", 1, EvictionClass.SESSION_RECENT), _block("new", 10, EvictionClass.SESSION_RECENT)]
    result = LRUEviction().select_victim(blocks, 100, 20)
    assert result.victims[0].block_id == "old"


def test_semantic_eviction_protects_reusable_prefixes():
    prefix = _block("prefix", 1, EvictionClass.REUSABLE_PREFIX)
    ephemeral = _block("tool", 10, EvictionClass.EPHEMERAL_TOOL_CALL)
    result = SemanticEviction().select_victim([prefix, ephemeral], 100, 20)
    assert [victim.block_id for victim in result.victims] == ["tool"]


def test_compression_changes_stored_bytes():
    block = _block("low", 1, EvictionClass.LOW_ATTENTION)
    saved = apply_compression(block, CompressionMode.BLOCK_QUANT_SIM)
    assert block.bytes_stored == 35
    assert saved == 65
