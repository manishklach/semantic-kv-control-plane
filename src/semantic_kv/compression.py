"""Compression accounting for simulated KV blocks."""

from __future__ import annotations

from semantic_kv.models import CompressionMode, KVBlock

COMPRESSION_RATIOS: dict[CompressionMode, float] = {
    CompressionMode.NONE: 1.0,
    CompressionMode.FP8_SIM: 0.5,
    CompressionMode.INT8_SIM: 0.5,
    CompressionMode.BLOCK_QUANT_SIM: 0.35,
    CompressionMode.DEDUP_REF: 0.02,
}

QUALITY_RISK: dict[CompressionMode, str] = {
    CompressionMode.NONE: "none",
    CompressionMode.FP8_SIM: "low",
    CompressionMode.INT8_SIM: "medium",
    CompressionMode.BLOCK_QUANT_SIM: "high",
    CompressionMode.DEDUP_REF: "none_if_exact_prefix_match",
}


def compressed_size(bytes_uncompressed: int, mode: CompressionMode) -> int:
    return max(1, int(bytes_uncompressed * COMPRESSION_RATIOS[mode]))


def apply_compression(block: KVBlock, mode: CompressionMode) -> int:
    before = block.bytes_stored
    block.compression_mode = mode
    block.compressed = mode is not CompressionMode.NONE
    block.bytes_stored = compressed_size(block.bytes_uncompressed, mode)
    return max(0, before - block.bytes_stored)
