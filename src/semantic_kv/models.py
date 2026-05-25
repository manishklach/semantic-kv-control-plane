"""Core data model for the semantic KV simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryTier(str, Enum):
    GPU_HBM = "GPU_HBM"
    KV_APPLIANCE = "KV_APPLIANCE"
    CXL_POOL = "CXL_POOL"
    NVME_OBJECT = "NVME_OBJECT"


class EvictionClass(str, Enum):
    HOT_ACTIVE = "HOT_ACTIVE"
    REUSABLE_PREFIX = "REUSABLE_PREFIX"
    SESSION_RECENT = "SESSION_RECENT"
    SESSION_COLD = "SESSION_COLD"
    SAFE_TO_RECOMPUTE = "SAFE_TO_RECOMPUTE"
    LOW_ATTENTION = "LOW_ATTENTION"
    EPHEMERAL_TOOL_CALL = "EPHEMERAL_TOOL_CALL"


class CompressionMode(str, Enum):
    NONE = "NONE"
    FP8_SIM = "FP8_SIM"
    INT8_SIM = "INT8_SIM"
    BLOCK_QUANT_SIM = "BLOCK_QUANT_SIM"
    DEDUP_REF = "DEDUP_REF"


@dataclass(slots=True)
class KVBlock:
    block_id: str
    session_id: str
    model_id: str
    layer_id: int
    head_id: int
    token_start: int
    token_count: int
    bytes_uncompressed: int
    bytes_stored: int
    tier: MemoryTier
    prefix_hash: str | None = None
    reuse_score: float = 0.0
    eviction_class: EvictionClass = EvictionClass.SESSION_RECENT
    last_access_step: int = 0
    created_step: int = 0
    compressed: bool = False
    compression_mode: CompressionMode = CompressionMode.NONE
    fanout_count: int = 0
    tenant_id: str | None = None

    @property
    def recompute_cost(self) -> float:
        return self.bytes_uncompressed / max(self.token_count, 1)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int
    block_tokens: int

    def estimate_kv_block_bytes(self, tokens: int | None = None) -> int:
        token_count = tokens or self.block_tokens
        return int(
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * token_count
            * self.dtype_bytes
        )

    def estimate_session_kv_bytes(self, context_tokens: int) -> int:
        return self.estimate_kv_block_bytes(context_tokens)


MODEL_PRESETS: dict[str, ModelProfile] = {
    "llama70b-gqa": ModelProfile(
        model_name="llama70b-gqa",
        num_layers=80,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
        block_tokens=128,
    ),
    "llama8b": ModelProfile(
        model_name="llama8b",
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
        block_tokens=128,
    ),
}
