"""Metadata directories for prefix reuse and deduplication."""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_kv.compression import apply_compression
from semantic_kv.models import CompressionMode, KVBlock


@dataclass
class PrefixDirectory:
    """Track canonical exact-match prefixes and dedup savings."""

    prefixes: dict[str, list[KVBlock]] = field(default_factory=dict)
    attached_sessions: dict[str, set[str]] = field(default_factory=dict)
    prefix_hit_count: int = 0
    prefix_miss_count: int = 0
    dedup_saved_bytes: int = 0

    def register_prefix(self, prefix_hash: str, blocks: list[KVBlock]) -> None:
        self.prefixes[prefix_hash] = blocks
        self.attached_sessions.setdefault(prefix_hash, set())

    def lookup_prefix(self, prefix_hash: str) -> list[KVBlock] | None:
        blocks = self.prefixes.get(prefix_hash)
        if blocks:
            self.prefix_hit_count += 1
            return blocks
        self.prefix_miss_count += 1
        return None

    def attach_session_to_prefix(
        self, session_id: str, prefix_hash: str, count_saved: bool = True
    ) -> int:
        blocks = self.prefixes.get(prefix_hash)
        if not blocks:
            self.prefix_miss_count += 1
            return 0
        sessions = self.attached_sessions.setdefault(prefix_hash, set())
        if session_id in sessions:
            return 0
        sessions.add(session_id)
        saved = sum(block.bytes_uncompressed for block in blocks)
        if count_saved:
            self.dedup_saved_bytes += saved
        for block in blocks:
            block.fanout_count = max(block.fanout_count, len(sessions))
        return saved

    def make_dedup_reference(self, block: KVBlock) -> int:
        return apply_compression(block, CompressionMode.DEDUP_REF)

    def compute_saved_bytes(self) -> int:
        return self.dedup_saved_bytes

    @property
    def prefix_hit_rate(self) -> float:
        total = self.prefix_hit_count + self.prefix_miss_count
        return self.prefix_hit_count / total if total else 0.0


@dataclass
class DistributedPrefixDirectory:
    """Rack-local prefix caches plus a global index.

    The model captures the distributed-systems question: should a shared prefix
    be duplicated per GPU, cached once per rack, or multicast from a canonical
    appliance? It is metadata-only and does not store tensors.
    """

    rack_caches: dict[str, dict[str, list[KVBlock]]] = field(default_factory=dict)
    global_index: dict[str, str] = field(default_factory=dict)
    fanout: dict[str, int] = field(default_factory=dict)
    multicast_saved_bytes: int = 0
    cross_rack_avoided_bytes: int = 0
    duplicate_kv_eliminated: int = 0
    prefix_hit_count: int = 0
    prefix_miss_count: int = 0
    allow_cross_tenant_dedup: bool = False

    def register_prefix(self, rack_id: str, prefix_hash: str, blocks: list[KVBlock]) -> None:
        self.rack_caches.setdefault(rack_id, {})[prefix_hash] = blocks
        self.global_index.setdefault(prefix_hash, rack_id)
        self.fanout.setdefault(prefix_hash, 0)

    def lookup(self, rack_id: str, prefix_hash: str) -> tuple[list[KVBlock] | None, str | None]:
        local = self.rack_caches.get(rack_id, {}).get(prefix_hash)
        if local:
            self.prefix_hit_count += 1
            return local, rack_id
        canonical_rack = self.global_index.get(prefix_hash)
        if canonical_rack:
            self.prefix_hit_count += 1
            return self.rack_caches[canonical_rack][prefix_hash], canonical_rack
        self.prefix_miss_count += 1
        return None, None

    def attach_session(
        self,
        rack_id: str,
        session_id: str,
        prefix_hash: str,
        bytes_per_prefix: int,
        tenant_id: str | None = None,
    ) -> int:
        if tenant_id and not self.allow_cross_tenant_dedup:
            scoped_hash = f"{tenant_id}:{prefix_hash}"
        else:
            scoped_hash = prefix_hash
        _, source_rack = self.lookup(rack_id, scoped_hash)
        if source_rack is None:
            self.global_index[scoped_hash] = rack_id
            self.rack_caches.setdefault(rack_id, {})[scoped_hash] = []
            return 0
        self.fanout[scoped_hash] = self.fanout.get(scoped_hash, 0) + 1
        self.duplicate_kv_eliminated += bytes_per_prefix
        if source_rack == rack_id:
            self.multicast_saved_bytes += bytes_per_prefix
        else:
            self.cross_rack_avoided_bytes += bytes_per_prefix
        return bytes_per_prefix

    @property
    def prefix_hit_rate(self) -> float:
        total = self.prefix_hit_count + self.prefix_miss_count
        return self.prefix_hit_count / total if total else 0.0
