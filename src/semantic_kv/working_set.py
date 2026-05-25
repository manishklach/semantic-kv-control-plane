"""Active decode working-set tracking and HBM residency reservations.

These helpers enforce a key realism constraint: even in a distributed KV
system, some decode-critical state must remain resident in HBM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_kv.models import KVBlock, MemoryTier


@dataclass
class ActiveDecodeWindow:
    """Track the currently decoding sessions, layers, and token windows."""

    current_layers: dict[str, set[int]] = field(default_factory=dict)
    current_tokens: dict[str, int] = field(default_factory=dict)
    active_sessions: set[str] = field(default_factory=set)
    decode_pressure_score: float = 0.0

    def update_access(self, block: KVBlock) -> None:
        """Record that a block is on the active decode path."""

        self.active_sessions.add(block.session_id)
        self.current_tokens[block.session_id] = max(
            self.current_tokens.get(block.session_id, 0),
            block.token_start + block.token_count,
        )
        self.current_layers.setdefault(block.session_id, set()).add(block.layer_id)
        total_layers = sum(len(layers) for layers in self.current_layers.values())
        self.decode_pressure_score = (
            len(self.active_sessions) + total_layers + sum(self.current_tokens.values()) / 4096
        )

    def release_session(self, session_id: str) -> None:
        """Remove a session from the active decode window."""

        self.active_sessions.discard(session_id)
        self.current_tokens.pop(session_id, None)
        self.current_layers.pop(session_id, None)
        total_layers = sum(len(layers) for layers in self.current_layers.values())
        self.decode_pressure_score = (
            len(self.active_sessions) + total_layers + sum(self.current_tokens.values()) / 4096
        )

    def is_active_block(self, block: KVBlock) -> bool:
        """Return whether a block belongs to an active decode window."""

        if block.session_id not in self.active_sessions:
            return False
        current_token = self.current_tokens.get(block.session_id, 0)
        return block.token_start + block.token_count >= max(
            0,
            current_token - block.token_count * 2,
        )


@dataclass
class DecodeResidencyTracker:
    """Track decode-hot bytes and which blocks are protected in HBM."""

    active_window: ActiveDecodeWindow = field(default_factory=ActiveDecodeWindow)
    protected_blocks: set[str] = field(default_factory=set)
    active_decode_bytes: int = 0

    def mark_access(self, block: KVBlock) -> None:
        """Mark a block as part of the decode-hot working set."""

        self.active_window.update_access(block)
        if block.tier is MemoryTier.GPU_HBM:
            self.protected_blocks.add(block.block_id)
            self.active_decode_bytes += block.bytes_stored
            block.pinned_in_hbm = True
        else:
            block.pinned_in_hbm = False

    def release_session(self, session_id: str, blocks: dict[str, KVBlock]) -> None:
        """Release all protected blocks for a finished session."""

        self.active_window.release_session(session_id)
        for block_id, block in list(blocks.items()):
            if block.session_id == session_id and block_id in self.protected_blocks:
                self.protected_blocks.discard(block_id)
                self.active_decode_bytes = max(0, self.active_decode_bytes - block.bytes_stored)
                block.pinned_in_hbm = False

    def refresh(self, blocks: dict[str, KVBlock]) -> None:
        """Recompute protected active-decode bytes from current blocks."""

        self.active_decode_bytes = 0
        for block_id in list(self.protected_blocks):
            block = blocks.get(block_id)
            if block is None:
                self.protected_blocks.discard(block_id)
                continue
            if self.active_window.is_active_block(block) and block.tier is MemoryTier.GPU_HBM:
                self.active_decode_bytes += block.bytes_stored
                block.pinned_in_hbm = True
            else:
                self.protected_blocks.discard(block_id)
                block.pinned_in_hbm = False


@dataclass
class HBMReservationManager:
    """Enforce a minimum active HBM floor for decode-critical KV."""

    active_hbm_floor: float = 0.15
    tracker: DecodeResidencyTracker = field(default_factory=DecodeResidencyTracker)

    def reserve_for_access(self, block: KVBlock) -> None:
        """Pin an accessed block into the active decode reservation set."""

        self.tracker.mark_access(block)

    def can_demote(
        self,
        block: KVBlock,
        *,
        hbm_used_bytes: int,
        hbm_capacity_bytes: int,
        projected_hbm_bytes: int | None = None,
    ) -> bool:
        """Return whether a block may leave HBM without violating the floor."""

        if block.pinned_in_hbm or block.block_id in self.tracker.protected_blocks:
            return False
        if projected_hbm_bytes is None:
            projected_hbm_bytes = max(0, hbm_used_bytes - block.bytes_stored)
        reserved_floor = int(hbm_capacity_bytes * self.active_hbm_floor)
        active_requirement = max(reserved_floor, self.tracker.active_decode_bytes)
        return projected_hbm_bytes >= active_requirement

    def should_force_hbm(self, block: KVBlock) -> bool:
        """Return whether a block should remain in HBM due to decode pressure."""

        return block.pinned_in_hbm or self.tracker.active_window.is_active_block(block)

    def floor_bytes(self, hbm_capacity_bytes: int) -> int:
        """Return the configured HBM floor in bytes."""

        return int(hbm_capacity_bytes * self.active_hbm_floor)
