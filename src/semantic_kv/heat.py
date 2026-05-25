"""KV heat and locality scoring for migration and eviction decisions."""

from __future__ import annotations

from dataclasses import dataclass

from semantic_kv.models import KVBlock


@dataclass
class HeatModel:
    """Track heat scores that combine recency, reuse, and decode importance."""

    default_cooling_rate: float = 0.08

    def apply(self, block: KVBlock, *, current_step: int, decode_priority: float) -> float:
        """Update and return a block's heat score."""

        age = max(0, current_step - block.last_access_step)
        recent_access_decay = max(0.05, 1.0 - age * block.cooling_rate)
        fanout_score = 1.0 + min(block.fanout_count, 64) / 32
        reuse_probability = max(block.reuse_score, block.attention_importance, 0.05)
        heat = reuse_probability * recent_access_decay * fanout_score * max(0.2, decode_priority)
        block.temporal_locality = recent_access_decay
        block.decode_priority = decode_priority
        block.cooling_rate = block.cooling_rate or self.default_cooling_rate
        block.predicted_reuse_window = max(1, int(128 * reuse_probability * fanout_score))
        block.heat_score = heat
        return heat

    def cool(self, block: KVBlock, steps: int = 1) -> float:
        """Cool a block when it is not accessed."""

        block.heat_score = max(0.0, block.heat_score * max(0.1, 1 - block.cooling_rate * steps))
        return block.heat_score
