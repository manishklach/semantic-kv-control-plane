"""Attention-aware importance modeling for synthetic KV management."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionEstimate:
    """Represent simulated attention importance and recompute preference."""

    importance_score: float
    recompute_worthiness: float
    attention_density: float


@dataclass
class AttentionImportanceEstimator:
    """Estimate whether KV is important enough to keep or cheap enough to recompute."""

    def estimate(
        self,
        *,
        layer_id: int,
        token_age: int,
        attention_density: float,
        session_type: str,
    ) -> AttentionEstimate:
        """Estimate attention importance for a synthetic block."""

        layer_weight = 1.2 if layer_id < 0 else max(0.35, 1.0 - layer_id / 120)
        age_decay = max(0.1, 1.0 - token_age / 8192)
        session_bias = {
            "agentic": 1.1,
            "shared-prefix": 1.0,
            "long-context": 0.7,
            "tool-call": 0.45,
        }.get(session_type, 0.85)
        importance = max(0.05, attention_density * layer_weight * age_decay * session_bias)
        recompute = max(0.0, 1.1 - importance - min(token_age / 16384, 0.6))
        return AttentionEstimate(importance, recompute, attention_density)
