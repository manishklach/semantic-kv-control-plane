"""Prefix fanout and multicast savings modeling for shared KV delivery."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MulticastDecision:
    """Represent one multicast fanout decision and its modeled savings."""

    fanout: int
    bytes_per_replica: int
    tree_depth: int
    avoided_bytes: int
    avoided_dma_ops: int
    avoided_cross_rack_bytes: int


@dataclass
class MulticastPlanner:
    """Estimate multicast savings for high-fanout reusable prefixes."""

    decisions: list[MulticastDecision] = field(default_factory=list)

    def plan(
        self,
        *,
        fanout: int,
        bytes_per_replica: int,
        cross_rack_ratio: float = 0.0,
    ) -> MulticastDecision:
        """Estimate multicast savings versus point-to-point delivery."""

        if fanout <= 1:
            decision = MulticastDecision(fanout, bytes_per_replica, 0, 0, 0, 0)
            self.decisions.append(decision)
            return decision
        tree_depth = 1 if fanout <= 8 else 2
        avoided_replica_transfers = max(0, fanout - tree_depth - 1)
        avoided_bytes = avoided_replica_transfers * bytes_per_replica
        avoided_cross_rack = int(avoided_bytes * cross_rack_ratio)
        decision = MulticastDecision(
            fanout=fanout,
            bytes_per_replica=bytes_per_replica,
            tree_depth=tree_depth,
            avoided_bytes=avoided_bytes,
            avoided_dma_ops=avoided_replica_transfers,
            avoided_cross_rack_bytes=avoided_cross_rack,
        )
        self.decisions.append(decision)
        return decision
