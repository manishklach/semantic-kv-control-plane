"""Lightweight fabric congestion model for rack-scale KV movement.

This module is deliberately not a packet simulator. It gives placement and
movement accounting a topology-sensitive cost model: bandwidth, baseline
latency, queue depth, utilization, simulated loss, and retry penalties.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FabricLink:
    link_id: str
    source: str
    target: str
    bandwidth_gbps: float
    latency_us: float
    utilization: float = 0.0
    queue_depth: int = 0
    packet_loss: float = 0.0
    retry_penalty_us: float = 0.0

    def routing_cost(self, bytes_to_move: int) -> float:
        """Return simulated movement latency in microseconds."""

        gb = bytes_to_move / (1024**3)
        transfer_us = gb / max(self.bandwidth_gbps, 1e-9) * 1_000_000
        congestion = 1.0 + min(4.0, self.utilization**2 + self.queue_depth / 128)
        retry = self.packet_loss * self.retry_penalty_us
        return self.latency_us + transfer_us * congestion + retry

    def reserve(self, bytes_to_move: int) -> float:
        """Account for movement on the link and return routing cost."""

        cost = self.routing_cost(bytes_to_move)
        gb = bytes_to_move / (1024**3)
        self.utilization = min(1.0, self.utilization + gb / max(self.bandwidth_gbps, 1))
        self.queue_depth += 1
        return cost

    def decay(self) -> None:
        self.utilization *= 0.85
        self.queue_depth = max(0, self.queue_depth - 1)


def default_fabric_links() -> dict[str, FabricLink]:
    return {
        "nvlink": FabricLink("nvlink", "gpu", "gpu", 900, 2),
        "pcie": FabricLink("pcie", "gpu", "root", 64, 8, retry_penalty_us=20),
        "cxl": FabricLink("cxl", "root", "cxl", 256, 40, retry_penalty_us=50),
        "rack-uplink": FabricLink(
            "rack-uplink", "rack", "rack", 400, 120, packet_loss=0.0005, retry_penalty_us=250
        ),
    }
