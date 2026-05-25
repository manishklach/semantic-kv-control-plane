"""Rack-scale topology model for memory-orchestrated inference simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from semantic_kv.fabric import FabricLink, default_fabric_links


@dataclass(frozen=True, slots=True)
class GPUNode:
    gpu_id: str
    local_hbm_capacity: int
    nvlink_peers: list[str]
    pcie_root: str
    appliance_affinity: str
    rack_id: str


@dataclass(slots=True)
class KVApplianceNode:
    appliance_id: str
    memory_capacity: int
    connected_gpus: list[str]
    bandwidth_limits: dict[str, float]
    current_load: float = 0.0
    rack_id: str = "rack-a"


@dataclass(slots=True)
class RackTopology:
    gpus: dict[str, GPUNode]
    appliances: dict[str, KVApplianceNode]
    cxl_pools: dict[str, int]
    uplinks: dict[str, FabricLink]
    cross_rack_links: dict[tuple[str, str], FabricLink]

    def gpu_for_session(self, session_id: str) -> GPUNode:
        gpus = list(self.gpus.values())
        index = abs(hash(session_id)) % len(gpus)
        return gpus[index]

    def preferred_appliance(self, gpu_id: str) -> KVApplianceNode:
        gpu = self.gpus[gpu_id]
        return self.appliances[gpu.appliance_affinity]

    def movement_latency_us(self, source_gpu_id: str, target_appliance_id: str, bytes_to_move: int) -> float:
        gpu = self.gpus[source_gpu_id]
        appliance = self.appliances[target_appliance_id]
        if gpu.rack_id != appliance.rack_id:
            link = self.cross_rack_links[(gpu.rack_id, appliance.rack_id)]
            return link.routing_cost(bytes_to_move)
        if target_appliance_id == gpu.appliance_affinity:
            return self.uplinks["pcie"].routing_cost(bytes_to_move)
        return self.uplinks["cxl"].routing_cost(bytes_to_move)

    def congestion_penalty(self, gpu_id: str, appliance_id: str) -> float:
        appliance = self.appliances[appliance_id]
        gpu = self.gpus[gpu_id]
        cross_rack = 0.35 if gpu.rack_id != appliance.rack_id else 0.0
        return appliance.current_load + cross_rack

    def reserve_appliance(self, appliance_id: str, load: float) -> None:
        appliance = self.appliances[appliance_id]
        appliance.current_load = min(1.0, appliance.current_load + load)

    def decay(self) -> None:
        for appliance in self.appliances.values():
            appliance.current_load *= 0.90
        for link in [*self.uplinks.values(), *self.cross_rack_links.values()]:
            link.decay()


def default_rack_topology(racks: int = 2, gpus_per_rack: int = 4) -> RackTopology:
    gb = 1024**3
    gpus: dict[str, GPUNode] = {}
    appliances: dict[str, KVApplianceNode] = {}
    cxl_pools: dict[str, int] = {}
    for rack_index in range(racks):
        rack_id = f"rack-{rack_index}"
        appliance_id = f"kvapp-{rack_index}"
        gpu_ids = [f"gpu-{rack_index}-{i}" for i in range(gpus_per_rack)]
        appliances[appliance_id] = KVApplianceNode(
            appliance_id=appliance_id,
            memory_capacity=512 * gb,
            connected_gpus=gpu_ids,
            bandwidth_limits={"local": 800, "remote": 400},
            rack_id=rack_id,
        )
        cxl_pools[f"cxl-{rack_index}"] = 2048 * gb
        for i, gpu_id in enumerate(gpu_ids):
            peers = [peer for peer in gpu_ids if peer != gpu_id]
            gpus[gpu_id] = GPUNode(
                gpu_id=gpu_id,
                local_hbm_capacity=80 * gb,
                nvlink_peers=peers,
                pcie_root=f"pcie-root-{rack_index}",
                appliance_affinity=appliance_id,
                rack_id=rack_id,
            )
    uplinks = default_fabric_links()
    cross = {}
    for a in range(racks):
        for b in range(racks):
            if a != b:
                cross[(f"rack-{a}", f"rack-{b}")] = FabricLink(
                    f"rack-{a}-to-rack-{b}", f"rack-{a}", f"rack-{b}", 400, 120
                )
    return RackTopology(gpus, appliances, cxl_pools, uplinks, cross)


class TopologyVisualizer:
    """Render a compact SVG topology map for docs and dashboard artifacts."""

    def __init__(self, topology: RackTopology) -> None:
        self.topology = topology

    def render_svg(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        racks = sorted({gpu.rack_id for gpu in self.topology.gpus.values()})
        width = 460 * len(racks) + 80
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="440" viewBox="0 0 {width} 440">',
            '<rect width="100%" height="100%" fill="#090d18"/>',
            '<text x="40" y="48" fill="#f8fafc" font-family="Inter,Segoe UI,sans-serif" font-size="28" font-weight="700">Rack-Scale KV Fabric</text>',
        ]
        for rack_i, rack_id in enumerate(racks):
            x = 40 + rack_i * 460
            parts.append(f'<rect x="{x}" y="90" width="400" height="290" rx="10" fill="#111827" stroke="#334155"/>')
            parts.append(f'<text x="{x+24}" y="126" fill="#93c5fd" font-family="Inter,Segoe UI,sans-serif" font-size="18" font-weight="700">{rack_id}</text>')
            rack_gpus = [gpu for gpu in self.topology.gpus.values() if gpu.rack_id == rack_id]
            for idx, gpu in enumerate(rack_gpus):
                gx = x + 24 + (idx % 2) * 170
                gy = 155 + (idx // 2) * 76
                parts.append(f'<rect x="{gx}" y="{gy}" width="135" height="46" rx="6" fill="#0f172a" stroke="#38bdf8"/>')
                parts.append(f'<text x="{gx+14}" y="{gy+29}" fill="#e5e7eb" font-family="Inter,Segoe UI,sans-serif" font-size="13">{gpu.gpu_id}</text>')
            app = next(app for app in self.topology.appliances.values() if app.rack_id == rack_id)
            parts.append(f'<rect x="{x+86}" y="318" width="220" height="44" rx="6" fill="#10201a" stroke="#22c55e"/>')
            parts.append(f'<text x="{x+106}" y="346" fill="#dcfce7" font-family="Inter,Segoe UI,sans-serif" font-size="14">{app.appliance_id} · KV appliance</text>')
        if len(racks) > 1:
            parts.append('<path d="M440 236 L500 236" stroke="#f97316" stroke-width="3" stroke-dasharray="8 6"/>')
            parts.append('<text x="447" y="222" fill="#fed7aa" font-family="Inter,Segoe UI,sans-serif" font-size="13">rack uplink</text>')
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path
