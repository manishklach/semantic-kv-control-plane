"""Rack-scale topology model for memory-orchestrated inference simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from semantic_kv.fabric import FabricLink, default_fabric_links


@dataclass(frozen=True, slots=True)
class GPUNode:
    """Describe one GPU participant in the simulated memory fabric."""

    gpu_id: str
    local_hbm_capacity: int
    nvlink_peers: list[str]
    pcie_root: str
    appliance_affinity: str
    rack_id: str
    numa_domain: str
    nvlink_island: str


@dataclass(slots=True)
class KVApplianceNode:
    """Describe a rack-local KV appliance tier."""

    appliance_id: str
    memory_capacity: int
    connected_gpus: list[str]
    bandwidth_limits: dict[str, float]
    current_load: float = 0.0
    rack_id: str = "rack-a"


@dataclass(frozen=True, slots=True)
class TopologyEdge:
    """Represent one weighted connectivity edge in the simulated fabric graph."""

    source: str
    target: str
    bandwidth_gbps: float
    latency_us: float
    oversubscription: float = 1.0
    link_type: str = "fabric"

    def routing_cost(self, bytes_to_move: int, utilization: float = 0.0) -> float:
        """Estimate route cost across the edge."""

        transfer_us = bytes_to_move / (1024**3) / max(self.bandwidth_gbps, 1e-9) * 1_000_000
        return self.latency_us + transfer_us * (1.0 + utilization) * self.oversubscription


@dataclass(frozen=True, slots=True)
class NetworkPath:
    """Represent one end-to-end topology path."""

    edges: tuple[TopologyEdge, ...]

    @property
    def bandwidth_gbps(self) -> float:
        """Return the bottleneck bandwidth on the path."""

        if not self.edges:
            return 0.0
        return min(edge.bandwidth_gbps / edge.oversubscription for edge in self.edges)

    @property
    def latency_us(self) -> float:
        """Return the sum of fixed latencies across the path."""

        return sum(edge.latency_us for edge in self.edges)

    @property
    def oversubscription(self) -> float:
        """Return the average oversubscription along the path."""

        if not self.edges:
            return 1.0
        return sum(edge.oversubscription for edge in self.edges) / len(self.edges)

    def routing_cost(self, bytes_to_move: int, utilization: float = 0.0) -> float:
        """Estimate end-to-end routing cost across the path."""

        return sum(edge.routing_cost(bytes_to_move, utilization) for edge in self.edges)


@dataclass(slots=True)
class RackTopology:
    """Describe GPUs, appliances, and links participating in one topology."""

    gpus: dict[str, GPUNode]
    appliances: dict[str, KVApplianceNode]
    cxl_pools: dict[str, int]
    uplinks: dict[str, FabricLink]
    cross_rack_links: dict[tuple[str, str], FabricLink]
    edges: list[TopologyEdge] = field(default_factory=list)
    leaf_spines: dict[str, tuple[str, str]] = field(default_factory=dict)
    edge_utilization: dict[tuple[str, str], float] = field(default_factory=dict)

    def gpu_for_session(self, session_id: str) -> GPUNode:
        """Select a deterministic GPU for a given session identifier."""

        gpus = list(self.gpus.values())
        index = abs(hash(session_id)) % len(gpus)
        return gpus[index]

    def preferred_appliance(self, gpu_id: str) -> KVApplianceNode:
        """Return the rack-local appliance preferred for a GPU."""

        gpu = self.gpus[gpu_id]
        return self.appliances[gpu.appliance_affinity]

    def path_between(self, source_gpu_id: str, target_appliance_id: str) -> NetworkPath:
        """Return a synthetic ECMP-style path from a GPU to an appliance."""

        gpu = self.gpus[source_gpu_id]
        appliance = self.appliances[target_appliance_id]
        root_edge = TopologyEdge(
            source_gpu_id,
            gpu.pcie_root,
            bandwidth_gbps=64,
            latency_us=8,
            oversubscription=1.0 if gpu.numa_domain.endswith("0") else 1.05,
            link_type="pcie-tree",
        )
        if gpu.rack_id == appliance.rack_id:
            leaf, spine = self.leaf_spines[gpu.rack_id]
            return NetworkPath(
                (
                    root_edge,
                    TopologyEdge(
                        gpu.pcie_root,
                        leaf,
                        bandwidth_gbps=128,
                        latency_us=12,
                        oversubscription=1.0,
                        link_type="leaf",
                    ),
                    TopologyEdge(
                        leaf,
                        appliance.appliance_id,
                        bandwidth_gbps=800,
                        latency_us=8,
                        oversubscription=1.0 + appliance.current_load,
                        link_type="appliance-uplink",
                    ),
                )
            )
        link = self.cross_rack_links[(gpu.rack_id, appliance.rack_id)]
        local_leaf, local_spine = self.leaf_spines[gpu.rack_id]
        remote_leaf, _ = self.leaf_spines[appliance.rack_id]
        return NetworkPath(
            (
                root_edge,
                TopologyEdge(
                    gpu.pcie_root,
                    local_leaf,
                    bandwidth_gbps=128,
                    latency_us=12,
                    oversubscription=1.0,
                    link_type="leaf",
                ),
                TopologyEdge(
                    local_leaf,
                    local_spine,
                    bandwidth_gbps=400,
                    latency_us=18,
                    oversubscription=1.2,
                    link_type="leaf-spine",
                ),
                TopologyEdge(
                    local_spine,
                    remote_leaf,
                    bandwidth_gbps=link.bandwidth_gbps,
                    latency_us=link.latency_us,
                    oversubscription=1.4,
                    link_type="cross-rack",
                ),
                TopologyEdge(
                    remote_leaf,
                    appliance.appliance_id,
                    bandwidth_gbps=400,
                    latency_us=10,
                    oversubscription=1.0 + appliance.current_load,
                    link_type="appliance-uplink",
                ),
            )
        )

    def movement_latency_us(
        self, source_gpu_id: str, target_appliance_id: str, bytes_to_move: int
    ) -> float:
        """Estimate KV movement latency from a GPU to an appliance."""

        path = self.path_between(source_gpu_id, target_appliance_id)
        utilization = self.path_utilization(path)
        return path.routing_cost(bytes_to_move, utilization)

    def path_utilization(self, path: NetworkPath) -> float:
        """Return the average utilization across a path."""

        if not path.edges:
            return 0.0
        values = [
            self.edge_utilization.get((edge.source, edge.target), 0.0) for edge in path.edges
        ]
        return sum(values) / len(values)

    def reserve_path(self, path: NetworkPath, load: float) -> None:
        """Reserve path utilization on every edge."""

        for edge in path.edges:
            key = (edge.source, edge.target)
            self.edge_utilization[key] = min(1.0, self.edge_utilization.get(key, 0.0) + load)

    def congestion_penalty(self, gpu_id: str, appliance_id: str) -> float:
        """Return a congestion penalty for a routing choice."""

        appliance = self.appliances[appliance_id]
        gpu = self.gpus[gpu_id]
        path = self.path_between(gpu_id, appliance_id)
        cross_rack = 0.35 if gpu.rack_id != appliance.rack_id else 0.0
        numa_penalty = 0.08 if gpu.numa_domain.endswith("1") else 0.0
        return appliance.current_load + cross_rack + path.oversubscription * 0.05 + numa_penalty

    def reserve_appliance(self, appliance_id: str, load: float) -> None:
        """Reserve a small fraction of appliance load for a new placement."""

        appliance = self.appliances[appliance_id]
        appliance.current_load = min(1.0, appliance.current_load + load)

    def decay(self) -> None:
        """Decay outstanding appliance and link load over time."""

        for appliance in self.appliances.values():
            appliance.current_load *= 0.90
        for key, value in list(self.edge_utilization.items()):
            self.edge_utilization[key] = value * 0.84
        for link in [*self.uplinks.values(), *self.cross_rack_links.values()]:
            link.decay()


def default_rack_topology(racks: int = 2, gpus_per_rack: int = 4) -> RackTopology:
    """Construct a default leaf-spine multi-rack topology for simulation runs."""

    gb = 1024**3
    gpus: dict[str, GPUNode] = {}
    appliances: dict[str, KVApplianceNode] = {}
    cxl_pools: dict[str, int] = {}
    leaf_spines: dict[str, tuple[str, str]] = {}
    edges: list[TopologyEdge] = []
    for rack_index in range(racks):
        rack_id = f"rack-{rack_index}"
        leaf_spines[rack_id] = (f"leaf-{rack_index}", f"spine-{rack_index % 2}")
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
                numa_domain=f"numa-{rack_index}-{i % 2}",
                nvlink_island=f"island-{rack_index}-{i // max(1, gpus_per_rack // 2)}",
            )
            for peer in peers:
                if gpu_id < peer:
                    edges.append(
                        TopologyEdge(
                            gpu_id,
                            peer,
                            bandwidth_gbps=900,
                            latency_us=2,
                            oversubscription=1.0,
                            link_type="nvlink",
                        )
                    )
    uplinks = default_fabric_links()
    cross = {}
    for a in range(racks):
        for b in range(racks):
            if a != b:
                cross[(f"rack-{a}", f"rack-{b}")] = FabricLink(
                    f"rack-{a}-to-rack-{b}",
                    f"rack-{a}",
                    f"rack-{b}",
                    400,
                    120,
                    retry_penalty_us=250,
                )
    return RackTopology(gpus, appliances, cxl_pools, uplinks, cross, edges, leaf_spines)


class TopologyVisualizer:
    """Render a compact SVG topology map for docs and dashboard artifacts."""

    def __init__(self, topology: RackTopology) -> None:
        self.topology = topology

    def render_svg(self, path: Path) -> Path:
        """Render a compact SVG view of the current rack topology."""

        path.parent.mkdir(parents=True, exist_ok=True)
        racks = sorted({gpu.rack_id for gpu in self.topology.gpus.values()})
        width = 460 * len(racks) + 80
        parts = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="460" viewBox="0 0 {width} 460">'
            ),
            '<rect width="100%" height="100%" fill="#090d18"/>',
            (
                '<text x="40" y="48" fill="#f8fafc" '
                'font-family="Inter,Segoe UI,sans-serif" font-size="28" '
                'font-weight="700">Leaf-Spine KV Fabric</text>'
            ),
        ]
        for rack_i, rack_id in enumerate(racks):
            x = 40 + rack_i * 460
            leaf, spine = self.topology.leaf_spines[rack_id]
            parts.append(
                f'<rect x="{x}" y="90" width="400" height="320" rx="10" '
                'fill="#111827" stroke="#334155"/>'
            )
            parts.append(
                f'<text x="{x + 24}" y="126" fill="#93c5fd" '
                'font-family="Inter,Segoe UI,sans-serif" font-size="18" '
                f'font-weight="700">{rack_id}</text>'
            )
            parts.append(
                f'<text x="{x + 280}" y="126" fill="#fbbf24" '
                'font-family="Inter,Segoe UI,sans-serif" font-size="13">'
                f"{leaf} → {spine}</text>"
            )
            rack_gpus = [gpu for gpu in self.topology.gpus.values() if gpu.rack_id == rack_id]
            for idx, gpu in enumerate(rack_gpus):
                gx = x + 24 + (idx % 2) * 170
                gy = 155 + (idx // 2) * 76
                parts.append(
                    f'<rect x="{gx}" y="{gy}" width="145" height="50" rx="6" '
                    'fill="#0f172a" stroke="#38bdf8"/>'
                )
                parts.append(
                    f'<text x="{gx + 12}" y="{gy + 23}" fill="#e5e7eb" '
                    'font-family="Inter,Segoe UI,sans-serif" font-size="13">'
                    f"{gpu.gpu_id}</text>"
                )
                parts.append(
                    f'<text x="{gx + 12}" y="{gy + 39}" fill="#94a3b8" '
                    'font-family="Inter,Segoe UI,sans-serif" font-size="10">'
                    f"{gpu.numa_domain} · {gpu.nvlink_island}</text>"
                )
            app = next(app for app in self.topology.appliances.values() if app.rack_id == rack_id)
            parts.append(
                f'<rect x="{x + 86}" y="346" width="220" height="44" rx="6" '
                'fill="#10201a" stroke="#22c55e"/>'
            )
            parts.append(
                f'<text x="{x + 106}" y="374" fill="#dcfce7" '
                'font-family="Inter,Segoe UI,sans-serif" font-size="14">'
                f"{app.appliance_id} · KV appliance</text>"
            )
        if len(racks) > 1:
            parts.append(
                '<path d="M440 256 L500 256" stroke="#f97316" '
                'stroke-width="3" stroke-dasharray="8 6"/>'
            )
            parts.append(
                '<text x="447" y="242" fill="#fed7aa" '
                'font-family="Inter,Segoe UI,sans-serif" font-size="13">'
                'oversubscribed spine uplink</text>'
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path
