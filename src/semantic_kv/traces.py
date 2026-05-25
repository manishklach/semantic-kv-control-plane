"""Trace model and replay support for synthetic KV-cache experiments.

The trace schema is runtime-neutral. It is designed to be easy to export from
synthetic generators today and from vLLM/TensorRT-LLM/LMCache-style traces later
without coupling the simulator to a real serving runtime.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from semantic_kv.eviction import DistributedSemanticEvictionPolicy, LRUEviction, SemanticEviction
from semantic_kv.models import EvictionClass, KVBlock, MemoryTier, ModelProfile
from semantic_kv.placement import (
    CXLSpillPolicy,
    DistributedSemanticKVPolicy,
    NaiveHBMPolicy,
    PlacementPolicy,
    SemanticKVPolicy,
    TopologyAwareSemanticPolicy,
)
from semantic_kv.simulator import SimulationEngine
from semantic_kv.tiers import default_tier_profiles
from semantic_kv.workloads import EventType, WorkloadEvent


class TraceEventType(StrEnum):
    """Event kinds supported by the runtime-neutral replay schema."""

    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"
    KV_ALLOC = "KV_ALLOC"
    KV_ACCESS = "KV_ACCESS"
    KV_PREFETCH = "KV_PREFETCH"
    KV_EVICT = "KV_EVICT"
    PREFIX_LOOKUP = "PREFIX_LOOKUP"
    PREFIX_HIT = "PREFIX_HIT"
    PREFIX_MISS = "PREFIX_MISS"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_END = "TOOL_CALL_END"
    TENANT_SWITCH = "TENANT_SWITCH"
    DECODE_STEP = "DECODE_STEP"


@dataclass(frozen=True)
class TraceEvent:
    """Represent a single timestamped event in a replayable trace."""

    step: int
    timestamp_us: float
    event_type: TraceEventType
    session_id: str
    tenant_id: str | None = None
    model_id: str | None = None
    gpu_id: str | None = None
    layer_id: int | None = None
    head_id: int | None = None
    token_start: int | None = None
    token_count: int | None = None
    bytes: int | None = None
    prefix_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trace:
    """Bundle events and metadata for a reproducible simulation trace."""

    events: list[TraceEvent]
    workload_name: str
    model_profile: ModelProfile
    topology_profile: str = "single-rack"
    description: str = ""
    assumptions: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Return the human-readable workload name."""

        return self.workload_name

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "__trace_header__": True,
            "workload_name": self.workload_name,
            "model_profile": asdict(self.model_profile),
            "topology_profile": self.topology_profile,
            "description": self.description,
            "assumptions": self.assumptions,
        }
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header) + "\n")
            for event in self.events:
                row = asdict(event)
                row["event_type"] = event.event_type.value
                handle.write(json.dumps(row) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path, name: str | None = None) -> Trace:
        events: list[TraceEvent] = []
        header: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("__trace_header__"):
                    header = row
                    continue
                row["event_type"] = TraceEventType(row["event_type"])
                events.append(TraceEvent(**row))
        profile_row = header.get("model_profile") or {
            "model_name": "llama70b-gqa",
            "num_layers": 80,
            "num_kv_heads": 8,
            "head_dim": 128,
            "dtype_bytes": 2,
            "block_tokens": 128,
        }
        return cls(
            events=events,
            workload_name=name or header.get("workload_name", path.stem),
            model_profile=ModelProfile(**profile_row),
            topology_profile=header.get("topology_profile", "unknown"),
            description=header.get("description", ""),
            assumptions=header.get("assumptions", []),
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for event in self.events:
            row = asdict(event)
            row["event_type"] = event.event_type.value
            rows.append(row)
        return pd.DataFrame(rows)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.events:
            errors.append("trace has no events")
        for index, event in enumerate(self.events):
            if event.step < 0:
                errors.append(f"event {index} has negative step")
            if event.timestamp_us < 0:
                errors.append(f"event {index} has negative timestamp")
            if (
                event.event_type in {TraceEventType.KV_ALLOC, TraceEventType.KV_ACCESS}
                and not event.session_id
            ):
                errors.append(f"event {index} missing session_id")
            if event.event_type is TraceEventType.KV_ALLOC and not event.token_count:
                errors.append(f"event {index} allocation missing token_count")
        return errors

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        sessions = set()
        tenants = set()
        total_bytes = 0
        for event in self.events:
            by_type[event.event_type.value] = by_type.get(event.event_type.value, 0) + 1
            sessions.add(event.session_id)
            if event.tenant_id:
                tenants.add(event.tenant_id)
            total_bytes += event.bytes or 0
        return {
            "workload_name": self.workload_name,
            "events": len(self.events),
            "sessions": len(sessions),
            "tenants": len(tenants),
            "allocated_gb": total_bytes / (1024**3),
            "event_types": by_type,
            "assumptions": self.assumptions,
        }


POLICY_MATRIX: dict[str, tuple[str, PlacementPolicy, object]] = {
    "naive": ("Naive HBM + LRU", NaiveHBMPolicy(), LRUEviction()),
    "cxl": ("Generic CXL Spill + LRU", CXLSpillPolicy(), LRUEviction()),
    "semantic": ("Single-node Semantic KV", SemanticKVPolicy(), SemanticEviction()),
    "topology-aware": (
        "Topology-aware Semantic KV",
        TopologyAwareSemanticPolicy(),
        SemanticEviction(),
    ),
    "distributed-semantic": (
        "Distributed Semantic KV",
        DistributedSemanticKVPolicy(),
        DistributedSemanticEvictionPolicy(),
    ),
}


class TraceReplayEngine:
    """Replay a trace through placement and eviction policies."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    def replay(self, policy: str = "distributed-semantic"):
        label, placement, eviction = POLICY_MATRIX[policy]
        workload = trace_to_workload_events(self.trace)
        metrics = SimulationEngine(
            self.trace.model_profile,
            workload,
            placement,
            eviction,
            default_tier_profiles(),
            active_hbm_floor=0.15,
        ).run()
        return label, metrics

    def replay_all(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for policy in POLICY_MATRIX:
            label, metrics = self.replay(policy)
            row = metrics.as_row(label)
            row["workload"] = self.trace.workload_name
            rows.append(row)
        return rows


def trace_to_workload_events(trace: Trace) -> list[WorkloadEvent]:
    """Convert runtime-neutral trace events into simulator workload events."""

    events: list[WorkloadEvent] = []
    profile = trace.model_profile
    for event in trace.events:
        if event.event_type is TraceEventType.KV_ALLOC:
            token_count = event.token_count or profile.block_tokens
            bytes_uncompressed = event.bytes or profile.estimate_kv_block_bytes(token_count)
            block_id = (
                event.metadata.get("block_id")
                or f"{event.session_id}:b{event.token_start or event.step}"
            )
            eviction_class = EvictionClass(
                event.metadata.get("eviction_class", EvictionClass.SESSION_RECENT.value)
            )
            block = KVBlock(
                block_id=block_id,
                session_id=event.session_id,
                model_id=event.model_id or profile.model_name,
                layer_id=event.layer_id if event.layer_id is not None else -1,
                head_id=event.head_id if event.head_id is not None else -1,
                token_start=event.token_start or 0,
                token_count=token_count,
                bytes_uncompressed=bytes_uncompressed,
                bytes_stored=bytes_uncompressed,
                tier=MemoryTier.GPU_HBM,
                prefix_hash=event.prefix_hash,
                reuse_score=float(
                    event.metadata.get("reuse_score", 0.8 if event.prefix_hash else 0.2)
                ),
                eviction_class=eviction_class,
                last_access_step=event.step,
                created_step=event.step,
                fanout_count=int(event.metadata.get("fanout_count", 0)),
                tenant_id=event.tenant_id,
            )
            events.append(WorkloadEvent(event.step, EventType.CREATE_BLOCK, block=block))
        elif event.event_type in {TraceEventType.KV_ACCESS, TraceEventType.DECODE_STEP}:
            block_id = (
                event.metadata.get("block_id") or f"{event.session_id}:b{event.token_start or 0}"
            )
            events.append(
                WorkloadEvent(
                    event.step,
                    EventType.ACCESS_BLOCK,
                    block_id=block_id,
                    session_id=event.session_id,
                )
            )
        elif event.event_type is TraceEventType.KV_PREFETCH:
            events.append(
                WorkloadEvent(event.step, EventType.PREFETCH_REQUEST, session_id=event.session_id)
            )
        elif event.event_type is TraceEventType.SESSION_END:
            events.append(
                WorkloadEvent(event.step, EventType.SESSION_END, session_id=event.session_id)
            )
    return sorted(events, key=lambda item: item.step)


def synthetic_trace_from_steps(name: str, steps: int, sessions: int) -> Trace:
    """Create a minimal synthetic trace directly from session and step counts."""

    profile = ModelProfile("llama8b", 32, 8, 128, 2, 128)
    events: list[TraceEvent] = []
    for session in range(sessions):
        session_id = f"s{session}"
        events.append(
            TraceEvent(
                0,
                0,
                TraceEventType.SESSION_START,
                session_id=session_id,
                model_id=profile.model_name,
            )
        )
        for step in range(steps):
            block_id = f"{session_id}:b{step}"
            events.append(
                TraceEvent(
                    step,
                    step * 1000,
                    TraceEventType.KV_ALLOC,
                    session_id=session_id,
                    model_id=profile.model_name,
                    token_start=step * profile.block_tokens,
                    token_count=profile.block_tokens,
                    bytes=profile.estimate_kv_block_bytes(),
                    metadata={"block_id": block_id},
                )
            )
            events.append(
                TraceEvent(
                    step + 1,
                    (step + 1) * 1000,
                    TraceEventType.KV_ACCESS,
                    session_id=session_id,
                    model_id=profile.model_name,
                    token_start=step * profile.block_tokens,
                    metadata={"block_id": block_id},
                )
            )
        events.append(
            TraceEvent(
                steps + 1,
                (steps + 1) * 1000,
                TraceEventType.SESSION_END,
                session_id=session_id,
                model_id=profile.model_name,
            )
        )
    return Trace(events, name, profile, "synthetic", "minimal synthetic trace", ["simulation only"])
