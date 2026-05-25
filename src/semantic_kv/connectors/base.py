"""Shared helpers for mock runtime-shaped trace connectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_kv.models import ModelProfile
from semantic_kv.traces import Trace, TraceEvent, TraceEventType


@dataclass
class BaseConnector:
    """Normalize runtime-shaped event rows into the simulator Trace schema."""

    name: str

    def to_trace(
        self,
        rows: list[dict[str, Any]],
        *,
        workload_name: str,
        model_profile: ModelProfile,
        topology_profile: str,
        description: str,
        assumptions: list[str],
    ) -> Trace:
        """Convert runtime-shaped rows into a `Trace`."""

        events = [self._normalize_row(row, model_profile.model_name) for row in rows]
        return Trace(
            events,
            workload_name,
            model_profile,
            topology_profile,
            description,
            assumptions,
        )

    def _normalize_row(self, row: dict[str, Any], default_model_id: str) -> TraceEvent:
        """Normalize one runtime-shaped row into a trace event."""

        event_type = TraceEventType(row["event_type"])
        return TraceEvent(
            step=int(row.get("step", 0)),
            timestamp_us=float(row.get("timestamp_us", row.get("step", 0) * 1000)),
            event_type=event_type,
            session_id=str(row.get("session_id", "unknown")),
            tenant_id=row.get("tenant_id"),
            model_id=row.get("model_id", default_model_id),
            gpu_id=row.get("gpu_id"),
            layer_id=row.get("layer_id"),
            head_id=row.get("head_id"),
            token_start=row.get("token_start"),
            token_count=row.get("token_count"),
            bytes=row.get("bytes"),
            prefix_hash=row.get("prefix_hash"),
            metadata=row.get("metadata", {}),
        )
