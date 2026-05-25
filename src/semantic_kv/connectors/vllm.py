"""Mock connector for vLLM-shaped runtime traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_kv.connectors.base import BaseConnector
from semantic_kv.models import ModelProfile
from semantic_kv.traces import Trace


@dataclass
class VLLMConnector(BaseConnector):
    """Normalize simplified vLLM-style rows into the simulator Trace schema."""

    name: str = "vllm"

    def from_rows(self, rows: list[dict[str, Any]], model_profile: ModelProfile) -> Trace:
        """Create a trace from mock vLLM runtime-shaped rows."""

        return self.to_trace(
            rows,
            workload_name="vllm_import",
            model_profile=model_profile,
            topology_profile="runtime-shaped",
            description="Mock vLLM connector import.",
            assumptions=["simulation only", "not real vLLM integration"],
        )
