"""Mock connector for TensorRT-LLM-shaped runtime traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_kv.connectors.base import BaseConnector
from semantic_kv.models import ModelProfile
from semantic_kv.traces import Trace


@dataclass
class TensorRTLLMConnector(BaseConnector):
    """Normalize simplified TensorRT-LLM trace rows into `Trace` events."""

    name: str = "tensorrt-llm"

    def from_rows(self, rows: list[dict[str, Any]], model_profile: ModelProfile) -> Trace:
        """Create a trace from mock TensorRT-LLM runtime-shaped rows."""

        return self.to_trace(
            rows,
            workload_name="tensorrt_llm_import",
            model_profile=model_profile,
            topology_profile="runtime-shaped",
            description="Mock TensorRT-LLM connector import.",
            assumptions=["simulation only", "not real TensorRT-LLM integration"],
        )
