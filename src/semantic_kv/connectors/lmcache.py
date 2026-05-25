"""Mock connector for LMCache-shaped runtime traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_kv.connectors.base import BaseConnector
from semantic_kv.models import ModelProfile
from semantic_kv.traces import Trace


@dataclass
class LMCacheConnector(BaseConnector):
    """Normalize simplified LMCache rows into the simulator Trace schema."""

    name: str = "lmcache"

    def from_rows(self, rows: list[dict[str, Any]], model_profile: ModelProfile) -> Trace:
        """Create a trace from mock LMCache runtime-shaped rows."""

        return self.to_trace(
            rows,
            workload_name="lmcache_import",
            model_profile=model_profile,
            topology_profile="runtime-shaped",
            description="Mock LMCache connector import.",
            assumptions=["simulation only", "not real LMCache integration"],
        )
