"""Failure and degradation modeling for inference memory infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FailureEvent:
    """Represent one synthetic failure or degraded-mode transition."""

    step: int
    component: str
    severity: str
    retry_penalty_us: float
    emergency_spill: bool = False


@dataclass
class FailureInjector:
    """Track rare synthetic failures and overload-driven degradation."""

    failure_events: list[FailureEvent] = field(default_factory=list)

    def maybe_trigger(
        self,
        *,
        step: int,
        hbm_occupancy: float,
        appliance_load: float,
        topology_congestion: float,
    ) -> FailureEvent | None:
        """Trigger a deterministic degraded event when pressure is extreme."""

        if hbm_occupancy > 0.98:
            event = FailureEvent(step, "GPU_HBM", "hbm_exhaustion", 40.0, emergency_spill=True)
        elif appliance_load > 0.92:
            event = FailureEvent(
                step,
                "KV_APPLIANCE",
                "appliance_overload",
                25.0,
                emergency_spill=True,
            )
        elif topology_congestion > 0.88:
            event = FailureEvent(step, "FABRIC", "congestion_collapse", 18.0)
        else:
            return None
        self.failure_events.append(event)
        return event
