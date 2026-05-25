"""Tests for working-set, stall, failure, and connector realism layers."""

from __future__ import annotations

from semantic_kv.connectors import LMCacheConnector, TensorRTLLMConnector, VLLMConnector
from semantic_kv.failures import FailureInjector
from semantic_kv.heat import HeatModel
from semantic_kv.models import MODEL_PRESETS, EvictionClass, KVBlock, MemoryTier
from semantic_kv.stalls import StallBreakdown, StallModel
from semantic_kv.topology import default_rack_topology
from semantic_kv.working_set import HBMReservationManager


def _block(block_id: str = "b0") -> KVBlock:
    return KVBlock(
        block_id=block_id,
        session_id="session-1",
        model_id="model",
        layer_id=1,
        head_id=0,
        token_start=0,
        token_count=128,
        bytes_uncompressed=1024,
        bytes_stored=1024,
        tier=MemoryTier.GPU_HBM,
        eviction_class=EvictionClass.HOT_ACTIVE,
        reuse_score=0.8,
        fanout_count=16,
    )


def test_active_hbm_floor_protects_decode_hot_blocks() -> None:
    manager = HBMReservationManager(active_hbm_floor=0.15)
    block = _block()
    manager.reserve_for_access(block)
    can_demote = manager.can_demote(
        block,
        hbm_used_bytes=10_000,
        hbm_capacity_bytes=80_000,
        projected_hbm_bytes=8_000,
    )
    assert can_demote is False


def test_stall_model_reports_percentiles() -> None:
    model = StallModel()
    for scale in [1, 2, 3, 4, 5]:
        model.record(
            StallBreakdown(
                queue_delay_us=scale,
                fabric_wait_us=scale,
                dma_transfer_us=scale,
                decode_pause_us=scale,
                cache_miss_penalty_us=scale,
                serialization_penalty_us=scale,
                prefetch_lateness_penalty_us=scale,
            )
        )
    summary = model.summary()
    assert summary["p99_us"] >= summary["p50_us"]
    assert summary["p999_us"] >= summary["p99_us"]


def test_heat_model_updates_block_heat() -> None:
    block = _block()
    heat = HeatModel().apply(block, current_step=4, decode_priority=1.2)
    assert heat > 0
    assert block.predicted_reuse_window > 0


def test_failure_injector_triggers_under_pressure() -> None:
    event = FailureInjector().maybe_trigger(
        step=10,
        hbm_occupancy=0.99,
        appliance_load=0.5,
        topology_congestion=0.2,
    )
    assert event is not None
    assert event.emergency_spill is True


def test_topology_path_contains_multiple_edges() -> None:
    topology = default_rack_topology(racks=2, gpus_per_rack=4)
    gpu = topology.gpu_for_session("session-a")
    appliance = topology.preferred_appliance(gpu.gpu_id)
    path = topology.path_between(gpu.gpu_id, appliance.appliance_id)
    assert len(path.edges) >= 3
    assert path.bandwidth_gbps > 0


def test_mock_connectors_normalize_runtime_rows() -> None:
    rows = [
        {
            "event_type": "KV_ALLOC",
            "step": 1,
            "session_id": "s1",
            "token_count": 128,
            "bytes": 2048,
            "metadata": {"block_id": "s1:b1"},
        }
    ]
    profile = MODEL_PRESETS["llama8b"]
    traces = [
        VLLMConnector().from_rows(rows, profile),
        TensorRTLLMConnector().from_rows(rows, profile),
        LMCacheConnector().from_rows(rows, profile),
    ]
    assert all(trace.events for trace in traces)
    assert all(trace.events[0].metadata["block_id"] == "s1:b1" for trace in traces)
