"""Synthetic workload generators for the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from semantic_kv.models import EvictionClass, KVBlock, MemoryTier, ModelProfile


class EventType(str, Enum):
    CREATE_BLOCK = "CREATE_BLOCK"
    ACCESS_BLOCK = "ACCESS_BLOCK"
    PREFETCH_REQUEST = "PREFETCH_REQUEST"
    RELEASE_BLOCK = "RELEASE_BLOCK"
    SESSION_END = "SESSION_END"


@dataclass(frozen=True)
class WorkloadEvent:
    step: int
    event_type: EventType
    block: KVBlock | None = None
    block_id: str | None = None
    session_id: str | None = None


def _block(
    profile: ModelProfile,
    session_id: str,
    block_index: int,
    step: int,
    eviction_class: EvictionClass,
    prefix_hash: str | None = None,
    tenant_id: str | None = None,
    fanout_count: int = 0,
) -> KVBlock:
    bytes_uncompressed = profile.estimate_kv_block_bytes()
    block_id = f"{session_id}:b{block_index}"
    return KVBlock(
        block_id=block_id,
        session_id=session_id,
        model_id=profile.model_name,
        layer_id=-1,
        head_id=-1,
        token_start=block_index * profile.block_tokens,
        token_count=profile.block_tokens,
        bytes_uncompressed=bytes_uncompressed,
        bytes_stored=bytes_uncompressed,
        tier=MemoryTier.GPU_HBM,
        prefix_hash=prefix_hash,
        reuse_score=0.8 if prefix_hash else 0.2,
        eviction_class=eviction_class,
        last_access_step=step,
        created_step=step,
        fanout_count=fanout_count,
        tenant_id=tenant_id,
    )


def basic_decode_workload(
    profile: ModelProfile, sessions: int, context: int, decode_steps: int
) -> list[WorkloadEvent]:
    events: list[WorkloadEvent] = []
    blocks_per_session = max(1, context // profile.block_tokens)
    for s in range(sessions):
        session_id = f"s{s}"
        for i in range(blocks_per_session):
            klass = EvictionClass.HOT_ACTIVE if i >= blocks_per_session - 2 else EvictionClass.SESSION_RECENT
            events.append(WorkloadEvent(i, EventType.CREATE_BLOCK, _block(profile, session_id, i, i, klass)))
    for step in range(blocks_per_session, blocks_per_session + decode_steps):
        for s in range(sessions):
            events.append(
                WorkloadEvent(step, EventType.ACCESS_BLOCK, block_id=f"s{s}:b{blocks_per_session - 1}")
            )
    return events


def shared_prefix_workload(
    profile: ModelProfile, sessions: int, context: int, decode_steps: int
) -> list[WorkloadEvent]:
    events: list[WorkloadEvent] = []
    blocks_per_session = max(1, context // profile.block_tokens)
    prefix_blocks = max(1, int(blocks_per_session * 0.5))
    prefix_hash = "shared-system-prompt-v1"
    for s in range(sessions):
        session_id = f"s{s}"
        for i in range(blocks_per_session):
            if i < prefix_blocks:
                klass = EvictionClass.REUSABLE_PREFIX
                phash = prefix_hash
                fanout = sessions
            else:
                klass = EvictionClass.HOT_ACTIVE if i >= blocks_per_session - 2 else EvictionClass.SESSION_RECENT
                phash = None
                fanout = 0
            events.append(WorkloadEvent(i, EventType.CREATE_BLOCK, _block(profile, session_id, i, i, klass, phash, fanout_count=fanout)))
    for step in range(blocks_per_session, blocks_per_session + decode_steps):
        for s in range(sessions):
            events.append(WorkloadEvent(step, EventType.ACCESS_BLOCK, block_id=f"s{s}:b{blocks_per_session - 1}"))
            if step % 32 == 0:
                events.append(WorkloadEvent(step, EventType.PREFETCH_REQUEST, session_id=f"s{s}"))
    return events


def agentic_tool_workload(
    profile: ModelProfile, sessions: int, context: int, decode_steps: int
) -> list[WorkloadEvent]:
    events = shared_prefix_workload(profile, sessions, context, decode_steps)
    base = max(1, context // profile.block_tokens)
    for s in range(sessions):
        session_id = f"s{s}"
        for i in range(3):
            events.append(
                WorkloadEvent(
                    base + i,
                    EventType.CREATE_BLOCK,
                    _block(profile, session_id, base + i, base + i, EvictionClass.EPHEMERAL_TOOL_CALL),
                )
            )
    return sorted(events, key=lambda e: e.step)


class AgenticWorkflowWorkload:
    """Planner/executor sessions with tool loops and branch divergence."""

    def generate(self, profile: ModelProfile, sessions: int, context: int, decode_steps: int) -> list[WorkloadEvent]:
        events = agentic_tool_workload(profile, sessions, context, decode_steps)
        base = max(1, context // profile.block_tokens)
        for s in range(sessions):
            session_id = f"agent{s}"
            for branch in range(2):
                events.append(
                    WorkloadEvent(
                        base + branch,
                        EventType.CREATE_BLOCK,
                        _block(
                            profile,
                            session_id,
                            base + branch,
                            base + branch,
                            EvictionClass.LOW_ATTENTION if branch else EvictionClass.SESSION_RECENT,
                            prefix_hash="agent-shared-memory",
                            fanout_count=max(2, sessions // 2),
                        ),
                    )
                )
        return sorted(events, key=lambda event: event.step)


class MultiTenantInferenceWorkload:
    """Competing tenants with isolation-scoped prefixes and QoS pressure."""

    def generate(self, profile: ModelProfile, sessions: int, context: int, decode_steps: int) -> list[WorkloadEvent]:
        events: list[WorkloadEvent] = []
        blocks_per_session = max(1, context // profile.block_tokens)
        for s in range(sessions):
            tenant_id = f"tenant-{s % 8}"
            session_id = f"{tenant_id}:s{s}"
            for i in range(blocks_per_session):
                prefix = f"{tenant_id}:policy-prefix" if i < blocks_per_session // 3 else None
                klass = EvictionClass.REUSABLE_PREFIX if prefix else EvictionClass.SESSION_RECENT
                events.append(
                    WorkloadEvent(
                        i,
                        EventType.CREATE_BLOCK,
                        _block(profile, session_id, i, i, klass, prefix, tenant_id, fanout_count=sessions // 8),
                    )
                )
        return sorted(events + basic_decode_workload(profile, sessions, profile.block_tokens, decode_steps), key=lambda e: e.step)


class SharedEnterprisePromptWorkload:
    """Large enterprise policy/RAG prompt shared by many users."""

    def generate(self, profile: ModelProfile, sessions: int, context: int, decode_steps: int) -> list[WorkloadEvent]:
        return shared_prefix_workload(profile, sessions, int(context * 1.5), decode_steps)


class MultiAgentCollaborationWorkload:
    """Agents share partial memory while diverging into role-specific context."""

    def generate(self, profile: ModelProfile, sessions: int, context: int, decode_steps: int) -> list[WorkloadEvent]:
        events = shared_prefix_workload(profile, sessions, context, decode_steps)
        blocks_per_session = max(1, context // profile.block_tokens)
        for s in range(sessions):
            for i in range(blocks_per_session // 4, blocks_per_session // 2):
                session_id = f"collab{s}"
                events.append(
                    WorkloadEvent(
                        i,
                        EventType.CREATE_BLOCK,
                        _block(
                            profile,
                            session_id,
                            i,
                            i,
                            EvictionClass.REUSABLE_PREFIX,
                            prefix_hash=f"shared-agent-memory-{s % 4}",
                            fanout_count=max(2, sessions // 4),
                        ),
                    )
                )
        return sorted(events, key=lambda e: e.step)


def long_context_workload(
    profile: ModelProfile, sessions: int, context: int, decode_steps: int
) -> list[WorkloadEvent]:
    return basic_decode_workload(profile, max(1, min(sessions, 4)), context * 2, decode_steps)


def mixed_tenant_workload(
    profile: ModelProfile, sessions: int, context: int, decode_steps: int
) -> list[WorkloadEvent]:
    events: list[WorkloadEvent] = []
    blocks_per_session = max(1, context // profile.block_tokens)
    for s in range(sessions):
        tenant_id = f"tenant-{s % 4}"
        session_id = f"{tenant_id}:s{s}"
        for i in range(blocks_per_session):
            klass = EvictionClass.SESSION_COLD if i < blocks_per_session // 2 else EvictionClass.SESSION_RECENT
            events.append(
                WorkloadEvent(
                    i,
                    EventType.CREATE_BLOCK,
                    _block(profile, session_id, i, i, klass, tenant_id=tenant_id),
                )
            )
    return events + basic_decode_workload(profile, sessions, profile.block_tokens, decode_steps)


def agentic_workflow_workload(profile: ModelProfile, sessions: int, context: int, decode_steps: int) -> list[WorkloadEvent]:
    return AgenticWorkflowWorkload().generate(profile, sessions, context, decode_steps)


def multi_tenant_inference_workload(profile: ModelProfile, sessions: int, context: int, decode_steps: int) -> list[WorkloadEvent]:
    return MultiTenantInferenceWorkload().generate(profile, sessions, context, decode_steps)


def shared_enterprise_prompt_workload(profile: ModelProfile, sessions: int, context: int, decode_steps: int) -> list[WorkloadEvent]:
    return SharedEnterprisePromptWorkload().generate(profile, sessions, context, decode_steps)


def multi_agent_collaboration_workload(profile: ModelProfile, sessions: int, context: int, decode_steps: int) -> list[WorkloadEvent]:
    return MultiAgentCollaborationWorkload().generate(profile, sessions, context, decode_steps)


def make_workload(
    name: str, profile: ModelProfile, sessions: int, context: int, decode_steps: int
) -> list[WorkloadEvent]:
    normalized = name.lower().replace("_", "-")
    generators = {
        "basic": basic_decode_workload,
        "basic-decode": basic_decode_workload,
        "shared-prefix": shared_prefix_workload,
        "agentic-tool": agentic_tool_workload,
        "long-context": long_context_workload,
        "mixed-tenant": mixed_tenant_workload,
        "multi-tenant": mixed_tenant_workload,
        "mixed-load": mixed_tenant_workload,
        "agentic-workflow": agentic_workflow_workload,
        "multi-tenant-inference": multi_tenant_inference_workload,
        "shared-enterprise-prompt": shared_enterprise_prompt_workload,
        "enterprise-prompt": shared_enterprise_prompt_workload,
        "multi-agent-collaboration": multi_agent_collaboration_workload,
    }
    if normalized not in generators:
        raise ValueError(f"Unknown workload: {name}")
    return sorted(generators[normalized](profile, sessions, context, decode_steps), key=lambda e: e.step)
