"""Synthetic trace generators for reproducible research scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from semantic_kv.models import EvictionClass, ModelProfile
from semantic_kv.traces import Trace, TraceEvent, TraceEventType


DEFAULT_PROFILE = ModelProfile("llama70b-gqa", 80, 8, 128, 2, 128)


@dataclass
class TraceGeneratorConfig:
    sessions: int = 100
    decode_steps: int = 128
    tenants: int = 1
    racks: int = 2
    gpus_per_rack: int = 4


class BaseTraceGenerator:
    workload_name = "base"
    description = ""
    expected_behavior = ""

    def __init__(self, profile: ModelProfile = DEFAULT_PROFILE) -> None:
        self.profile = profile

    def export(self, trace: Trace, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        jsonl = directory / f"{trace.workload_name}.jsonl"
        md = directory / f"{trace.workload_name}.md"
        trace.to_jsonl(jsonl)
        md.write_text(
            f"# {trace.workload_name}\n\n{trace.description}\n\n"
            "## Expected semantic advantage\n\n"
            f"{self.expected_behavior}\n\n"
            "## Assumptions\n\n"
            + "\n".join(f"- {item}" for item in trace.assumptions)
            + "\n",
            encoding="utf-8",
        )
        return jsonl, md

    def _alloc(
        self,
        step: int,
        session_id: str,
        block_index: int,
        token_count: int,
        eviction_class: EvictionClass,
        tenant_id: str | None = None,
        prefix_hash: str | None = None,
        fanout_count: int = 0,
        gpu_id: str | None = None,
    ) -> TraceEvent:
        return TraceEvent(
            step=step,
            timestamp_us=step * 1000,
            event_type=TraceEventType.KV_ALLOC,
            session_id=session_id,
            tenant_id=tenant_id,
            model_id=self.profile.model_name,
            gpu_id=gpu_id,
            token_start=block_index * self.profile.block_tokens,
            token_count=token_count,
            bytes=self.profile.estimate_kv_block_bytes(token_count),
            prefix_hash=prefix_hash,
            metadata={
                "block_id": f"{session_id}:b{block_index}",
                "eviction_class": eviction_class.value,
                "fanout_count": fanout_count,
            },
        )

    def _access(self, step: int, session_id: str, block_index: int) -> TraceEvent:
        return TraceEvent(
            step,
            step * 1000,
            TraceEventType.KV_ACCESS,
            session_id=session_id,
            model_id=self.profile.model_name,
            token_start=block_index * self.profile.block_tokens,
            metadata={"block_id": f"{session_id}:b{block_index}"},
        )


class SharedEnterprisePromptTrace(BaseTraceGenerator):
    workload_name = "shared_enterprise"
    description = "Many enterprise users share the same system prompt, policy block, tool schema, or RAG context."
    expected_behavior = "High prefix hit rate, high dedup savings, lower HBM pressure, and reduced repeated KV movement."

    def generate(
        self,
        sessions: int = 1000,
        shared_prefix_tokens: int = 8192,
        unique_tokens_per_session: int = 1024,
        decode_steps: int = 64,
        tenants: int = 4,
        prompt_fanout: int | None = None,
    ) -> Trace:
        events: list[TraceEvent] = []
        prefix_blocks = max(1, shared_prefix_tokens // self.profile.block_tokens)
        unique_blocks = max(1, unique_tokens_per_session // self.profile.block_tokens)
        fanout = prompt_fanout or sessions
        for s in range(sessions):
            tenant_id = f"tenant-{s % tenants}"
            session_id = f"{tenant_id}:s{s}"
            events.append(TraceEvent(0, 0, TraceEventType.SESSION_START, session_id, tenant_id, self.profile.model_name))
            for i in range(prefix_blocks):
                events.append(self._alloc(i, session_id, i, self.profile.block_tokens, EvictionClass.REUSABLE_PREFIX, tenant_id, "enterprise-policy-v1", fanout, gpu_id=f"gpu-{s % 8}"))
                events.append(TraceEvent(i, i * 1000, TraceEventType.PREFIX_LOOKUP, session_id, tenant_id, self.profile.model_name, prefix_hash="enterprise-policy-v1"))
                events.append(TraceEvent(i, i * 1000, TraceEventType.PREFIX_HIT if s else TraceEventType.PREFIX_MISS, session_id, tenant_id, self.profile.model_name, prefix_hash="enterprise-policy-v1"))
            for i in range(unique_blocks):
                block_i = prefix_blocks + i
                events.append(self._alloc(block_i, session_id, block_i, self.profile.block_tokens, EvictionClass.SESSION_RECENT, tenant_id))
            for d in range(decode_steps):
                events.append(self._access(prefix_blocks + unique_blocks + d, session_id, prefix_blocks + unique_blocks - 1))
        return Trace(events, self.workload_name, self.profile, "2 racks x 4 GPUs", self.description, ["synthetic simulation", "exact-match prefix hash"])


class AgenticLoopTrace(BaseTraceGenerator):
    workload_name = "agentic_loop"
    description = "Tool-calling agents repeatedly plan, call tools, observe, reflect, and continue."
    expected_behavior = "Evict ephemeral tool KV earlier, protect persistent memory, and prefetch reflection windows."

    def generate(
        self,
        sessions: int = 64,
        tools_per_session: int = 4,
        reflection_steps: int = 8,
        persistent_memory_tokens: int = 2048,
        ephemeral_tool_tokens: int = 512,
        decode_steps: int = 64,
    ) -> Trace:
        events: list[TraceEvent] = []
        persistent_blocks = max(1, persistent_memory_tokens // self.profile.block_tokens)
        tool_blocks = max(1, ephemeral_tool_tokens // self.profile.block_tokens)
        for s in range(sessions):
            session_id = f"agent{s}"
            events.append(TraceEvent(0, 0, TraceEventType.SESSION_START, session_id, model_id=self.profile.model_name))
            for i in range(persistent_blocks):
                events.append(self._alloc(i, session_id, i, self.profile.block_tokens, EvictionClass.REUSABLE_PREFIX, prefix_hash="agent-memory", fanout_count=sessions))
            step = persistent_blocks
            for tool in range(tools_per_session):
                events.append(TraceEvent(step, step * 1000, TraceEventType.TOOL_CALL_START, session_id, model_id=self.profile.model_name))
                for j in range(tool_blocks):
                    events.append(self._alloc(step + j, session_id, step + j, self.profile.block_tokens, EvictionClass.EPHEMERAL_TOOL_CALL))
                step += tool_blocks
                events.append(TraceEvent(step, step * 1000, TraceEventType.TOOL_CALL_END, session_id, model_id=self.profile.model_name))
                for r in range(reflection_steps):
                    events.append(TraceEvent(step + r, (step + r) * 1000, TraceEventType.KV_PREFETCH, session_id, model_id=self.profile.model_name))
            for d in range(decode_steps):
                events.append(self._access(step + d, session_id, persistent_blocks - 1))
        return Trace(events, self.workload_name, self.profile, "agentic single rack", self.description, ["synthetic tool-loop structure"])


class LongContextTrace(BaseTraceGenerator):
    workload_name = "long_context"
    description = "Few users with very large contexts and cold historical ranges."
    expected_behavior = "Compression and tiering reduce HBM pressure; generic spill can increase stall proxy."

    def generate(self, sessions: int = 8, context_tokens: int = 131072, decode_steps: int = 64, sliding_window: bool = True, cold_prefix_ratio: float = 0.7) -> Trace:
        events: list[TraceEvent] = []
        blocks = max(1, context_tokens // self.profile.block_tokens)
        cold_until = int(blocks * cold_prefix_ratio)
        for s in range(sessions):
            session_id = f"long{s}"
            events.append(TraceEvent(0, 0, TraceEventType.SESSION_START, session_id, model_id=self.profile.model_name))
            for i in range(blocks):
                klass = EvictionClass.LOW_ATTENTION if i < cold_until else EvictionClass.HOT_ACTIVE
                events.append(self._alloc(i, session_id, i, self.profile.block_tokens, klass))
            for d in range(decode_steps):
                events.append(self._access(blocks + d, session_id, blocks - 1 if sliding_window else d % blocks))
        return Trace(events, self.workload_name, self.profile, "large-context", self.description, ["synthetic long-context trace", f"cold_prefix_ratio={cold_prefix_ratio}"])


class MultiTenantRackTrace(BaseTraceGenerator):
    workload_name = "multi_tenant_rack"
    description = "Multiple tenants share a rack-scale inference system with overlapping and isolated prefixes."
    expected_behavior = "Rack-local prefix reuse, tenant-aware isolation, and avoided cross-rack movement."

    def generate(
        self,
        tenants: int = 8,
        sessions_per_tenant: int = 32,
        shared_prefix_probability: float = 0.65,
        cross_tenant_dedup_allowed: bool = False,
        racks: int = 2,
        gpus_per_rack: int = 4,
        decode_steps: int = 64,
    ) -> Trace:
        events: list[TraceEvent] = []
        prefix_blocks = 32
        unique_blocks = 8
        sessions = tenants * sessions_per_tenant
        for t in range(tenants):
            tenant_id = f"tenant-{t}"
            for s in range(sessions_per_tenant):
                global_s = t * sessions_per_tenant + s
                session_id = f"{tenant_id}:s{s}"
                gpu_id = f"gpu-{global_s % (racks * gpus_per_rack)}"
                prefix_hash = "global-policy" if cross_tenant_dedup_allowed else f"{tenant_id}:policy"
                events.append(TraceEvent(0, 0, TraceEventType.SESSION_START, session_id, tenant_id, self.profile.model_name, gpu_id=gpu_id))
                for i in range(prefix_blocks):
                    shared = (s / max(1, sessions_per_tenant)) < shared_prefix_probability
                    events.append(self._alloc(i, session_id, i, self.profile.block_tokens, EvictionClass.REUSABLE_PREFIX if shared else EvictionClass.SESSION_COLD, tenant_id, prefix_hash if shared else None, sessions, gpu_id))
                for i in range(unique_blocks):
                    events.append(self._alloc(prefix_blocks + i, session_id, prefix_blocks + i, self.profile.block_tokens, EvictionClass.SESSION_RECENT, tenant_id, gpu_id=gpu_id))
                for d in range(decode_steps):
                    events.append(self._access(prefix_blocks + unique_blocks + d, session_id, prefix_blocks + unique_blocks - 1))
        return Trace(events, self.workload_name, self.profile, f"{racks} racks x {gpus_per_rack} GPUs", self.description, ["synthetic multi-tenant rack", f"cross_tenant_dedup_allowed={cross_tenant_dedup_allowed}"])


class MixedProductionTrace(BaseTraceGenerator):
    workload_name = "mixed_production"
    description = "Blend of enterprise prompts, agentic loops, and long-context sessions."
    expected_behavior = "Shows control-plane value under mixed HBM, movement, prefix, and tool-loop pressure."

    def generate(self) -> Trace:
        traces = [
            SharedEnterprisePromptTrace(self.profile).generate(sessions=128, shared_prefix_tokens=4096, unique_tokens_per_session=512, decode_steps=32),
            AgenticLoopTrace(self.profile).generate(sessions=32, tools_per_session=3, decode_steps=32),
            LongContextTrace(self.profile).generate(sessions=4, context_tokens=32768, decode_steps=32),
        ]
        events: list[TraceEvent] = []
        offset = 0
        for trace in traces:
            for event in trace.events:
                events.append(
                    TraceEvent(
                        event.step + offset,
                        event.timestamp_us + offset * 1000,
                        event.event_type,
                        f"{trace.workload_name}:{event.session_id}",
                        event.tenant_id,
                        event.model_id,
                        event.gpu_id,
                        event.layer_id,
                        event.head_id,
                        event.token_start,
                        event.token_count,
                        event.bytes,
                        event.prefix_hash,
                        event.metadata,
                    )
                )
            offset += max(event.step for event in trace.events) + 10
        return Trace(events, self.workload_name, self.profile, "mixed rack-scale", self.description, ["synthetic mixed production blend"])


GENERATORS = {
    "shared-enterprise": SharedEnterprisePromptTrace,
    "agentic-loop": AgenticLoopTrace,
    "long-context": LongContextTrace,
    "multi-tenant-rack": MultiTenantRackTrace,
    "mixed-production": MixedProductionTrace,
}


def generate_named_trace(name: str, **kwargs) -> Trace:
    generator_cls = GENERATORS[name]
    return generator_cls().generate(**kwargs)


def export_sample_traces(directory: Path) -> list[Path]:
    traces = [
        SharedEnterprisePromptTrace().generate(sessions=100, shared_prefix_tokens=4096, unique_tokens_per_session=512, decode_steps=16),
        AgenticLoopTrace().generate(sessions=16, tools_per_session=2, decode_steps=16),
        LongContextTrace().generate(sessions=2, context_tokens=16384, decode_steps=16),
        MultiTenantRackTrace().generate(tenants=4, sessions_per_tenant=8, decode_steps=16),
        MixedProductionTrace().generate(),
    ]
    paths: list[Path] = []
    for trace in traces:
        generator = GENERATORS.get(trace.workload_name.replace("_", "-"), BaseTraceGenerator)()
        jsonl, md = generator.export(trace, directory)
        paths.extend([jsonl, md])
    return paths
