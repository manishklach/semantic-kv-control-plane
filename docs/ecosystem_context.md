# Ecosystem Context

This project explores a complementary abstraction layer for semantic KV orchestration. It is not a replacement for production inference engines or KV cache systems.

## vLLM PagedAttention And Prefix Caching

vLLM popularized PagedAttention-style KV management, where KV cache is partitioned into blocks and managed with paging-inspired allocation. vLLM documentation describes automatic prefix caching around KV blocks identified by block tokens plus preceding prefix context.

Relevant references:

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.9.0/design/automatic_prefix_caching.html)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)

This repo does not implement PagedAttention kernels. It assumes a block abstraction and asks what a higher-level control plane could do with semantic metadata, topology, and movement costs.

## TensorRT-LLM KV Reuse

TensorRT-LLM supports KV cache reuse across requests that begin with the same prompt, and its KV cache system includes reuse, offload, and prioritized eviction concepts. NVIDIA has also described KV cache event APIs and event-aware routing/scheduling for reuse opportunities.

Relevant references:

- [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)
- [TensorRT-LLM KV cache reuse](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/kv-cache-reuse.md)
- [NVIDIA blog on KV cache reuse optimizations](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/)

This repo’s trace replay and semantic eviction experiments are inspired by that direction, but remain synthetic simulation.

## LMCache

LMCache focuses on reusable KV cache storage, movement, and serving-engine integration. Its documentation describes prefill-once, reuse-everywhere semantics for reusable text, not only strict prefixes.

Relevant reference:

- [LMCache documentation](https://docs.lmcache.ai/)

This project treats LMCache-like systems as part of the broader ecosystem and explores policy questions around topology, semantic classes, and movement economics.

## SGLang / Mooncake-Style Distributed Serving

Mooncake frames LLM serving around KV-cache-centric disaggregation and scheduling. SGLang integrations with Mooncake-style backends point toward distributed KV storage and hierarchical cache designs.

Relevant references:

- [Mooncake paper](https://arxiv.org/abs/2407.00079)
- [SGLang Mooncake L3 KV cache documentation mirror](https://contextqmd.com/libraries/sglang/versions/0.5.9/pages/python/sglang/srt/mem_cache/storage/mooncake_store/README)

This repo is aligned with the question those systems raise: what happens when KV locality, routing, and reuse become distributed-systems concerns?

## CXL Memory Expansion

CXL-style memory expansion can provide capacity, but capacity is not the same as intent. CXL does not decide whether a KV block is a hot decode block, reusable prefix, ephemeral tool-call block, or cheap-to-recompute history.

This repo models CXL-like memory as one tier in a larger semantic control-plane simulation.

## Position Of This Repo

This project explores a complementary abstraction layer:

- semantic metadata around KV blocks
- topology-aware placement
- rack-local and global prefix reuse
- predictive prefetch
- distributed semantic eviction
- movement and energy proxy accounting

It does not claim superiority over vLLM, TensorRT-LLM, LMCache, SGLang, or Mooncake. It is a research simulator for asking what a memory-orchestrated inference control plane might need to know.
