# Social Thread Draft: When KV Cache Becomes a Distributed Systems Problem

1. KV cache is starting to look less like a runtime detail and more like distributed infrastructure.

2. Long context, shared prompts, RAG payloads, and agent loops all create memory pressure that generic spill policies cannot fully understand.

3. I built a synthetic simulation platform to ask a narrow question: what changes if KV metadata includes prefix hashes, fanout, eviction class, topology, and movement cost?

4. The project does not run CUDA and does not claim real hardware speedups. It is a policy sandbox for memory-orchestrated inference.

5. The interesting result is not a single number. It is the shape of the problem: HBM pressure, movement, prefix reuse, congestion, and energy proxy interact.

6. CXL adds capacity, but semantic orchestration asks what should move, where it should live, and whether movement is worth it.

7. Next step: import real serving traces and compare simulated policies against runtime-observed KV behavior.
