# Benchmark Summary

Synthetic simulator results for policy comparison. These are not hardware measurements.

| workload | policy | hbm_used_peak | bytes_moved | dedup_saved_bytes | compression_saved_bytes | simulated_stall_us | prefetch_success_rate | bytes_avoided | multicast_saved_bytes | avoided_cross_rack_bytes | energy_per_token | topology_congestion_score | estimated_throughput_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared-prefix | NaiveHBMPolicy | 80.00 GB | 1920.00 GB | 0.00 GB | 0.00 GB | 404175 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 2.202e-06 J | 98.79 | 0.01 |
| shared-prefix | CXLSpillPolicy | 7.81 GB | 1000.00 GB | 0.00 GB | 0.00 GB | 46098 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 62.30 | 0.01 |
| shared-prefix | SemanticSingleNode | 7.81 GB | 565.20 GB | 443.67 GB | 434.80 GB | 10069 us | 100% | 443.67 GB | 443.67 GB | 0.00 GB | 6.482e-07 J | 30.11 | 1.18 |
| shared-prefix | TopologyAwareSemantic | 7.81 GB | 565.20 GB | 443.67 GB | 434.80 GB | 26294 us | 100% | 443.67 GB | 443.67 GB | 0.00 GB | 6.482e-07 J | 34.71 | 0.99 |
| shared-prefix | DistributedSemanticKV | 7.81 GB | 565.20 GB | 443.67 GB | 434.80 GB | 26294 us | 100% | 443.67 GB | 443.67 GB | 443.67 GB | 6.482e-07 J | 34.71 | 0.99 |
| long-context | NaiveHBMPolicy | 80.00 GB | 240.00 GB | 0.00 GB | 0.00 GB | 56545 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.720e-06 J | 0.47 | 0.66 |
| long-context | CXLSpillPolicy | 0.31 GB | 160.00 GB | 0.00 GB | 0.00 GB | 6059 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 0.00 | 0.84 |
| long-context | SemanticSingleNode | 0.31 GB | 160.00 GB | 0.00 GB | 0.00 GB | 1507 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 0.00 | 0.84 |
| long-context | TopologyAwareSemantic | 0.31 GB | 160.00 GB | 0.00 GB | 0.00 GB | 6059 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 0.00 | 0.84 |
| long-context | DistributedSemanticKV | 0.31 GB | 160.00 GB | 0.00 GB | 0.00 GB | 6059 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 0.00 | 0.84 |
| agentic-workflow | NaiveHBMPolicy | 80.00 GB | 585.00 GB | 0.00 GB | 0.00 GB | 127221 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 2.018e-06 J | 39.59 | 0.41 |
| agentic-workflow | CXLSpillPolicy | 5.00 GB | 332.50 GB | 0.00 GB | 0.00 GB | 15619 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 33.97 | 0.90 |
| agentic-workflow | SemanticSingleNode | 5.00 GB | 186.94 GB | 145.51 GB | 146.37 GB | 7023 us | 100% | 145.51 GB | 145.51 GB | 0.00 GB | 9.460e-07 J | 15.47 | 2.36 |
| agentic-workflow | TopologyAwareSemantic | 5.00 GB | 186.94 GB | 145.51 GB | 146.37 GB | 12084 us | 100% | 145.51 GB | 145.51 GB | 0.00 GB | 9.460e-07 J | 19.35 | 2.27 |
| agentic-workflow | DistributedSemanticKV | 5.00 GB | 186.94 GB | 145.51 GB | 146.37 GB | 12084 us | 100% | 145.51 GB | 145.51 GB | 145.51 GB | 9.460e-07 J | 19.35 | 2.27 |
| multi-tenant-inference | NaiveHBMPolicy | 80.00 GB | 245.00 GB | 0.00 GB | 0.00 GB | 59584 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.729e-06 J | 30.13 | 0.41 |
| multi-tenant-inference | CXLSpillPolicy | 2.50 GB | 162.50 GB | 0.00 GB | 0.00 GB | 7714 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 38.80 | 0.46 |
| multi-tenant-inference | SemanticSingleNode | 2.50 GB | 123.15 GB | 40.16 GB | 39.35 GB | 1999 us | 0% | 40.16 GB | 40.16 GB | 0.00 GB | 8.691e-07 J | 20.76 | 1.55 |
| multi-tenant-inference | TopologyAwareSemantic | 2.50 GB | 123.15 GB | 40.16 GB | 39.35 GB | 5503 us | 0% | 40.16 GB | 40.16 GB | 0.00 GB | 8.691e-07 J | 26.97 | 1.48 |
| multi-tenant-inference | DistributedSemanticKV | 2.50 GB | 123.15 GB | 40.16 GB | 39.35 GB | 5503 us | 0% | 40.16 GB | 40.16 GB | 0.00 GB | 8.691e-07 J | 26.97 | 1.48 |
| shared-enterprise-prompt | NaiveHBMPolicy | 80.00 GB | 1840.00 GB | 0.00 GB | 0.00 GB | 392132 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 2.198e-06 J | 134.28 | 0.01 |
| shared-enterprise-prompt | CXLSpillPolicy | 10.00 GB | 960.00 GB | 0.00 GB | 0.00 GB | 45106 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 80.85 | 0.01 |
| shared-enterprise-prompt | SemanticSingleNode | 10.00 GB | 541.55 GB | 426.99 GB | 418.45 GB | 10599 us | 100% | 426.99 GB | 426.99 GB | 0.00 GB | 6.470e-07 J | 40.75 | 1.18 |
| shared-enterprise-prompt | TopologyAwareSemantic | 10.00 GB | 541.55 GB | 426.99 GB | 418.45 GB | 26090 us | 100% | 426.99 GB | 426.99 GB | 0.00 GB | 6.470e-07 J | 45.32 | 1.04 |
| shared-enterprise-prompt | DistributedSemanticKV | 10.00 GB | 541.55 GB | 426.99 GB | 418.45 GB | 26090 us | 100% | 426.99 GB | 426.99 GB | 426.99 GB | 6.470e-07 J | 45.32 | 1.04 |
| multi-agent-collaboration | NaiveHBMPolicy | 80.00 GB | 520.00 GB | 0.00 GB | 0.00 GB | 116052 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.988e-06 J | 45.23 | 0.51 |
| multi-agent-collaboration | CXLSpillPolicy | 5.00 GB | 300.00 GB | 0.00 GB | 0.00 GB | 14251 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 36.40 | 0.99 |
| multi-agent-collaboration | SemanticSingleNode | 5.00 GB | 141.75 GB | 161.48 GB | 158.25 GB | 3425 us | 100% | 161.48 GB | 161.48 GB | 0.00 GB | 5.419e-07 J | 14.54 | 2.51 |
| multi-agent-collaboration | TopologyAwareSemantic | 5.00 GB | 141.75 GB | 161.48 GB | 158.25 GB | 7169 us | 100% | 161.48 GB | 161.48 GB | 0.00 GB | 5.419e-07 J | 17.89 | 2.44 |
| multi-agent-collaboration | DistributedSemanticKV | 5.00 GB | 141.75 GB | 161.48 GB | 158.25 GB | 7169 us | 100% | 161.48 GB | 161.48 GB | 161.48 GB | 5.419e-07 J | 17.89 | 2.44 |
