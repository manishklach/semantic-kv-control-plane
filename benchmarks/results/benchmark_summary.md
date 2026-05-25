# Benchmark Summary

Synthetic simulator results for policy comparison. These are not hardware measurements.

| workload | policy | hbm_used_peak | bytes_moved | dedup_saved_bytes | compression_saved_bytes | simulated_stall_us | prefetch_success_rate | bytes_avoided | multicast_saved_bytes | avoided_cross_rack_bytes | energy_per_token | topology_congestion_score | estimated_throughput_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared-prefix | NaiveHBMPolicy | 80.00 GB | 1920.00 GB | 0.00 GB | 0.00 GB | 404175 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 2.202e-06 J | 98.79 | 0.01 |
| shared-prefix | CXLSpillPolicy | 7.81 GB | 1000.00 GB | 0.00 GB | 0.00 GB | 46098 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 62.30 | 0.01 |
| shared-prefix | SemanticSingleNode | 7.81 GB | 514.90 GB | 495.00 GB | 485.10 GB | 7921 us | 100% | 495.00 GB | 495.00 GB | 0.00 GB | 5.905e-07 J | 26.38 | 1.48 |
| shared-prefix | TopologyAwareSemantic | 7.81 GB | 514.90 GB | 495.00 GB | 485.10 GB | 24063 us | 100% | 495.00 GB | 495.00 GB | 0.00 GB | 5.905e-07 J | 30.97 | 1.28 |
| shared-prefix | DistributedSemanticKV | 7.81 GB | 514.90 GB | 495.00 GB | 485.10 GB | 24063 us | 100% | 495.00 GB | 495.00 GB | 495.00 GB | 5.905e-07 J | 30.97 | 1.28 |
| long-context | NaiveHBMPolicy | 80.00 GB | 240.00 GB | 0.00 GB | 0.00 GB | 56545 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.720e-06 J | 0.47 | 0.66 |
| long-context | CXLSpillPolicy | 0.31 GB | 160.00 GB | 0.00 GB | 0.00 GB | 6059 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 0.00 | 0.84 |
| long-context | SemanticSingleNode | 0.31 GB | 160.00 GB | 0.00 GB | 0.00 GB | 1507 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 0.00 | 0.84 |
| long-context | TopologyAwareSemantic | 0.31 GB | 160.00 GB | 0.00 GB | 0.00 GB | 6059 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 0.00 | 0.84 |
| long-context | DistributedSemanticKV | 0.31 GB | 160.00 GB | 0.00 GB | 0.00 GB | 6059 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 0.00 | 0.84 |
| agentic-workflow | NaiveHBMPolicy | 80.00 GB | 585.00 GB | 0.00 GB | 0.00 GB | 127221 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 2.018e-06 J | 39.59 | 0.41 |
| agentic-workflow | CXLSpillPolicy | 5.00 GB | 332.50 GB | 0.00 GB | 0.00 GB | 15619 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 33.97 | 0.90 |
| agentic-workflow | SemanticSingleNode | 5.00 GB | 170.36 GB | 162.42 GB | 162.95 GB | 6311 us | 100% | 162.42 GB | 162.42 GB | 0.00 GB | 8.888e-07 J | 13.59 | 2.52 |
| agentic-workflow | TopologyAwareSemantic | 5.00 GB | 170.36 GB | 162.42 GB | 162.95 GB | 11365 us | 100% | 162.42 GB | 162.42 GB | 0.00 GB | 8.888e-07 J | 17.47 | 2.43 |
| agentic-workflow | DistributedSemanticKV | 5.00 GB | 170.36 GB | 162.42 GB | 162.95 GB | 11365 us | 100% | 162.42 GB | 162.42 GB | 162.42 GB | 8.888e-07 J | 17.47 | 2.43 |
| multi-tenant-inference | NaiveHBMPolicy | 80.00 GB | 245.00 GB | 0.00 GB | 0.00 GB | 59584 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.729e-06 J | 30.13 | 0.41 |
| multi-tenant-inference | CXLSpillPolicy | 2.50 GB | 162.50 GB | 0.00 GB | 0.00 GB | 7714 us | 0% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 38.80 | 0.46 |
| multi-tenant-inference | SemanticSingleNode | 2.50 GB | 117.48 GB | 45.94 GB | 45.02 GB | 1944 us | 0% | 45.94 GB | 45.94 GB | 0.00 GB | 8.292e-07 J | 19.51 | 1.69 |
| multi-tenant-inference | TopologyAwareSemantic | 2.50 GB | 117.48 GB | 45.94 GB | 45.02 GB | 5448 us | 0% | 45.94 GB | 45.94 GB | 0.00 GB | 8.292e-07 J | 25.71 | 1.62 |
| multi-tenant-inference | DistributedSemanticKV | 2.50 GB | 117.48 GB | 45.94 GB | 45.02 GB | 5448 us | 0% | 45.94 GB | 45.94 GB | 0.00 GB | 8.292e-07 J | 25.71 | 1.62 |
| shared-enterprise-prompt | NaiveHBMPolicy | 80.00 GB | 1840.00 GB | 0.00 GB | 0.00 GB | 392132 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 2.198e-06 J | 134.28 | 0.01 |
| shared-enterprise-prompt | CXLSpillPolicy | 10.00 GB | 960.00 GB | 0.00 GB | 0.00 GB | 45106 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 80.85 | 0.01 |
| shared-enterprise-prompt | SemanticSingleNode | 10.00 GB | 493.27 GB | 476.25 GB | 466.73 GB | 8534 us | 100% | 476.25 GB | 476.25 GB | 0.00 GB | 5.893e-07 J | 35.61 | 1.48 |
| shared-enterprise-prompt | TopologyAwareSemantic | 10.00 GB | 493.27 GB | 476.25 GB | 466.73 GB | 23949 us | 100% | 476.25 GB | 476.25 GB | 0.00 GB | 5.893e-07 J | 40.18 | 1.34 |
| shared-enterprise-prompt | DistributedSemanticKV | 10.00 GB | 493.27 GB | 476.25 GB | 466.73 GB | 23949 us | 100% | 476.25 GB | 476.25 GB | 476.25 GB | 5.893e-07 J | 40.18 | 1.34 |
| multi-agent-collaboration | NaiveHBMPolicy | 80.00 GB | 520.00 GB | 0.00 GB | 0.00 GB | 116052 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.988e-06 J | 45.23 | 0.51 |
| multi-agent-collaboration | CXLSpillPolicy | 5.00 GB | 300.00 GB | 0.00 GB | 0.00 GB | 14251 us | 100% | 0.00 GB | 0.00 GB | 0.00 GB | 1.147e-06 J | 36.40 | 0.99 |
| multi-agent-collaboration | SemanticSingleNode | 5.00 GB | 129.11 GB | 174.38 GB | 170.89 GB | 2880 us | 100% | 174.38 GB | 174.38 GB | 0.00 GB | 4.936e-07 J | 10.54 | 2.67 |
| multi-agent-collaboration | TopologyAwareSemantic | 5.00 GB | 129.11 GB | 174.38 GB | 170.89 GB | 6621 us | 100% | 174.38 GB | 174.38 GB | 0.00 GB | 4.936e-07 J | 13.90 | 2.60 |
| multi-agent-collaboration | DistributedSemanticKV | 5.00 GB | 129.11 GB | 174.38 GB | 170.89 GB | 6621 us | 100% | 174.38 GB | 174.38 GB | 174.38 GB | 4.936e-07 J | 13.90 | 2.60 |
