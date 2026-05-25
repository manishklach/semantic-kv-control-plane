import pandas as pd

from scripts.generate_paper_figures import generate_figures


def test_figure_generation_does_not_crash(tmp_path):
    rows = []
    policies = ["Naive HBM + LRU", "Distributed Semantic KV"]
    for scenario in ["s1", "s2"]:
        for policy in policies:
            rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "peak_hbm_gb": 10,
                    "bytes_moved_gb": 20,
                    "dedup_saved_gb": 5,
                    "cross_rack_traffic_gb": 3,
                    "eviction_count": 1,
                "prefetch_hit_rate": 0.5,
                "stall_p99_ms": 30,
                "energy_per_token": 1e-6,
                "active_hbm_gb": 2,
                "multicast_saved_gb": 1,
                "compression_saved_gb": 2,
                "topology_congestion_score": 0.2,
            }
        )
    csv = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    paths = generate_figures(csv, tmp_path / "figures")
    assert len(paths) == 14
    assert all(path.exists() for path in paths)
