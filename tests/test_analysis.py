import pandas as pd

from semantic_kv.analysis import ResultInterpreter


def test_analysis_produces_simulation_findings():
    frame = pd.DataFrame(
        [
            {
                "scenario": "s",
                "policy": "Naive HBM + LRU",
                "peak_hbm_gb": 100,
                "bytes_moved_gb": 100,
                "simulated_stall_ms": 100,
            },
            {
                "scenario": "s",
                "policy": "Distributed Semantic KV",
                "peak_hbm_gb": 50,
                "bytes_moved_gb": 40,
                "simulated_stall_ms": 60,
                "avoided_cross_rack_gb": 10,
            },
        ]
    )
    findings = ResultInterpreter(frame).findings()
    assert findings
    assert "Simulated" in findings[0].text
