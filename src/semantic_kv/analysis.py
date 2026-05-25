"""Interpret benchmark outputs into cautious, human-readable findings."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Finding:
    """Represent one human-readable benchmark interpretation."""

    text: str
    confidence: str
    scenario: str
    metric: str


class ResultInterpreter:
    """Generate simulation-labeled findings from benchmark result tables."""

    def __init__(self, results: pd.DataFrame) -> None:
        self.results = results

    def findings(self) -> list[Finding]:
        findings: list[Finding] = []
        for scenario, frame in self.results.groupby("scenario"):
            naive = self._row(frame, "Naive HBM + LRU")
            distributed = self._row(frame, "Distributed Semantic KV")
            topology = self._row(frame, "Topology-aware Semantic KV")
            semantic = self._row(frame, "Single-node Semantic KV")
            if naive is not None and distributed is not None:
                findings.extend(
                    [
                        self._relative_finding(
                            scenario,
                            "peak_hbm_gb",
                            naive,
                            distributed,
                            "Distributed semantic KV reduced peak HBM usage",
                        ),
                        self._relative_finding(
                            scenario,
                            "bytes_moved_gb",
                            naive,
                            distributed,
                            "Distributed semantic KV reduced memory movement",
                        ),
                        self._relative_finding(
                            scenario,
                            "simulated_stall_ms",
                            naive,
                            distributed,
                            "Distributed semantic KV reduced the decode-stall proxy",
                        ),
                    ]
                )
            if topology is not None and distributed is not None:
                avoided = distributed.get("avoided_cross_rack_gb", 0)
                if avoided > 0:
                    findings.append(
                        Finding(
                            "Simulated under synthetic workload assumptions, "
                            f"distributed prefix reuse avoided {avoided:.2f} GB "
                            f"of cross-rack KV movement in {scenario}.",
                            "MEDIUM",
                            scenario,
                            "avoided_cross_rack_gb",
                        )
                    )
            if semantic is not None and semantic.get("compression_saved_gb", 0) > 0:
                findings.append(
                    Finding(
                        "Simulated under synthetic workload assumptions, "
                        "semantic compression saved "
                        f"{semantic['compression_saved_gb']:.2f} GB in {scenario}; "
                        "quality risk is modeled, not measured.",
                        "MEDIUM",
                        scenario,
                        "compression_saved_gb",
                    )
                )
        return [finding for finding in findings if finding.text]

    def to_markdown(self) -> str:
        lines = [
            "# Result Interpretation",
            "",
            "All findings are synthetic simulation results under workload "
            "assumptions, not hardware measurements.",
            "",
        ]
        for finding in self.findings():
            lines.append(
                f"- **{finding.confidence}** ({finding.scenario}, "
                f"`{finding.metric}`): {finding.text}"
            )
        return "\n".join(lines) + "\n"

    def _relative_finding(
        self,
        scenario: str,
        metric: str,
        baseline: pd.Series,
        candidate: pd.Series,
        phrase: str,
    ) -> Finding:
        base = float(baseline.get(metric, 0))
        cand = float(candidate.get(metric, 0))
        if base <= 0:
            return Finding("", "LOW", scenario, metric)
        improvement = max(0.0, (base - cand) / base * 100)
        confidence = "HIGH" if improvement >= 40 else "MEDIUM" if improvement >= 15 else "LOW"
        return Finding(
            "Simulated under synthetic workload assumptions, "
            f"{phrase} by {improvement:.1f}% vs naive HBM in {scenario}.",
            confidence,
            scenario,
            metric,
        )

    def _row(self, frame: pd.DataFrame, policy: str) -> pd.Series | None:
        matched = frame[frame["policy"] == policy]
        return None if matched.empty else matched.iloc[0]
