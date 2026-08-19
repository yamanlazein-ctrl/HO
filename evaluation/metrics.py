"""
Evaluation Metrics — 🔴 core contribution.

Per-scenario evaluation, not just aggregate F1. This is the project's
answer to "is this just detecting trash vs. detecting the *behavior*
of littering?".

A run produces, for each test clip:
    - predicted: did the system emit a LITTERING_CONFIRMED event? (bool)
    - ground_truth: is this clip a real littering event? (bool)
    - scenario: which behavioral category (see dataset_schema.md)

We then compute per-scenario and aggregate:
    Event Precision  = TP / (TP + FP)
    Event Recall     = TP / (TP + FN)
    F1               = 2 * P * R / (P + R)
    False Positive Rate = FP / (FP + TN)
    Latency          = mean time from event timestamp to confirmation
    FPS              = measured during the run

The per-scenario breakdown is the defensible artifact: it shows the
system distinguishes *put-down* from *throw*, *carry* from *drop*, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class ClipResult:
    clip_id: str
    scenario: str
    ground_truth: bool          # True if this clip is a real littering event
    predicted: bool             # True if system confirmed littering
    latency_seconds: float = 0.0  # event_ts → confirmation (0 if no event)
    fps: float = 0.0


@dataclass
class ScenarioMetrics:
    scenario: str
    n: int
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fpr: float = 0.0  # false positive rate
    mean_latency: float = 0.0
    mean_fps: float = 0.0


@dataclass
class EvaluationReport:
    aggregate: ScenarioMetrics
    per_scenario: Dict[str, ScenarioMetrics] = field(default_factory=dict)
    confusion_matrix: Dict[str, int] = field(default_factory=dict)
    all_results: List[ClipResult] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "aggregate": asdict(self.aggregate),
                "per_scenario": {k: asdict(v) for k, v in self.per_scenario.items()},
                "confusion_matrix": self.confusion_matrix,
                "all_results": [asdict(r) for r in self.all_results],
            }, f, indent=2)

    def summary_str(self) -> str:
        lines = ["=" * 60, "EVALUATION REPORT", "=" * 60, ""]
        a = self.aggregate
        lines.append(f"AGGREGATE (n={a.n}): P={a.precision:.3f} R={a.recall:.3f} F1={a.f1:.3f} FPR={a.fpr:.3f}")
        lines.append(f"  TP={a.tp} FP={a.fp} FN={a.fn} TN={a.tn}")
        lines.append(f"  mean latency={a.mean_latency:.2f}s  mean FPS={a.mean_fps:.1f}")
        lines.append("")
        lines.append("PER-SCENARIO:")
        lines.append(f"  {'scenario':<28} {'n':>3} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6}")
        for name, m in sorted(self.per_scenario.items()):
            lines.append(f"  {name:<28} {m.n:>3} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} {m.fpr:>6.3f}")
        return "\n".join(lines)


def _prf(tp: int, fp: int, fn: int, tn: int) -> tuple:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, f1, fpr


def evaluate(results: List[ClipResult]) -> EvaluationReport:
    """Compute per-scenario + aggregate metrics from clip results."""

    # group by scenario
    by_scenario: Dict[str, List[ClipResult]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r)

    per_scenario: Dict[str, ScenarioMetrics] = {}
    for name, rs in by_scenario.items():
        tp = sum(1 for r in rs if r.predicted and r.ground_truth)
        fp = sum(1 for r in rs if r.predicted and not r.ground_truth)
        fn = sum(1 for r in rs if not r.predicted and r.ground_truth)
        tn = sum(1 for r in rs if not r.predicted and not r.ground_truth)
        p, rec, f1, fpr = _prf(tp, fp, fn, tn)
        latencies = [r.latency_seconds for r in rs if r.predicted and r.latency_seconds > 0]
        fps_vals = [r.fps for r in rs if r.fps > 0]
        per_scenario[name] = ScenarioMetrics(
            scenario=name, n=len(rs), tp=tp, fp=fp, fn=fn, tn=tn,
            precision=p, recall=rec, f1=f1, fpr=fpr,
            mean_latency=sum(latencies) / len(latencies) if latencies else 0.0,
            mean_fps=sum(fps_vals) / len(fps_vals) if fps_vals else 0.0,
        )

    # aggregate
    tp = sum(1 for r in results if r.predicted and r.ground_truth)
    fp = sum(1 for r in results if r.predicted and not r.ground_truth)
    fn = sum(1 for r in results if not r.predicted and r.ground_truth)
    tn = sum(1 for r in results if not r.predicted and not r.ground_truth)
    p, rec, f1, fpr = _prf(tp, fp, fn, tn)
    latencies = [r.latency_seconds for r in results if r.predicted and r.latency_seconds > 0]
    fps_vals = [r.fps for r in results if r.fps > 0]
    aggregate = ScenarioMetrics(
        scenario="ALL", n=len(results), tp=tp, fp=fp, fn=fn, tn=tn,
        precision=p, recall=rec, f1=f1, fpr=fpr,
        mean_latency=sum(latencies) / len(latencies) if latencies else 0.0,
        mean_fps=sum(fps_vals) / len(fps_vals) if fps_vals else 0.0,
    )

    confusion = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return EvaluationReport(aggregate=aggregate, per_scenario=per_scenario,
                            confusion_matrix=confusion, all_results=results)
