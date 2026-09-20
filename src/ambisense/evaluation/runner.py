"""Phase 7: run the labelled dataset through the full pipeline.

Mirrors ``main.py::command_live_llm_test``'s shape - one shared adjudicator
across cases, so the response cache and diagnostics behave the same way -
but collects (ground truth, predicted verdict) pairs into metrics instead of
just printing each report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ambisense.evaluation.dataset import EvaluationCase
from ambisense.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationOutcome,
    build_outcome,
    compute_metrics,
)
from ambisense.llm import build_adjudicator
from ambisense.llm.adjudicator import LLMAdjudicator
from ambisense.pipeline import analyze
from ambisense.schemas import AdjudicationReport


@dataclass(frozen=True)
class EvaluationResults:
    outcomes: list[EvaluationOutcome]
    metrics: EvaluationMetrics
    reports: dict[str, AdjudicationReport]


def run_evaluation(
    settings,
    dataset: list[EvaluationCase],
    adjudicator: Optional[LLMAdjudicator] = None,
) -> EvaluationResults:
    """Run every case through the real pipeline and score the predictions.

    Args:
        adjudicator: Reused across cases when supplied (so the response
            cache and diagnostics are shared, exactly like
            ``--live-llm-test``); built fresh from ``settings`` otherwise.
    """
    adjudicator = adjudicator or build_adjudicator(settings)
    outcomes: list[EvaluationOutcome] = []
    reports: dict[str, AdjudicationReport] = {}
    for case in dataset:
        report = analyze(settings, case.text, case.context, adjudicator=adjudicator)
        reports[case.id] = report
        outcomes.append(build_outcome(case, report.summary.verdict))
    return EvaluationResults(
        outcomes=outcomes, metrics=compute_metrics(outcomes), reports=reports
    )
