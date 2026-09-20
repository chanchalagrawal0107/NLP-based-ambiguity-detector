"""Evaluation layer (Phase 7): labelled dataset, metrics, and the runner
behind ``--evaluate``. Opt-in and slow against a real LLM, exactly like
``--live-llm-test`` - never invoked by pytest.
"""

from __future__ import annotations

from ambisense.evaluation.dataset import DatasetError, EvaluationCase, load_dataset
from ambisense.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationOutcome,
    build_outcome,
    classify_verdict,
    compute_metrics,
)
from ambisense.evaluation.runner import EvaluationResults, run_evaluation

__all__ = [
    "DatasetError",
    "EvaluationCase",
    "load_dataset",
    "EvaluationMetrics",
    "EvaluationOutcome",
    "build_outcome",
    "classify_verdict",
    "compute_metrics",
    "EvaluationResults",
    "run_evaluation",
]
