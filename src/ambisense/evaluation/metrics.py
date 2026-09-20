"""Phase 7: turn (ground truth, predicted verdict) pairs into honest metrics.

Pure functions, no I/O and no LLM - feed them known outcomes and the numbers
are deterministic, which is what makes them unit-testable without Ollama.

``SentenceVerdict.UNCERTAIN`` and ``INCOMPLETE`` are treated as an
**abstention**, never coerced into a binary label: Section 5 of the README
requires the system to be able to say "the evidence does not allow a
confident answer" rather than force one, and scoring an abstention as if it
were a guess would erase that distinction. Two numbers are reported instead
of one: ``accuracy`` (over confident predictions only) and
``pessimistic_accuracy`` (every abstention counted as wrong, over all
cases) - the optimistic number is never shown without the pessimistic one
beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ambisense.evaluation.dataset import EvaluationCase
from ambisense.schemas import SentenceVerdict

#: Binary label space. Order matters for confusion_matrix's axes.
_LABELS = ("not_ambiguous", "ambiguous")


def classify_verdict(verdict: SentenceVerdict) -> Optional[str]:
    """Map a five-way ``SentenceVerdict`` onto a binary prediction, or
    ``None`` for an abstention (``UNCERTAIN`` / ``INCOMPLETE``)."""
    if verdict is SentenceVerdict.AMBIGUOUS:
        return "ambiguous"
    if verdict in (SentenceVerdict.NOT_AMBIGUOUS, SentenceVerdict.NO_CANDIDATES):
        return "not_ambiguous"
    return None


@dataclass(frozen=True)
class EvaluationOutcome:
    """One case's ground truth against the system's prediction."""

    case_id: str
    label: str
    verdict: SentenceVerdict
    prediction: Optional[str]

    @property
    def abstained(self) -> bool:
        return self.prediction is None

    @property
    def correct(self) -> Optional[bool]:
        return None if self.abstained else self.prediction == self.label


def build_outcome(case: EvaluationCase, verdict: SentenceVerdict) -> EvaluationOutcome:
    return EvaluationOutcome(
        case_id=case.id,
        label=case.label,
        verdict=verdict,
        prediction=classify_verdict(verdict),
    )


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metrics over a set of outcomes. ``None`` where a number cannot be
    honestly computed (e.g. no confident predictions at all) rather than a
    manufactured 0.0."""

    total: int
    abstained: int
    confident: int
    abstention_rate: float
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    pessimistic_accuracy: float
    confusion: dict[str, int] = field(default_factory=dict)


def compute_metrics(outcomes: list[EvaluationOutcome]) -> EvaluationMetrics:
    total = len(outcomes)
    if total == 0:
        raise ValueError("compute_metrics requires at least one outcome")

    confident = [o for o in outcomes if not o.abstained]
    abstained = total - len(confident)
    correct_confident = sum(1 for o in confident if o.correct)

    if not confident:
        return EvaluationMetrics(
            total=total, abstained=abstained, confident=0,
            abstention_rate=abstained / total,
            accuracy=None, precision=None, recall=None, f1=None,
            pessimistic_accuracy=0.0, confusion={},
        )

    y_true = [o.label for o in confident]
    y_pred = [o.prediction for o in confident]

    accuracy = correct_confident / len(confident)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=_LABELS, pos_label="ambiguous",
        average="binary", zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=_LABELS)
    tn, fp, fn, tp = matrix.ravel()

    return EvaluationMetrics(
        total=total, abstained=abstained, confident=len(confident),
        abstention_rate=abstained / total,
        accuracy=accuracy, precision=float(precision), recall=float(recall),
        f1=float(f1), pessimistic_accuracy=correct_confident / total,
        confusion={"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    )
