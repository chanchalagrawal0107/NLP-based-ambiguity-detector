"""Phase 7: dataset loading, verdict mapping and metric computation.

Pure-function tests with hand-built outcomes - no LLM, no spaCy, no network.
The metrics are the numbers reported in the README, so the arithmetic and, in
particular, the treatment of abstentions are pinned down here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ambisense.evaluation.dataset import DatasetError, EvaluationCase, load_dataset
from ambisense.evaluation.metrics import (
    EvaluationOutcome,
    build_outcome,
    classify_verdict,
    compute_metrics,
)
from ambisense.schemas import SentenceVerdict

SHIPPED_DATASET = (
    Path(__file__).resolve().parents[1] / "data" / "evaluation" / "sentence_labels.json"
)


def _outcome(label: str, verdict: SentenceVerdict, case_id: str = "c") -> EvaluationOutcome:
    return build_outcome(
        EvaluationCase(id=case_id, text="x", label=label), verdict
    )


class TestClassifyVerdict:
    @pytest.mark.parametrize(
        "verdict, expected",
        [
            (SentenceVerdict.AMBIGUOUS, "ambiguous"),
            (SentenceVerdict.NOT_AMBIGUOUS, "not_ambiguous"),
            (SentenceVerdict.NO_CANDIDATES, "not_ambiguous"),
            (SentenceVerdict.UNCERTAIN, None),
            (SentenceVerdict.INCOMPLETE, None),
        ],
    )
    def test_mapping(self, verdict, expected):
        assert classify_verdict(verdict) == expected

    def test_every_verdict_is_mapped(self):
        for verdict in SentenceVerdict:
            classify_verdict(verdict)  # must not raise


class TestOutcome:
    def test_abstention_is_neither_correct_nor_wrong(self):
        outcome = _outcome("ambiguous", SentenceVerdict.UNCERTAIN)
        assert outcome.abstained
        assert outcome.correct is None

    def test_correct_and_wrong(self):
        assert _outcome("ambiguous", SentenceVerdict.AMBIGUOUS).correct is True
        assert _outcome("ambiguous", SentenceVerdict.NOT_AMBIGUOUS).correct is False


class TestComputeMetrics:
    def test_known_confusion_matrix(self):
        outcomes = [
            _outcome("ambiguous", SentenceVerdict.AMBIGUOUS),          # TP
            _outcome("ambiguous", SentenceVerdict.AMBIGUOUS),          # TP
            _outcome("ambiguous", SentenceVerdict.NOT_AMBIGUOUS),      # FN
            _outcome("not_ambiguous", SentenceVerdict.AMBIGUOUS),      # FP
            _outcome("not_ambiguous", SentenceVerdict.NOT_AMBIGUOUS),  # TN
            _outcome("not_ambiguous", SentenceVerdict.NO_CANDIDATES),  # TN
        ]
        metrics = compute_metrics(outcomes)

        assert metrics.confusion == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
        assert metrics.total == 6 and metrics.confident == 6 and metrics.abstained == 0
        assert metrics.accuracy == pytest.approx(4 / 6)
        assert metrics.precision == pytest.approx(2 / 3)
        assert metrics.recall == pytest.approx(2 / 3)
        assert metrics.f1 == pytest.approx(2 / 3)

    def test_abstentions_are_excluded_from_confident_and_counted_against_pessimistic(self):
        outcomes = [
            _outcome("ambiguous", SentenceVerdict.AMBIGUOUS),
            _outcome("not_ambiguous", SentenceVerdict.NOT_AMBIGUOUS),
            _outcome("ambiguous", SentenceVerdict.UNCERTAIN),
            _outcome("not_ambiguous", SentenceVerdict.INCOMPLETE),
        ]
        metrics = compute_metrics(outcomes)

        assert metrics.abstained == 2 and metrics.confident == 2
        assert metrics.abstention_rate == pytest.approx(0.5)
        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.pessimistic_accuracy == pytest.approx(0.5)
        assert metrics.pessimistic_accuracy <= metrics.accuracy

    def test_all_abstentions_yield_no_manufactured_numbers(self):
        metrics = compute_metrics(
            [_outcome("ambiguous", SentenceVerdict.UNCERTAIN)]
        )
        assert metrics.confident == 0
        assert metrics.accuracy is None
        assert metrics.precision is None and metrics.recall is None and metrics.f1 is None
        assert metrics.pessimistic_accuracy == 0.0

    def test_no_positive_predictions_gives_zero_precision_not_an_error(self):
        metrics = compute_metrics(
            [
                _outcome("ambiguous", SentenceVerdict.NOT_AMBIGUOUS),
                _outcome("not_ambiguous", SentenceVerdict.NOT_AMBIGUOUS),
            ]
        )
        assert metrics.precision == 0.0 and metrics.recall == 0.0 and metrics.f1 == 0.0

    def test_empty_outcomes_are_rejected(self):
        with pytest.raises(ValueError):
            compute_metrics([])


class TestLoadDataset:
    def test_shipped_dataset_loads_and_has_both_classes(self):
        cases = load_dataset(SHIPPED_DATASET)
        labels = {case.label for case in cases}
        assert labels == {"ambiguous", "not_ambiguous"}
        assert len({case.id for case in cases}) == len(cases)

    def test_shipped_dataset_does_not_overlap_the_prompt_or_demo_cases(self):
        """The evaluation sentences must be independent of the seven demo
        sentences the prompts and README were tuned against."""
        demo_path = SHIPPED_DATASET.parents[1] / "examples" / "adjudication_cases.json"
        demo_texts = {c["text"] for c in json.loads(demo_path.read_text("utf-8"))["cases"]}
        assert not demo_texts & {case.text for case in load_dataset(SHIPPED_DATASET)}

    def _write(self, tmp_path, payload) -> Path:
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetError, match="could not read"):
            load_dataset(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "labels.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(DatasetError, match="not valid JSON"):
            load_dataset(path)

    @pytest.mark.parametrize("payload", [[], {"cases": []}, {"cases": "x"}])
    def test_requires_non_empty_cases_list(self, tmp_path, payload):
        with pytest.raises(DatasetError, match="non-empty 'cases'"):
            load_dataset(self._write(tmp_path, payload))

    def test_unknown_label_is_rejected_not_dropped(self, tmp_path):
        payload = {"cases": [{"id": "a", "text": "t", "label": "maybe"}]}
        with pytest.raises(DatasetError, match="case 0"):
            load_dataset(self._write(tmp_path, payload))

    def test_duplicate_ids_are_rejected(self, tmp_path):
        case = {"id": "a", "text": "t", "label": "ambiguous"}
        with pytest.raises(DatasetError, match="duplicate"):
            load_dataset(self._write(tmp_path, {"cases": [case, case]}))


class TestRunEvaluation:
    """``run_evaluation`` end to end against a scripted provider: real Phases
    1-3 evidence, fake LLM, no network."""

    def _adjudicator(self, verdict: str):
        from tests.test_pipeline_runner import _adjudicator

        return _adjudicator(verdict)

    def test_predictions_flow_into_metrics(self):
        from ambisense.config import load_settings
        from ambisense.evaluation import run_evaluation

        cases = [
            EvaluationCase(
                id="telescope", text="I saw the man with the telescope.",
                label="ambiguous",
            ),
        ]
        results = run_evaluation(
            load_settings(), cases, adjudicator=self._adjudicator("genuine_ambiguity")
        )

        assert [o.case_id for o in results.outcomes] == ["telescope"]
        assert results.outcomes[0].verdict is SentenceVerdict.AMBIGUOUS
        assert results.metrics.confusion["tp"] == 1
        assert results.metrics.accuracy == 1.0
        assert set(results.reports) == {"telescope"}
