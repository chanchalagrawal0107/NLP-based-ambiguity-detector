"""Phase 5: sentence-level rollup built from candidate-level adjudications.

Every case here hand-builds a list of ``CandidateAdjudication`` objects and
checks ``build_sentence_summary`` against them directly - no LLM, no spaCy,
no network. The five verdict branches and their precedence are the whole
contract, so each gets its own test.
"""

from __future__ import annotations

from ambisense.pipeline.summary import build_sentence_summary
from ambisense.schemas import (
    AdjudicationStatus,
    AdjudicationVerdict,
    AmbiguityCandidate,
    AmbiguityType,
    CandidateAdjudication,
    LLMJudgement,
    SentenceVerdict,
)


def _candidate(candidate_id: str, span: str = "word") -> AmbiguityCandidate:
    return AmbiguityCandidate(
        span_text=span,
        char_start=0,
        char_end=len(span),
        type_hint=AmbiguityType.SYNTACTIC,
        detector_name="syntactic",
        prior=0.7,
        explanation=f"Rule fired on {span}.",
    )


def _judged(
    candidate_id: str,
    verdict: str,
    *,
    interpretations: int = 2,
) -> CandidateAdjudication:
    return CandidateAdjudication(
        candidate_id=candidate_id,
        candidate=_candidate(candidate_id),
        status=AdjudicationStatus.ADJUDICATED,
        judgement=LLMJudgement(
            candidate_id=candidate_id,
            verdict=verdict,
            interpretations=[
                {"meaning": f"{candidate_id} reading {i}"}
                for i in range(interpretations)
            ],
            explanation="Justification.",
            confidence=0.8,
        ),
    )


def _unresolved(candidate_id: str, status: AdjudicationStatus) -> CandidateAdjudication:
    return CandidateAdjudication(
        candidate_id=candidate_id,
        candidate=_candidate(candidate_id),
        status=status,
        judgement=None,
        error="simulated failure",
    )


class TestNoCandidates:
    def test_empty_list_is_no_candidates(self):
        summary = build_sentence_summary([])
        assert summary.verdict is SentenceVerdict.NO_CANDIDATES
        assert summary.total_candidates == 0
        assert summary.genuine_candidate_ids == []
        assert summary.explanation


class TestAmbiguous:
    def test_any_genuine_makes_the_sentence_ambiguous(self):
        items = [
            _judged("c1", AdjudicationVerdict.GENUINE_AMBIGUITY),
            _judged("c2", AdjudicationVerdict.NOT_AMBIGUOUS, interpretations=1),
        ]
        summary = build_sentence_summary(items)
        assert summary.verdict is SentenceVerdict.AMBIGUOUS
        assert summary.total_candidates == 2
        assert summary.genuine_count == 1
        assert summary.not_ambiguous_count == 1
        assert summary.genuine_candidate_ids == ["c1"]

    def test_multiple_genuine_all_listed(self):
        items = [
            _judged("c1", AdjudicationVerdict.GENUINE_AMBIGUITY),
            _judged("c2", AdjudicationVerdict.GENUINE_AMBIGUITY),
        ]
        summary = build_sentence_summary(items)
        assert summary.genuine_candidate_ids == ["c1", "c2"]
        assert summary.genuine_count == 2


class TestNotAmbiguous:
    def test_all_rejected_is_not_ambiguous(self):
        items = [
            _judged("c1", AdjudicationVerdict.NOT_AMBIGUOUS, interpretations=1),
            _judged("c2", AdjudicationVerdict.NOT_AMBIGUOUS, interpretations=1),
        ]
        summary = build_sentence_summary(items)
        assert summary.verdict is SentenceVerdict.NOT_AMBIGUOUS
        assert summary.not_ambiguous_count == 2
        assert summary.genuine_candidate_ids == []


class TestUncertain:
    def test_uncertain_without_genuine_is_uncertain(self):
        items = [
            _judged("c1", AdjudicationVerdict.UNCERTAIN, interpretations=1),
            _judged("c2", AdjudicationVerdict.NOT_AMBIGUOUS, interpretations=1),
        ]
        summary = build_sentence_summary(items)
        assert summary.verdict is SentenceVerdict.UNCERTAIN
        assert summary.uncertain_count == 1

    def test_genuine_beats_uncertain(self):
        items = [
            _judged("c1", AdjudicationVerdict.UNCERTAIN, interpretations=1),
            _judged("c2", AdjudicationVerdict.GENUINE_AMBIGUITY),
        ]
        summary = build_sentence_summary(items)
        assert summary.verdict is SentenceVerdict.AMBIGUOUS


class TestIncomplete:
    def test_any_unresolved_candidate_makes_it_incomplete(self):
        items = [
            _judged("c1", AdjudicationVerdict.GENUINE_AMBIGUITY),
            _unresolved("c2", AdjudicationStatus.LLM_UNAVAILABLE),
        ]
        summary = build_sentence_summary(items)
        assert summary.verdict is SentenceVerdict.INCOMPLETE
        assert summary.unresolved_count == 1
        # Even a confirmed genuine candidate does not override incompleteness:
        # a missing judgement elsewhere is a fact about the run, not evidence
        # the unresolved candidate was harmless.
        assert summary.genuine_count == 1

    def test_incomplete_takes_precedence_over_no_candidates_case(self):
        items = [_unresolved("c1", AdjudicationStatus.LLM_NOT_CONFIGURED)]
        summary = build_sentence_summary(items)
        assert summary.verdict is SentenceVerdict.INCOMPLETE
        assert summary.total_candidates == 1
