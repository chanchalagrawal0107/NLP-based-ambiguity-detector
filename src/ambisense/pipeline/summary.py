"""Phase 5: roll candidate-level adjudications into one sentence verdict.

Deliberately produces no numeric score. ``build_sentence_summary`` only counts
verdicts that already exist on each ``CandidateAdjudication`` and applies a
fixed precedence rule to pick one ``SentenceVerdict`` - every field of the
result can be recomputed by hand from the input list.
"""

from __future__ import annotations

from ambisense.schemas import (
    AdjudicationStatus,
    AdjudicationVerdict,
    CandidateAdjudication,
    SentenceSummary,
    SentenceVerdict,
)


def build_sentence_summary(
    adjudications: list[CandidateAdjudication],
) -> SentenceSummary:
    """Deterministic rollup. Precedence: incomplete > no candidates > ambiguous
    > uncertain > not ambiguous.

    A single unresolved candidate makes the whole sentence ``INCOMPLETE``:
    the LLM failure is a fact about the run, not evidence that the missing
    candidate was harmless, so the sentence cannot be confidently classified
    either way.
    """
    total = len(adjudications)
    unresolved = [
        a for a in adjudications if a.status is not AdjudicationStatus.ADJUDICATED
    ]
    genuine = [a for a in adjudications if a.is_genuine]
    not_ambiguous = [
        a for a in adjudications
        if a.judgement is not None
        and a.judgement.verdict is AdjudicationVerdict.NOT_AMBIGUOUS
    ]
    uncertain = [
        a for a in adjudications
        if a.judgement is not None
        and a.judgement.verdict is AdjudicationVerdict.UNCERTAIN
    ]

    counts = dict(
        total_candidates=total,
        genuine_count=len(genuine),
        not_ambiguous_count=len(not_ambiguous),
        uncertain_count=len(uncertain),
        unresolved_count=len(unresolved),
        genuine_candidate_ids=[a.candidate_id for a in genuine],
    )

    if unresolved:
        return SentenceSummary(
            verdict=SentenceVerdict.INCOMPLETE,
            explanation=(
                f"{len(unresolved)} of {total} candidate(s) could not be "
                "judged, so the sentence cannot be confidently classified."
            ),
            **counts,
        )

    if total == 0:
        return SentenceSummary(
            verdict=SentenceVerdict.NO_CANDIDATES,
            explanation=(
                "No structural candidates were found by the rule-based "
                "detectors. This is not proof the text is unambiguous - "
                "see the README limitations."
            ),
            **counts,
        )

    if genuine:
        return SentenceSummary(
            verdict=SentenceVerdict.AMBIGUOUS,
            explanation=(
                f"{len(genuine)} of {total} candidate(s) confirmed as "
                "genuine ambiguities."
            ),
            **counts,
        )

    if uncertain:
        return SentenceSummary(
            verdict=SentenceVerdict.UNCERTAIN,
            explanation=(
                f"{len(uncertain)} of {total} candidate(s) were judged "
                "uncertain and none were confirmed genuine; the evidence "
                "does not allow a confident verdict."
            ),
            **counts,
        )

    return SentenceSummary(
        verdict=SentenceVerdict.NOT_AMBIGUOUS,
        explanation=(
            f"All {total} candidate(s) were judged not ambiguous."
        ),
        **counts,
    )
