"""Turn Phase 2 candidates and Phase 3 rankings into a compact evidence package.

The LLM should see the evidence it needs to judge a candidate, and nothing
else. Smaller prompts mean lower latency, fewer tokens and lower cost, and
less irrelevant material for the model to latch onto.

What is sent, per candidate
---------------------------
* ``candidate_id``, ``type``, ``span``
* ``detector_reason`` - the detector's one-line explanation
* ``rule_evidence``  - the detector's evidence dict, minus internals
* ``semantic_evidence`` - for lexical candidates only: the top few senses with
  truncated glosses and similarity scores, and the margin

What is deliberately withheld
-----------------------------
* Token indices, thresholds, detector names, duplicated definitions - project
  internals with no bearing on meaning.
* ``prior`` (the detector's signal strength). It is an uncalibrated number; the
  LLM is asked to judge plausibility, and showing it a number would invite it
  to anchor on that number instead.
* ``resolved_by_context`` from Phase 3. It is a thresholded restatement of the
  margin, and it is known to be ``true`` for a *wrong* sense in the crane
  case. Sending "resolved: true" would push the model towards the very error
  this layer exists to catch. The raw margin is sent instead.

Output is serialised with sorted keys, so the same input always yields a
byte-identical prompt - which also makes the response cache effective.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from ambisense.config import AdjudicationConfig
from ambisense.schemas import (
    AmbiguityCandidate,
    AmbiguityType,
    CandidateAdjudication,
    SenseRanking,
)

#: Evidence keys that are project internals rather than linguistic evidence.
_WITHHELD_KEYS = frozenset({
    "detector",
    "thresholds",
    "sample_definitions",
    "antecedent_details",
    "also_detected_by",
})

SIMILARITY_NOTE = (
    "cosine similarity between averaged static word vectors of the context "
    "and of each gloss; supporting evidence, not a probability"
)


def candidate_ids(count: int) -> list[str]:
    """Stable, human-readable identifiers: c1, c2, ..."""
    return [f"c{position}" for position in range(1, count + 1)]


def compact_rule_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evidence.items()
        if key not in _WITHHELD_KEYS and not key.endswith("_index")
    }


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def prompt_senses(
    ranking: Optional[SenseRanking],
    config: AdjudicationConfig,
) -> list:
    """The ranked senses actually shown to the LLM for one candidate.

    The single definition used both to build the prompt and to validate the
    model's ``selected_sense``, so the set of acceptable keys can never drift
    from the set the model was offered.
    """
    if ranking is None or config.max_senses_in_prompt == 0:
        return []
    if not ranking.status.is_usable:
        return []
    return list(ranking.senses[: config.max_senses_in_prompt])


def prompt_sense_keys(
    ranking: Optional[SenseRanking],
    config: AdjudicationConfig,
) -> frozenset[str]:
    return frozenset(sense.sense_key for sense in prompt_senses(ranking, config))


def semantic_payload(
    ranking: Optional[SenseRanking],
    config: AdjudicationConfig,
) -> Optional[dict[str, Any]]:
    if ranking is None or config.max_senses_in_prompt == 0:
        return None
    if not ranking.status.is_usable:
        return {"status": ranking.status.value}
    return {
        "status": ranking.status.value,
        "context_words": list(ranking.context_words),
        "measure": SIMILARITY_NOTE,
        "margin_top_two": ranking.margin,
        "ranked_senses": [
            {
                "rank": sense.rank,
                "sense": sense.sense_key,
                "gloss": _truncate(sense.definition, config.max_gloss_chars),
                "similarity": (
                    None if sense.context_similarity is None
                    else round(sense.context_similarity, 3)
                ),
            }
            for sense in prompt_senses(ranking, config)
        ],
    }


def match_ranking(
    candidate: AmbiguityCandidate,
    rankings: Sequence[SenseRanking],
) -> Optional[SenseRanking]:
    """Phase 3 rankings are linked to candidates by exact character offsets."""
    if candidate.type_hint is not AmbiguityType.LEXICAL:
        return None
    for ranking in rankings:
        if (ranking.char_start == candidate.char_start
                and ranking.char_end == candidate.char_end):
            return ranking
    return None


def candidate_payload(
    candidate_id: str,
    candidate: AmbiguityCandidate,
    ranking: Optional[SenseRanking],
    config: AdjudicationConfig,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "type": candidate.type_hint.value,
        "span": candidate.span_text,
        "detector_reason": candidate.explanation,
        "rule_evidence": compact_rule_evidence(candidate.evidence),
        "semantic_evidence": semantic_payload(ranking, config),
    }


def rewrite_payload(adjudication: CandidateAdjudication) -> dict[str, Any]:
    judgement = adjudication.judgement
    return {
        "candidate_id": adjudication.candidate_id,
        "type": adjudication.candidate.type_hint.value,
        "span": adjudication.candidate.span_text,
        "interpretations": [
            interpretation.meaning
            for interpretation in (judgement.interpretations if judgement else [])
        ],
    }


def to_json(payload: Any) -> str:
    """Deterministic, compact serialisation for prompts."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ": "))
