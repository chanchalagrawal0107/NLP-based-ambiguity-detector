"""Deterministic comparison of the LLM's selected sense with Phase 3's ranking.

Why this is computed in code
----------------------------
An earlier version asked the LLM to *report* whether it agreed with the Phase 3
ranking. On a sentence about a machine lifting a container, the model correctly
chose the lifting-machine sense - overruling Phase 3, which ranked an animal
sense first - yet reported that it "agreed". The model's verdict was right; its
description of its own relationship to the evidence was wrong.

That relationship is a fact the application can calculate exactly, so it no
longer comes from the model. The LLM names the sense it selected by its stable
WordNet key, and this function compares that key with Phase 3's rank-1 key.

The rule
--------
    phase3_top_sense = ranking.top_sense_key   (only when the ranking is usable)
    agrees_with_phase3 = selected_sense == phase3_top_sense
                         when BOTH are present, otherwise None

``None`` is not "disagrees". It means there is nothing to compare: the
candidate has no usable sense ranking (every non-lexical candidate), or the
LLM selected no single sense (none of the offered senses fits, or it judged the
span genuinely ambiguous between senses).
"""

from __future__ import annotations

from typing import Optional

from ambisense.schemas import SenseRanking


def phase3_top_sense(ranking: Optional[SenseRanking]) -> Optional[str]:
    """Phase 3's rank-1 sense key, or None when no usable ranking exists."""
    if ranking is None or not ranking.status.is_usable:
        return None
    return ranking.top_sense_key


def agrees_with_phase3(
    ranking: Optional[SenseRanking],
    selected_sense: Optional[str],
) -> Optional[bool]:
    """Compare stable sense keys. Never inspects explanations or confidence."""
    top = phase3_top_sense(ranking)
    if top is None or selected_sense is None:
        return None
    return selected_sense == top
