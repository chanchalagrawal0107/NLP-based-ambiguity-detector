"""Referential ambiguity candidate detection.

    "John told David that he was late."  ->  he = John? he = David?

This is **not** a coreference resolution system. It does not try to decide
which antecedent is correct; it reports that more than one survives filtering.
That is the honest limit of a rule-based layer, and choosing between them is
exactly the kind of judgement the Phase 4 LLM is there to make.

Filtering pipeline for each pronoun:

1. **Tag filter** - only ``PRP``/``PRP$`` count. Demonstratives are excluded
   because spaCy tags "that" as a subordinating conjunction in the sentence
   above, which would produce spurious searches.
2. **Window** - antecedents are sought in the pronoun's own sentence and a
   configurable number of preceding ones.
3. **Position** - an antecedent must appear before the pronoun.
4. **Number agreement** - a plural pronoun needs a plural antecedent, and
   vice versa.
5. **Animacy agreement** - "he"/"she" need an animate candidate; "it" needs
   an inanimate one.
6. **Self-exclusion** - the pronoun's own noun chunk is not its antecedent.

If two or more candidates survive, the pronoun is reported.
"""

from __future__ import annotations

from ambisense.ambiguity.base import Detector
from ambisense.schemas import (
    AmbiguityCandidate,
    AmbiguityType,
    LinguisticAnalysis,
    NounChunkInfo,
    TokenInfo,
)

_DEFAULT_PRONOUNS = (
    "he", "she", "it", "they", "him", "her", "them", "his", "hers",
    "their", "theirs",
)
_DEFAULT_PRONOUN_TAGS = ("PRP", "PRP$")

#: Pronouns requiring an animate antecedent.
_ANIMATE_PRONOUNS = frozenset({"he", "she", "him", "her", "his", "hers"})
#: Pronouns requiring an inanimate antecedent.
_INANIMATE_PRONOUNS = frozenset({"it", "its"})
#: Pronouns requiring a plural antecedent.
_PLURAL_PRONOUNS = frozenset({"they", "them", "their", "theirs"})
#: Pronouns requiring a singular antecedent.
_SINGULAR_PRONOUNS = frozenset({"he", "she", "him", "her", "his", "hers", "it", "its"})


class ReferentialDetector(Detector):
    """Flags pronouns with more than one surviving antecedent candidate."""

    name = "referential"
    ambiguity_type = AmbiguityType.REFERENTIAL

    def detect(self, analysis: LinguisticAnalysis) -> list[AmbiguityCandidate]:
        pronouns = self.lowered_set("pronouns", _DEFAULT_PRONOUNS)
        pronoun_tags = {
            tag.upper()
            for tag in self.option("pronoun_tags", list(_DEFAULT_PRONOUN_TAGS))
        }
        window = int(self.option("antecedent_window_sentences", 2))
        min_candidates = int(self.option("min_candidates", 2))

        found: list[AmbiguityCandidate] = []
        for token in analysis.tokens:
            if token.tag.upper() not in pronoun_tags:
                continue
            if token.lemma not in pronouns and token.text.lower() not in pronouns:
                continue

            antecedents = self._antecedents(analysis, token, window)
            if len(antecedents) < min_candidates:
                continue

            found.append(self._build(token, antecedents, analysis, window))
        return found

    # -- internals -------------------------------------------------------

    def _antecedents(
        self,
        analysis: LinguisticAnalysis,
        pronoun: TokenInfo,
        window: int,
    ) -> list[NounChunkInfo]:
        surface = pronoun.text.lower()
        earliest_sentence = max(0, pronoun.sentence_index - window)

        surviving: list[NounChunkInfo] = []
        for chunk in analysis.noun_chunks:
            if not (earliest_sentence <= chunk.sentence_index
                    <= pronoun.sentence_index):
                continue
            if chunk.char_start >= pronoun.char_start:
                continue
            # A pronoun is not its own antecedent.
            if chunk.root_pos == "PRON":
                continue
            if not self._number_agrees(surface, chunk):
                continue
            if not self._animacy_agrees(surface, chunk):
                continue
            surviving.append(chunk)
        return surviving

    @staticmethod
    def _number_agrees(pronoun_surface: str, chunk: NounChunkInfo) -> bool:
        if pronoun_surface in _PLURAL_PRONOUNS:
            return chunk.is_plural
        if pronoun_surface in _SINGULAR_PRONOUNS:
            return not chunk.is_plural
        return True

    @staticmethod
    def _animacy_agrees(pronoun_surface: str, chunk: NounChunkInfo) -> bool:
        if pronoun_surface in _ANIMATE_PRONOUNS:
            return chunk.is_animate_candidate
        if pronoun_surface in _INANIMATE_PRONOUNS:
            return not chunk.is_animate_candidate
        return True

    def _build(
        self,
        pronoun: TokenInfo,
        antecedents: list[NounChunkInfo],
        analysis: LinguisticAnalysis,
        window: int,
    ) -> AmbiguityCandidate:
        texts = [chunk.text for chunk in antecedents]
        return self.make_candidate(
            span_text=pronoun.text,
            char_start=pronoun.char_start,
            char_end=pronoun.char_end,
            prior=self._prior(len(antecedents)),
            explanation=(
                f"The pronoun '{pronoun.text}' has {len(antecedents)} possible "
                f"antecedents that agree in number and animacy: "
                f"{', '.join(repr(text) for text in texts)}. Nothing in the "
                f"sentence structure selects between them."
            ),
            evidence={
                "rule": "multiple_antecedents",
                "pronoun": pronoun.text,
                "pronoun_tag": pronoun.tag,
                "antecedent_count": len(antecedents),
                "antecedents": texts,
                "antecedent_details": [
                    {
                        "text": chunk.text,
                        "root": chunk.root_text,
                        "dep": chunk.root_dep,
                        "is_plural": chunk.is_plural,
                        "is_animate_candidate": chunk.is_animate_candidate,
                        "sentence_index": chunk.sentence_index,
                    }
                    for chunk in antecedents
                ],
                "window_sentences": window,
                "sentence_index": pronoun.sentence_index,
            },
        )

    @staticmethod
    def _prior(count: int) -> float:
        """More surviving antecedents means a stronger signal.

        Two candidates is the classic ambiguous case; beyond that the increase
        is modest, so the scale saturates rather than growing without bound.
        """
        if count <= 1:
            return 0.0
        return round(min(0.55 + 0.15 * (count - 1), 0.95), 4)
