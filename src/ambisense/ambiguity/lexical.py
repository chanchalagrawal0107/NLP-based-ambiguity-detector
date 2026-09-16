"""Lexical ambiguity candidate detection.

    "I went to the bank."  ->  bank

A word is proposed as a lexical candidate when WordNet records several senses
for it *under the part of speech spaCy assigned*, and those senses spread
across several distinct semantic domains.

Filtering, in order of application:

1. **POS filter** - only NOUN, VERB, ADJ, ADV are considered. Function words
   are never lexically ambiguous in the sense that matters here.
2. **Stoplist** - very frequent verbs and nouns ("go", "make", "time") carry
   many WordNet senses but are rarely perceived as ambiguous.
3. **Length filter** - very short words are dominated by function words.
4. **Sense and domain thresholds** - see ``wordnet_support``.
5. **Per-sentence cap** - keep only the strongest few candidates so the
   report stays readable.

The detector reports *where* and *why*. It does not state what the competing
meanings are - producing interpretations is the LLM's job in Phase 4.
"""

from __future__ import annotations

from collections import defaultdict

from ambisense.ambiguity.base import Detector
from ambisense.ambiguity.wordnet_support import SenseProfile, sense_profile
from ambisense.logging_setup import get_logger
from ambisense.schemas import AmbiguityCandidate, AmbiguityType, LinguisticAnalysis

logger = get_logger(__name__)

_DEFAULT_ALLOWED_POS = ("NOUN", "VERB", "ADJ", "ADV")
_DEFAULT_STOPLIST = (
    "be", "have", "do", "get", "make", "take", "go", "come", "give", "say",
    "know", "thing", "way", "time", "year", "day", "people", "use", "work",
)

# Used to normalise the prior. A word with this many senses or more scores
# full marks on the sense component; chosen because it is roughly where
# WordNet's most polysemous everyday words sit (see "go" at 30).
_SENSE_SATURATION = 12.0
_DOMAIN_SATURATION = 6.0


class LexicalDetector(Detector):
    """Flags content words with several distinct WordNet senses."""

    name = "lexical"
    ambiguity_type = AmbiguityType.LEXICAL

    def detect(self, analysis: LinguisticAnalysis) -> list[AmbiguityCandidate]:
        min_senses = int(self.option("min_senses", 4))
        min_domains = int(self.option("min_domains", 3))
        min_length = int(self.option("min_word_length", 3))
        max_per_sentence = int(self.option("max_candidates_per_sentence", 3))
        allowed_pos = self.lowered_set("allowed_pos", _DEFAULT_ALLOWED_POS)
        stoplist = self.lowered_set("stoplist", _DEFAULT_STOPLIST)

        by_sentence: dict[int, list[AmbiguityCandidate]] = defaultdict(list)
        seen_lemmas: set[tuple[int, str]] = set()

        for token in analysis.tokens:
            if token.pos.lower() not in allowed_pos:
                continue
            if not token.is_alpha or len(token.text) < min_length:
                continue
            if token.lemma in stoplist or token.is_stop:
                continue

            # One candidate per lemma per sentence: repeating "bank" twice in
            # a sentence is one ambiguity, not two.
            lemma_key = (token.sentence_index, token.lemma)
            if lemma_key in seen_lemmas:
                continue

            profile = sense_profile(token.lemma, token.pos)
            if profile.sense_count < min_senses:
                continue
            if profile.domain_count < min_domains:
                continue

            seen_lemmas.add(lemma_key)
            by_sentence[token.sentence_index].append(
                self._build(token, profile, min_senses, min_domains, analysis)
            )

        # Apply the per-sentence cap, strongest first, then restore document
        # order so the caller always receives a stable sequence.
        kept: list[AmbiguityCandidate] = []
        for sentence_index in sorted(by_sentence):
            ranked = sorted(
                by_sentence[sentence_index],
                key=lambda c: (-c.prior, c.char_start),
            )
            if len(ranked) > max_per_sentence:
                logger.debug(
                    "Sentence %d: capping %d lexical candidates to %d",
                    sentence_index, len(ranked), max_per_sentence,
                )
            kept.extend(ranked[:max_per_sentence])
        return sorted(kept, key=lambda c: c.char_start)

    # -- internals -------------------------------------------------------

    def _build(
        self,
        token,
        profile: SenseProfile,
        min_senses: int,
        min_domains: int,
        analysis: LinguisticAnalysis,
    ) -> AmbiguityCandidate:
        short_domains = [domain.split(".")[-1] for domain in profile.domains]
        return self.make_candidate(
            span_text=token.text,
            char_start=token.char_start,
            char_end=token.char_end,
            prior=self._prior(profile),
            explanation=(
                f"'{token.text}' has {profile.sense_count} WordNet senses as a "
                f"{token.pos.lower()}, spanning {profile.domain_count} semantic "
                f"domains ({', '.join(short_domains)}). Several distinct "
                f"meanings are therefore available in principle."
            ),
            evidence={
                "rule": "wordnet_polysemy",
                "lemma": token.lemma,
                "pos": token.pos,
                "wordnet_pos": profile.wordnet_pos,
                "sense_count": profile.sense_count,
                "domain_count": profile.domain_count,
                "domains": short_domains,
                "sample_definitions": list(profile.example_definitions),
                "thresholds": {
                    "min_senses": min_senses,
                    "min_domains": min_domains,
                },
                "sentence_index": token.sentence_index,
            },
        )

    @staticmethod
    def _prior(profile: SenseProfile) -> float:
        """Signal strength from sense count and domain spread.

        Deterministic and deliberately simple: the mean of two saturating
        ratios, so it rises with both measures and is bounded at 1.0. It is a
        ranking aid, not a probability - see the module docstring in
        ``base.py``.
        """
        sense_component = min(profile.sense_count / _SENSE_SATURATION, 1.0)
        domain_component = min(profile.domain_count / _DOMAIN_SATURATION, 1.0)
        return round((sense_component + domain_component) / 2.0, 4)
