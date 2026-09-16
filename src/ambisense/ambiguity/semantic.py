"""Semantic ambiguity candidate detection.

The sentence is structurally clear, but the semantic roles are not fixed.

Rule 1 - "tough constructions"
------------------------------
    "The chicken is ready to eat."

* The chicken is about to eat something  (subject = agent)
* The chicken is ready to be eaten       (subject = patient)

Detected from the dependency parse, which on the shipped model gives::

    chicken/nsubj -> is ,  ready/acomp -> is ,  eat/xcomp -> ready

So: an adjective from a configured list, complemented by a to-infinitive whose
verb has **no object**. The missing object is what leaves the subject free to
fill either role. Verbs that cannot take an object at all ("ready to go") are
excluded, because the patient reading is then unavailable.

Rule 2 - noun-noun compounds
----------------------------
    "student protest"  ->  a protest by students? about students?

A compound leaves the relation between the two nouns unstated. This fires
often, so it carries a low prior and is reported as weak evidence.

This is deliberately a small rule set. Full semantic-role analysis is out of
scope, and the detector never states what the readings are - that is Phase 4.
"""

from __future__ import annotations

from ambisense.ambiguity.base import Detector, span_of_tokens, subtree_tokens
from ambisense.schemas import (
    AmbiguityCandidate,
    AmbiguityType,
    LinguisticAnalysis,
    TokenInfo,
)

_DEFAULT_TOUGH_ADJECTIVES = (
    "ready", "easy", "hard", "difficult", "tough", "simple", "safe",
    "dangerous", "pleasant", "nice", "good", "fun", "impossible",
)
_DEFAULT_INTRANSITIVE = (
    "go", "come", "arrive", "leave", "sleep", "sit", "stand", "wait",
)

#: Dependency labels marking an object of a verb.
_OBJECT_DEPS = frozenset({"dobj", "obj", "dative", "attr", "oprd"})


class SemanticDetector(Detector):
    """Flags constructions permitting more than one semantic role assignment."""

    name = "semantic"
    ambiguity_type = AmbiguityType.SEMANTIC

    def detect(self, analysis: LinguisticAnalysis) -> list[AmbiguityCandidate]:
        candidates: list[AmbiguityCandidate] = []
        if bool(self.option("detect_tough_constructions", True)):
            candidates.extend(self._tough_constructions(analysis))
        if bool(self.option("detect_noun_compounds", True)):
            candidates.extend(self._noun_compounds(analysis))
        return sorted(candidates, key=lambda c: c.char_start)

    # -- rule 1 ----------------------------------------------------------

    def _tough_constructions(
        self, analysis: LinguisticAnalysis
    ) -> list[AmbiguityCandidate]:
        adjectives = self.lowered_set(
            "tough_adjectives", _DEFAULT_TOUGH_ADJECTIVES
        )
        intransitive = self.lowered_set(
            "intransitive_verbs", _DEFAULT_INTRANSITIVE
        )
        found: list[AmbiguityCandidate] = []

        for token in analysis.tokens:
            if token.pos != "ADJ" or token.lemma not in adjectives:
                continue

            infinitive = self._infinitive_complement(analysis, token)
            if infinitive is None:
                continue
            if infinitive.lemma in intransitive:
                continue
            # An explicit object fixes the roles: "ready to eat the corn".
            if self._has_object(analysis, infinitive):
                continue

            subject = self._subject_of(analysis, token)
            phrase_tokens = [token] + subtree_tokens(analysis, infinitive.index)
            span_text, start, end = span_of_tokens(phrase_tokens, analysis)

            subject_text = subject.text if subject else "the subject"
            found.append(
                self.make_candidate(
                    span_text=span_text,
                    char_start=start,
                    char_end=end,
                    prior=0.7,
                    explanation=(
                        f"In '{span_text}', the verb '{infinitive.text}' has no "
                        f"object, so '{subject_text}' can be read as either the "
                        f"one performing '{infinitive.lemma}' or the one it is "
                        f"performed on."
                    ),
                    evidence={
                        "rule": "tough_construction",
                        "construction": span_text,
                        "adjective": token.text,
                        "infinitive": infinitive.text,
                        "infinitive_lemma": infinitive.lemma,
                        "infinitive_has_object": False,
                        "subject": subject_text,
                        "reason": (
                            "The construction permits alternative semantic "
                            "role interpretations for the subject."
                        ),
                        "sentence_index": token.sentence_index,
                    },
                )
            )
        return found

    @staticmethod
    def _infinitive_complement(
        analysis: LinguisticAnalysis, adjective: TokenInfo
    ) -> TokenInfo | None:
        """Find a to-infinitive complementing this adjective."""
        for child in analysis.children_of(adjective.index):
            if child.pos != "VERB" or child.dep not in {"xcomp", "ccomp", "acl"}:
                continue
            has_to = any(
                grandchild.tag == "TO" or grandchild.lemma == "to"
                for grandchild in analysis.children_of(child.index)
            )
            if has_to:
                return child
        return None

    @staticmethod
    def _has_object(analysis: LinguisticAnalysis, verb: TokenInfo) -> bool:
        return any(
            child.dep in _OBJECT_DEPS
            for child in analysis.children_of(verb.index)
        )

    @staticmethod
    def _subject_of(
        analysis: LinguisticAnalysis, adjective: TokenInfo
    ) -> TokenInfo | None:
        """Locate the subject the adjective is predicated of."""
        head = analysis.token_by_index(adjective.head_index)
        for candidate in (head, adjective):
            if candidate is None:
                continue
            for child in analysis.children_of(candidate.index):
                if child.dep in {"nsubj", "nsubjpass"}:
                    return child
        return None

    # -- rule 2 ----------------------------------------------------------

    def _noun_compounds(
        self, analysis: LinguisticAnalysis
    ) -> list[AmbiguityCandidate]:
        found: list[AmbiguityCandidate] = []
        for token in analysis.tokens:
            if token.dep != "compound" or token.pos not in {"NOUN", "PROPN"}:
                continue
            head = analysis.token_by_index(token.head_index)
            if head is None or head.pos not in {"NOUN", "PROPN"}:
                continue
            # Proper-name compounds ("New York") are not role-ambiguous.
            if token.pos == "PROPN" and head.pos == "PROPN":
                continue

            start, end = token.char_start, head.char_end
            span_text = analysis.text[start:end]
            found.append(
                self.make_candidate(
                    span_text=span_text,
                    char_start=start,
                    char_end=end,
                    prior=0.35,
                    explanation=(
                        f"In the compound '{span_text}', the relation between "
                        f"'{token.text}' and '{head.text}' is not stated and "
                        f"can be read in more than one way."
                    ),
                    evidence={
                        "rule": "noun_compound",
                        "modifier_noun": token.text,
                        "head_noun": head.text,
                        "reason": (
                            "The semantic relation within a noun-noun compound "
                            "is unexpressed."
                        ),
                        "sentence_index": token.sentence_index,
                    },
                )
            )
        return found
