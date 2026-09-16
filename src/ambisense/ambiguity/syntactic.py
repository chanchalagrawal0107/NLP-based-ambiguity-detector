"""Syntactic ambiguity candidate detection.

Two conservative, explainable rules.

Rule 1 - prepositional-phrase attachment
----------------------------------------
    "I saw the man with the telescope."

The parser commits to one attachment and discards the alternative, so the
parse alone never reveals the ambiguity. Measured on the shipped model::

    I saw the man with the telescope.  ->  with -> man  (noun attachment)
    The cat sat on the mat.            ->  on   -> sat  (verb attachment)

A naive rule ("a preposition and a nearby noun") fires on both. The
discriminator is **linear position**: genuine PP-attachment ambiguity needs a
competing noun lying *between* the verb and the preposition.

* "the man" sits between "saw" and "with"   -> two attachment sites -> flag
* nothing sits between "sat" and "on"        -> one attachment site  -> ignore

So the detector reports the ambiguity whichever way the parser resolved it,
and stays silent when no genuine alternative exists.

Rule 2 - coordination scope
---------------------------
    "The old men and women sat outside."

When a modifier attaches to the first noun of a coordination, it may or may
not distribute over the second: *old* may modify only *men*, or *men and
women* as a whole. Detected as an ``amod``/``compound`` child on a noun that
also has a ``conj`` sibling.
"""

from __future__ import annotations

from typing import Optional

from ambisense.ambiguity.base import Detector, span_of_tokens, subtree_tokens
from ambisense.schemas import (
    AmbiguityCandidate,
    AmbiguityType,
    LinguisticAnalysis,
    TokenInfo,
)

_DEFAULT_PREPOSITIONS = (
    "with", "in", "on", "at", "from", "by", "near", "under", "over",
    "through", "behind", "beside",
)

# Dependency labels that mark a noun able to host a prepositional phrase.
_NOUN_HOST_DEPS = frozenset({"dobj", "obj", "pobj", "nsubj", "attr", "dative"})
_MODIFIER_DEPS = frozenset({"amod", "compound", "nmod"})


class SyntacticDetector(Detector):
    """Flags structural configurations permitting more than one parse."""

    name = "syntactic"
    ambiguity_type = AmbiguityType.SYNTACTIC

    def detect(self, analysis: LinguisticAnalysis) -> list[AmbiguityCandidate]:
        candidates: list[AmbiguityCandidate] = []
        if bool(self.option("detect_pp_attachment", True)):
            candidates.extend(self._pp_attachment(analysis))
        if bool(self.option("detect_coordination_scope", True)):
            candidates.extend(self._coordination(analysis))
        return sorted(candidates, key=lambda c: c.char_start)

    # -- rule 1 ----------------------------------------------------------

    def _pp_attachment(
        self, analysis: LinguisticAnalysis
    ) -> list[AmbiguityCandidate]:
        prepositions = self.lowered_set(
            "attachment_prepositions", _DEFAULT_PREPOSITIONS
        )
        found: list[AmbiguityCandidate] = []

        for token in analysis.tokens:
            if token.dep != "prep" or token.lemma not in prepositions:
                continue

            head = analysis.token_by_index(token.head_index)
            if head is None:
                continue

            verb, noun = self._attachment_sites(analysis, token, head)
            if verb is None or noun is None:
                continue

            # The competing noun must lie between the verb and the
            # preposition; otherwise there is only one attachment site.
            if not (verb.char_start < noun.char_start < token.char_start):
                continue

            phrase_tokens = subtree_tokens(analysis, token.index)
            span_text, start, end = span_of_tokens(phrase_tokens, analysis)
            if not span_text:
                continue

            attached_to = "noun" if head.index == noun.index else "verb"
            found.append(
                self.make_candidate(
                    span_text=span_text,
                    char_start=start,
                    char_end=end,
                    prior=0.75,
                    explanation=(
                        f"The phrase '{span_text}' can attach either to the "
                        f"verb '{verb.text}' or to the noun '{noun.text}'. "
                        f"The parser chose the {attached_to}, but the other "
                        f"attachment is structurally available."
                    ),
                    evidence={
                        "rule": "pp_attachment",
                        "preposition": token.text,
                        "parser_attached_to": head.text,
                        "parser_attachment_type": attached_to,
                        "alternative_verb_site": verb.text,
                        "alternative_noun_site": noun.text,
                        "verb_index": verb.index,
                        "noun_index": noun.index,
                        "preposition_index": token.index,
                        "sentence_index": token.sentence_index,
                    },
                )
            )
        return found

    @staticmethod
    def _attachment_sites(
        analysis: LinguisticAnalysis,
        preposition: TokenInfo,
        head: TokenInfo,
    ) -> tuple[Optional[TokenInfo], Optional[TokenInfo]]:
        """Find the competing verb and noun attachment sites.

        Works from whichever site the parser picked, then looks for the other:

        * parser attached the PP to a **noun** -> the competing site is that
          noun's governing verb;
        * parser attached the PP to a **verb** -> the competing site is an
          object noun of that verb.
        """
        if head.pos in {"NOUN", "PROPN", "PRON"}:
            noun = head
            verb = analysis.token_by_index(head.head_index)
            if verb is None or verb.pos not in {"VERB", "AUX"}:
                return None, None
            return verb, noun

        if head.pos in {"VERB", "AUX"}:
            verb = head
            objects = [
                child for child in analysis.children_of(verb.index)
                if child.dep in _NOUN_HOST_DEPS
                and child.pos in {"NOUN", "PROPN"}
                and child.char_start < preposition.char_start
                and child.char_start > verb.char_start
            ]
            if not objects:
                return None, None
            # Nearest preceding object is the most plausible competitor.
            return verb, max(objects, key=lambda t: t.char_start)

        return None, None

    # -- rule 2 ----------------------------------------------------------

    def _coordination(
        self, analysis: LinguisticAnalysis
    ) -> list[AmbiguityCandidate]:
        found: list[AmbiguityCandidate] = []

        for token in analysis.tokens:
            if token.dep != "conj" or token.pos not in {"NOUN", "PROPN"}:
                continue
            first = analysis.token_by_index(token.head_index)
            if first is None or first.pos not in {"NOUN", "PROPN"}:
                continue

            modifiers = [
                child for child in analysis.children_of(first.index)
                if child.dep in _MODIFIER_DEPS
                and child.char_start < first.char_start
            ]
            if not modifiers:
                continue

            modifier = min(modifiers, key=lambda t: t.char_start)
            start = modifier.char_start
            end = token.char_end
            span_text = analysis.text[start:end]

            found.append(
                self.make_candidate(
                    span_text=span_text,
                    char_start=start,
                    char_end=end,
                    prior=0.65,
                    explanation=(
                        f"'{modifier.text}' modifies '{first.text}', which is "
                        f"coordinated with '{token.text}'. The modifier may "
                        f"apply to '{first.text}' alone or to both nouns."
                    ),
                    evidence={
                        "rule": "coordination_scope",
                        "modifier": modifier.text,
                        "modifier_dep": modifier.dep,
                        "first_conjunct": first.text,
                        "second_conjunct": token.text,
                        "sentence_index": token.sentence_index,
                    },
                )
            )
        return found
