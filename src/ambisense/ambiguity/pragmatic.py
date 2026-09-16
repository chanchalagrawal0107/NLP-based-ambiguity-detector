"""Pragmatic ambiguity candidate detection.

    "Can you open the window?"

* A literal question about the hearer's ability.
* An indirect request to open the window.

Measured parse of the example::

    Can/AUX/MD/aux -> open ,  you/PRON/nsubj -> open ,  open/VERB/ROOT

so the rule is: a modal auxiliary, a second-person subject, an action verb,
and a question mark.

Two deliberate restraints:

1. The detector reports that the sentence **can function** as either a
   question or a request. It never asserts which the speaker intended - that
   depends on situation, relationship and tone, none of which are in the text.
2. A question mark is required by default. "You can open the window" is a
   statement, not an indirect request, and flagging it would be noise.

Stative verbs are excluded: "Can you see the window?" asks about perception,
which the hearer cannot simply choose to perform, so the request reading is
much weaker.
"""

from __future__ import annotations

from ambisense.ambiguity.base import Detector
from ambisense.schemas import (
    AmbiguityCandidate,
    AmbiguityType,
    LinguisticAnalysis,
    SentenceInfo,
    TokenInfo,
)

_DEFAULT_MODALS = ("can", "could", "would", "will", "might")
_DEFAULT_MARKERS = (
    "do you know", "is it possible", "would you mind", "have you got",
    "i wonder",
)
_SECOND_PERSON = frozenset({"you", "your"})

# Verbs describing states or perceptions rather than performable actions.
_STATIVE_VERBS = frozenset({
    "see", "hear", "know", "think", "believe", "understand", "remember",
    "feel", "want", "like", "seem", "appear", "be", "have",
})


class PragmaticDetector(Detector):
    """Flags sentences that can function as either a question or a request."""

    name = "pragmatic"
    ambiguity_type = AmbiguityType.PRAGMATIC

    def detect(self, analysis: LinguisticAnalysis) -> list[AmbiguityCandidate]:
        candidates: list[AmbiguityCandidate] = []
        for sentence in analysis.sentences:
            candidate = self._modal_request(analysis, sentence)
            if candidate is not None:
                candidates.append(candidate)
                continue
            marker = self._indirect_marker(analysis, sentence)
            if marker is not None:
                candidates.append(marker)
        return sorted(candidates, key=lambda c: c.char_start)

    # -- rule 1 ----------------------------------------------------------

    def _modal_request(
        self, analysis: LinguisticAnalysis, sentence: SentenceInfo
    ) -> AmbiguityCandidate | None:
        modals = self.lowered_set("request_modals", _DEFAULT_MODALS)
        require_question = bool(self.option("require_question_mark", True))

        tokens = analysis.tokens_in_sentence(sentence.index)
        if not tokens:
            return None
        if require_question and not sentence.text.rstrip().endswith("?"):
            return None

        modal = self._first(
            tokens,
            lambda t: t.tag == "MD" and t.lemma in modals,
        )
        if modal is None:
            return None

        main_verb = analysis.token_by_index(modal.head_index)
        if main_verb is None or main_verb.pos not in {"VERB", "AUX"}:
            return None
        if main_verb.lemma in _STATIVE_VERBS:
            return None

        subject = self._first(
            analysis.children_of(main_verb.index),
            lambda t: t.dep in {"nsubj", "nsubjpass"}
            and t.text.lower() in _SECOND_PERSON,
        )
        if subject is None:
            return None

        return self.make_candidate(
            span_text=sentence.text,
            char_start=sentence.char_start,
            char_end=sentence.char_end,
            prior=0.7,
            explanation=(
                f"'{modal.text} {subject.text} {main_verb.text}...' is a modal "
                f"question addressed to the hearer about a performable action. "
                f"It can function as a question about ability or as an "
                f"indirect request. The text alone does not decide which."
            ),
            evidence={
                "rule": "modal_indirect_request",
                "modal": modal.text,
                "subject": subject.text,
                "action_verb": main_verb.text,
                "action_verb_lemma": main_verb.lemma,
                "is_question": sentence.text.rstrip().endswith("?"),
                "literal_force": "question about ability",
                "possible_indirect_force": "request to perform the action",
                "sentence_index": sentence.index,
            },
        )

    # -- rule 2 ----------------------------------------------------------

    def _indirect_marker(
        self, analysis: LinguisticAnalysis, sentence: SentenceInfo
    ) -> AmbiguityCandidate | None:
        markers = self.lowered_set("indirect_markers", _DEFAULT_MARKERS)
        lowered = sentence.text.lower()

        matched = self._first(sorted(markers), lambda m: m in lowered)
        if matched is None:
            return None

        return self.make_candidate(
            span_text=sentence.text,
            char_start=sentence.char_start,
            char_end=sentence.char_end,
            prior=0.5,
            explanation=(
                f"The sentence contains the conventional indirect marker "
                f"'{matched}', which often introduces a request rather than a "
                f"literal enquiry."
            ),
            evidence={
                "rule": "indirect_marker",
                "marker": matched,
                "is_question": sentence.text.rstrip().endswith("?"),
                "sentence_index": sentence.index,
            },
        )

    # -- helper ----------------------------------------------------------

    @staticmethod
    def _first(items, predicate):
        """First item satisfying ``predicate``, or ``None``."""
        for item in items:
            if predicate(item):
                return item
        return None
