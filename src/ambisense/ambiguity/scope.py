"""Scope ambiguity candidate detection.

    "Every student didn't submit the assignment."

* No student submitted it       (negation scopes over the quantifier)
* Not all students submitted it (the quantifier scopes over negation)

Two rules, both requiring the trigger words to sit in the **same clause** -
otherwise "Every student passed, but I didn't attend" would be flagged, where
the quantifier and the negation belong to different clauses and cannot
interact.

Rule 1 - quantifier + negation. Measured parse of the example above::

    Every/det -> student ,  n't/neg -> submit ,  student/nsubj -> submit

Both resolve to the clause headed by "submit", so they interact.

Rule 2 - two quantifiers in one clause ("Every student read some book"),
where the order of evaluation changes the meaning.

Note on the word lists: "no" appears in both the quantifier and the negation
lists, because "no student" is a negative quantifier. The rule therefore
requires two **distinct token positions**, so a single word cannot satisfy
both halves and trigger on itself.

This is candidate generation, not formal semantics. No logical form is built.
"""

from __future__ import annotations

from itertools import combinations

from ambisense.ambiguity.base import Detector, clause_root_of
from ambisense.schemas import (
    AmbiguityCandidate,
    AmbiguityType,
    LinguisticAnalysis,
    TokenInfo,
)

_DEFAULT_QUANTIFIERS = (
    "every", "all", "each", "some", "any", "no", "none", "many", "few",
    "most", "several",
)
_DEFAULT_NEGATIONS = ("not", "n't", "never", "no", "none", "nobody", "nothing")


class ScopeDetector(Detector):
    """Flags quantifier/negation and quantifier/quantifier interactions."""

    name = "scope"
    ambiguity_type = AmbiguityType.SCOPE

    def detect(self, analysis: LinguisticAnalysis) -> list[AmbiguityCandidate]:
        quantifiers = self._matching(
            analysis, self.lowered_set("quantifiers", _DEFAULT_QUANTIFIERS)
        )
        if not quantifiers:
            return []

        candidates: list[AmbiguityCandidate] = []
        candidates.extend(self._quantifier_negation(analysis, quantifiers))
        if bool(self.option("detect_quantifier_pairs", True)):
            candidates.extend(self._quantifier_pairs(analysis, quantifiers))
        return sorted(candidates, key=lambda c: c.char_start)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _matching(
        analysis: LinguisticAnalysis, vocabulary: frozenset[str]
    ) -> list[TokenInfo]:
        return [
            token for token in analysis.tokens
            if token.lemma in vocabulary or token.text.lower() in vocabulary
        ]

    @staticmethod
    def _same_clause(
        analysis: LinguisticAnalysis, first: TokenInfo, second: TokenInfo
    ) -> tuple[bool, str]:
        left = clause_root_of(analysis, first)
        right = clause_root_of(analysis, second)
        return left.index == right.index, left.text

    def _span(
        self, analysis: LinguisticAnalysis, first: TokenInfo, second: TokenInfo
    ) -> tuple[str, int, int]:
        start = min(first.char_start, second.char_start)
        end = max(first.char_end, second.char_end)
        return analysis.text[start:end], start, end

    # -- rule 1 ----------------------------------------------------------

    def _quantifier_negation(
        self, analysis: LinguisticAnalysis, quantifiers: list[TokenInfo]
    ) -> list[AmbiguityCandidate]:
        negations = self._matching(
            analysis, self.lowered_set("negations", _DEFAULT_NEGATIONS)
        )
        require_distinct = bool(self.option("require_distinct_tokens", True))
        found: list[AmbiguityCandidate] = []
        seen: set[tuple[int, int]] = set()

        for quantifier in quantifiers:
            for negation in negations:
                # "no" is in both lists; a single token must not satisfy both.
                if require_distinct and quantifier.index == negation.index:
                    continue
                pair = (quantifier.index, negation.index)
                if pair in seen:
                    continue

                same_clause, clause_head = self._same_clause(
                    analysis, quantifier, negation
                )
                if not same_clause:
                    continue

                seen.add(pair)
                span_text, start, end = self._span(analysis, quantifier, negation)
                found.append(
                    self.make_candidate(
                        span_text=span_text,
                        char_start=start,
                        char_end=end,
                        prior=0.8,
                        explanation=(
                            f"The quantifier '{quantifier.text}' and the "
                            f"negation '{negation.text}' occur in the same "
                            f"clause (headed by '{clause_head}'). Their "
                            f"relative scope is not fixed by the word order."
                        ),
                        evidence={
                            "rule": "quantifier_negation",
                            "quantifier": quantifier.text,
                            "quantifier_index": quantifier.index,
                            "negation": negation.text,
                            "negation_index": negation.index,
                            "clause_head": clause_head,
                            "sentence_index": quantifier.sentence_index,
                        },
                    )
                )
        return found

    # -- rule 2 ----------------------------------------------------------

    def _quantifier_pairs(
        self, analysis: LinguisticAnalysis, quantifiers: list[TokenInfo]
    ) -> list[AmbiguityCandidate]:
        found: list[AmbiguityCandidate] = []
        for first, second in combinations(quantifiers, 2):
            if first.index == second.index:
                continue
            same_clause, clause_head = self._same_clause(analysis, first, second)
            if not same_clause:
                continue

            span_text, start, end = self._span(analysis, first, second)
            found.append(
                self.make_candidate(
                    span_text=span_text,
                    char_start=start,
                    char_end=end,
                    prior=0.6,
                    explanation=(
                        f"Two quantifiers, '{first.text}' and '{second.text}', "
                        f"occur in the same clause. The order in which they "
                        f"take scope changes what the sentence asserts."
                    ),
                    evidence={
                        "rule": "quantifier_pair",
                        "first_quantifier": first.text,
                        "second_quantifier": second.text,
                        "clause_head": clause_head,
                        "sentence_index": first.sentence_index,
                    },
                )
            )
        return found
