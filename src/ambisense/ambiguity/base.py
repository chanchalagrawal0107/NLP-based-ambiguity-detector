"""The detector interface and the helpers every detector shares.

Design contract for this layer
------------------------------

A detector answers exactly one question:

    "Is there a *structural* reason to suspect more than one reading here?"

It does **not** answer "is this sentence genuinely ambiguous?" - that is the
LLM's job in Phase 4. Consequently detectors are tuned for **high recall**:
they over-flag on purpose, and the later adjudication layer is what removes
the false positives.

Every detector must:

* accept a :class:`~ambisense.schemas.LinguisticAnalysis` (never a spaCy
  ``Doc``),
* be deterministic - same input, same output, same order,
* make no network call and use no LLM,
* attach machine-readable ``evidence`` and a human-readable ``explanation``
  to every candidate it emits.

The ``prior`` score
-------------------

``AmbiguityCandidate.prior`` is **the strength of the linguistic signal**, not
a probability that the sentence is ambiguous. It is a deterministic number
derived from the rule that fired - for example, how many antecedent candidates
a pronoun has, or how many WordNet domains a word spans. It is used to rank and
cap candidates, and it is passed to the LLM as evidence. It must never be
presented to a user as "the chance this is ambiguous".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Sequence

from ambisense.logging_setup import get_logger
from ambisense.schemas import (
    AmbiguityCandidate,
    AmbiguityType,
    LinguisticAnalysis,
    TokenInfo,
)

logger = get_logger(__name__)


# Punctuation and determiners are never interesting as span edges.
_TRIM_DEPS = frozenset({"punct"})


class Detector(ABC):
    """Base class for all rule-based ambiguity detectors."""

    #: Stable identifier, used in config, logs, evidence and the CLI.
    name: str = "base"
    #: The ambiguity category this detector proposes.
    ambiguity_type: AmbiguityType = AmbiguityType.UNKNOWN

    def __init__(self, settings: Optional[dict[str, Any]] = None) -> None:
        """Args:
        settings: The detector's ``<name>_settings`` block from config.yaml.
            Missing keys fall back to the detector's own defaults, so a
            partially specified block is valid.
        """
        self.settings: dict[str, Any] = settings or {}

    # -- configuration helpers ------------------------------------------

    def option(self, key: str, default: Any) -> Any:
        """Read one configuration value, falling back to ``default``."""
        value = self.settings.get(key)
        return default if value is None else value

    def lowered_set(self, key: str, default: Iterable[str] = ()) -> frozenset[str]:
        """Read a configured word list as a lowercase frozenset."""
        values = self.option(key, list(default))
        if isinstance(values, str):
            values = [values]
        return frozenset(str(item).lower() for item in values)

    # -- the interface --------------------------------------------------

    @abstractmethod
    def detect(self, analysis: LinguisticAnalysis) -> list[AmbiguityCandidate]:
        """Return zero or more candidates, in deterministic order."""

    # -- shared construction helper --------------------------------------

    def make_candidate(
        self,
        *,
        span_text: str,
        char_start: int,
        char_end: int,
        prior: float,
        explanation: str,
        evidence: dict[str, Any],
    ) -> AmbiguityCandidate:
        """Build a candidate, stamping in this detector's identity."""
        return AmbiguityCandidate(
            span_text=span_text,
            char_start=char_start,
            char_end=char_end,
            type_hint=self.ambiguity_type,
            detector_name=self.name,
            prior=max(0.0, min(1.0, prior)),
            explanation=explanation,
            evidence={"detector": self.name, **evidence},
        )


# ---------------------------------------------------------------------------
# Shared span utilities
#
# These live here rather than in each detector so that span construction is
# identical everywhere: a span is always a contiguous character range of the
# original text, so the UI can highlight it without re-tokenising.
# ---------------------------------------------------------------------------


def span_of_tokens(
    tokens: Sequence[TokenInfo],
    analysis: LinguisticAnalysis,
) -> tuple[str, int, int]:
    """Return ``(text, start, end)`` covering ``tokens`` in source order.

    Trailing punctuation is trimmed so that a span reads as a phrase rather
    than ending mid-sentence.
    """
    if not tokens:
        return "", 0, 0
    ordered = sorted(tokens, key=lambda token: token.char_start)
    while len(ordered) > 1 and ordered[-1].dep in _TRIM_DEPS:
        ordered = ordered[:-1]
    start = ordered[0].char_start
    end = ordered[-1].char_end
    return analysis.text[start:end], start, end


def subtree_tokens(
    analysis: LinguisticAnalysis,
    root_index: int,
    *,
    max_depth: int = 12,
) -> list[TokenInfo]:
    """Collect the dependency subtree rooted at ``root_index``.

    ``max_depth`` is a cycle guard. Dependency parses are trees in theory, but
    a malformed or hand-constructed analysis (as used in tests) can contain a
    cycle, and an unguarded walk would hang.
    """
    root = analysis.token_by_index(root_index)
    if root is None:
        return []

    collected: dict[int, TokenInfo] = {root.index: root}
    frontier = [root.index]
    depth = 0
    while frontier and depth < max_depth:
        next_frontier: list[int] = []
        for index in frontier:
            for child in analysis.children_of(index):
                if child.index not in collected:
                    collected[child.index] = child
                    next_frontier.append(child.index)
        frontier = next_frontier
        depth += 1
    return sorted(collected.values(), key=lambda token: token.index)


def clause_root_of(
    analysis: LinguisticAnalysis,
    token: TokenInfo,
    *,
    max_depth: int = 12,
) -> TokenInfo:
    """Walk up to the verb heading this token's clause.

    Used by the scope detector, which must confirm that a quantifier and a
    negation sit in the *same* clause before reporting an interaction.
    """
    current = token
    for _ in range(max_depth):
        if current.pos in {"VERB", "AUX"} or current.head_index == current.index:
            return current
        parent = analysis.token_by_index(current.head_index)
        if parent is None:
            return current
        current = parent
    return current


def is_content_token(token: TokenInfo) -> bool:
    """Whether a token is a real word rather than punctuation or a symbol."""
    return token.is_alpha and token.pos not in {"PUNCT", "SPACE", "SYM", "X"}
