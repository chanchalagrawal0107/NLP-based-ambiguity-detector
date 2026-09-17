"""Vector representations and cosine similarity (Phase 3).

Method
------
A piece of text is represented by the **mean of its content-word vectors**.
This is the standard explainable baseline:

    vector("sloping land beside a body of water")
        = mean(vec(sloping), vec(land), vec(body), vec(water))

Two such vectors are compared with cosine similarity. Nothing is trained and
nothing is random, so the result is fully deterministic.

Why spaCy's ``en_core_web_md`` vectors
--------------------------------------
The model is already loaded for parsing, it ships 300-dimensional vectors, and
using it adds no new dependency and no download. A transformer would give
context-sensitive vectors, but at the cost of a large dependency and a method
that is much harder to explain in a viva. That trade-off is documented in the
README, along with the accuracy cost it carries.

On the spaCy isolation rule
---------------------------
Phase 1 established that spaCy ``Doc`` objects do not leave the preprocessing
layer. This module is the one deliberate, contained exception: computing a
vector for a *WordNet gloss* means tokenising text that was never part of the
user's input, so a tokeniser is unavoidable here. The exception is kept narrow
- no ``Doc`` is returned, only NumPy arrays and plain strings - so everything
downstream still sees plain data.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, Sequence

import numpy as np

from ambisense.logging_setup import get_logger

logger = get_logger(__name__)

#: Below this norm a vector is treated as absent rather than meaningful.
_ZERO_NORM_EPSILON = 1e-8


def cosine_similarity(
    first: Optional[np.ndarray],
    second: Optional[np.ndarray],
) -> Optional[float]:
    """Cosine of the angle between two vectors, or ``None``.

    Returns ``None`` - never 0.0 - when either vector is missing or has
    effectively zero length. A zero vector has no direction, so the angle is
    undefined; reporting 0.0 would be inventing a score, which Phase 3 must
    not do.
    """
    if first is None or second is None:
        return None
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < _ZERO_NORM_EPSILON or second_norm < _ZERO_NORM_EPSILON:
        return None
    similarity = float(np.dot(first, second) / (first_norm * second_norm))
    # Guard against floating-point drift just outside [-1, 1].
    return max(-1.0, min(1.0, similarity))


class EmbeddingBackend(Protocol):
    """Anything able to turn words or text into a mean vector.

    Declared as a Protocol so tests can substitute a tiny deterministic fake
    and exercise the ranking logic without loading spaCy at all.
    """

    @property
    def dimension(self) -> int: ...

    def vector_for_words(self, words: Sequence[str]) -> Optional[np.ndarray]: ...

    def vector_for_text(self, text: str) -> Optional[np.ndarray]: ...


class SpacyEmbeddingBackend:
    """Mean-of-word-vectors backend built on an existing spaCy pipeline."""

    def __init__(self, nlp: Any) -> None:
        """Args:
        nlp: A loaded spaCy pipeline, normally ``LinguisticAnalyzer.nlp``,
            so that no second model is loaded.
        """
        self._nlp = nlp
        self._gloss_cache: dict[str, Optional[np.ndarray]] = {}

    # -- introspection ---------------------------------------------------

    @property
    def dimension(self) -> int:
        return int(self._nlp.vocab.vectors.shape[1])

    @property
    def has_vectors(self) -> bool:
        return bool(self._nlp.vocab.vectors.shape[0])

    # -- vectors ---------------------------------------------------------

    def vector_for_words(
        self, words: Sequence[str]
    ) -> Optional[np.ndarray]:
        """Mean vector of the given words, skipping those without vectors.

        Used for the context, whose words come from the existing
        :class:`LinguisticAnalysis` - so the sentence is never re-parsed.
        """
        vectors = []
        for word in words:
            lexeme = self._nlp.vocab[word]
            if lexeme.has_vector:
                vectors.append(lexeme.vector)
        return self._mean(vectors)

    def vector_for_text(self, text: str) -> Optional[np.ndarray]:
        """Mean content-word vector for a piece of text, e.g. a gloss.

        Results are cached by exact text. A single analysis compares one
        context against several glosses, and a demo re-runs the same
        sentences repeatedly, so the same gloss is tokenised many times
        otherwise. The cache is a plain dict on the instance: bounded in
        practice by the number of distinct glosses seen, and discarded with
        the backend.
        """
        if not text or not text.strip():
            return None
        cached = self._gloss_cache.get(text)
        if cached is not None or text in self._gloss_cache:
            return cached

        doc = self._nlp.make_doc(text)
        vectors = [
            token.vector for token in doc
            if token.has_vector and not token.is_stop and not token.is_punct
        ]
        result = self._mean(vectors)
        self._gloss_cache[text] = result
        return result

    @staticmethod
    def _mean(vectors: Iterable[np.ndarray]) -> Optional[np.ndarray]:
        stacked = list(vectors)
        if not stacked:
            return None
        return np.mean(stacked, axis=0)
