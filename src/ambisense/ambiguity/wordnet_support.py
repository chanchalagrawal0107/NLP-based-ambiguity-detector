"""Minimal WordNet access for the lexical detector.

Scope note
----------
This module answers only "how many, and how varied, are this word's senses?".
Phase 3 will add ``semantic/wordnet_senses.py`` for glosses, examples and
embedding-based sense ranking; that module can reuse :func:`to_wordnet_pos`.

Why sense *count* is not enough
-------------------------------
WordNet is very fine-grained. Measured noun sense counts::

    bank 10   cat 8   window 8   mat 7   crane 5   chicken 4   telescope 1

A threshold on the count alone flags ordinary concrete nouns such as *cat* and
*mat*, so "The cat sat on the mat." - an unambiguous control sentence - would
produce three lexical candidates.

We therefore also count distinct WordNet **lexnames** (semantic domains such
as ``noun.animal`` or ``noun.artifact``). A word whose senses spread across
several unrelated domains is closer to true homonymy::

    bank      -> act, artifact, group, object, possession   (5 domains)
    telescope -> artifact                                   (1 domain)

This is a useful signal, not a solution: *cat* still spans 4 domains. The
detector combines both measures and the report is capped, with the residual
false positives left for the Phase 4 LLM to reject. That trade-off is
deliberate and is documented in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from ambisense.logging_setup import get_logger

logger = get_logger(__name__)


#: spaCy coarse POS -> WordNet POS tag.
_POS_MAP: dict[str, str] = {
    "NOUN": "n",
    "PROPN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
}


class WordNetUnavailableError(RuntimeError):
    """Raised when the WordNet corpus cannot be loaded."""


def to_wordnet_pos(spacy_pos: str) -> Optional[str]:
    """Map a spaCy coarse POS tag to a WordNet POS, or ``None``.

    Returning ``None`` for unmapped tags (DET, ADP, PRON...) is what keeps the
    detector from looking up function words at all.
    """
    return _POS_MAP.get(spacy_pos.upper())


@dataclass(frozen=True)
class SenseProfile:
    """A summary of one word's WordNet senses for a single part of speech."""

    lemma: str
    wordnet_pos: str
    sense_count: int = 0
    domains: tuple[str, ...] = field(default_factory=tuple)
    example_definitions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def domain_count(self) -> int:
        return len(self.domains)

    @property
    def is_known(self) -> bool:
        """Whether WordNet knows this lemma at all."""
        return self.sense_count > 0


@lru_cache(maxsize=1)
def _wordnet():
    """Import and warm up the WordNet corpus exactly once.

    Raises:
        WordNetUnavailableError: if NLTK or the corpus is not installed.
    """
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("test")  # forces the lazy corpus load
    except ImportError as exc:
        raise WordNetUnavailableError(
            "NLTK is not installed. Run: pip install -r requirements.txt"
        ) from exc
    except LookupError as exc:
        raise WordNetUnavailableError(
            "The WordNet corpus is not downloaded. Run:\n"
            '  python -c "import nltk; nltk.download(\'wordnet\'); '
            "nltk.download('omw-1.4')\""
        ) from exc
    return wordnet


def wordnet_available() -> bool:
    """Whether WordNet can be used, without raising if it cannot."""
    try:
        _wordnet()
    except WordNetUnavailableError:
        return False
    return True


@lru_cache(maxsize=4096)
def sense_profile(lemma: str, spacy_pos: str) -> SenseProfile:
    """Summarise a lemma's senses for the given spaCy POS.

    Cached: an ordinary sentence repeats lemmas, and Phase 7 evaluation will
    call this across a whole dataset.

    Returns an empty profile when the POS is not mappable or WordNet is
    unavailable, so callers never need a try/except.
    """
    wordnet_pos = to_wordnet_pos(spacy_pos)
    if wordnet_pos is None:
        return SenseProfile(lemma=lemma, wordnet_pos="")

    try:
        wordnet = _wordnet()
    except WordNetUnavailableError as exc:
        logger.warning("Lexical analysis degraded: %s", exc)
        return SenseProfile(lemma=lemma, wordnet_pos=wordnet_pos)

    synsets = wordnet.synsets(lemma, pos=wordnet_pos)
    if not synsets:
        return SenseProfile(lemma=lemma, wordnet_pos=wordnet_pos)

    domains = sorted({synset.lexname() for synset in synsets})
    definitions = tuple(synset.definition() for synset in synsets[:3])

    return SenseProfile(
        lemma=lemma,
        wordnet_pos=wordnet_pos,
        sense_count=len(synsets),
        domains=tuple(domains),
        example_definitions=definitions,
    )
