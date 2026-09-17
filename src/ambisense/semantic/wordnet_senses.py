"""WordNet sense retrieval with full glosses (Phase 3).

Relationship to ``ambiguity/wordnet_support.py``
------------------------------------------------
That Phase 2 module answers "how many senses, and how varied?" - counts only,
used to decide whether a word is worth flagging. This module retrieves the
sense *content*: definitions, usage examples and lemma names, which are what
the semantic layer actually compares against the context.

Both share the POS mapping, imported from the Phase 2 module so there is one
definition of how spaCy tags map to WordNet tags.
"""

from __future__ import annotations

from functools import lru_cache

from ambisense.ambiguity.wordnet_support import (
    WordNetUnavailableError,
    _wordnet,
    to_wordnet_pos,
)
from ambisense.logging_setup import get_logger
from ambisense.schemas import SenseOption

logger = get_logger(__name__)


@lru_cache(maxsize=2048)
def retrieve_senses(
    lemma: str,
    spacy_pos: str,
    max_senses: int = 8,
) -> tuple[SenseOption, ...]:
    """Retrieve WordNet senses for ``lemma`` under ``spacy_pos``.

    Only senses matching the token's part of speech are returned: looking up
    every sense across all parts of speech would, for example, mix the verb
    "to bank" into the analysis of the noun "bank".

    WordNet orders synsets roughly by frequency, so truncating to
    ``max_senses`` keeps the common readings and drops the obscure tail.

    Returns an empty tuple when the POS is not mappable, the lemma is unknown,
    or WordNet is unavailable. Cached because a sentence and an evaluation run
    both repeat lemmas; the result is immutable, which keeps the cache safe.
    """
    wordnet_pos = to_wordnet_pos(spacy_pos)
    if wordnet_pos is None:
        return ()

    try:
        wordnet = _wordnet()
    except WordNetUnavailableError as exc:
        logger.warning("Semantic analysis degraded: %s", exc)
        return ()

    synsets = wordnet.synsets(lemma, pos=wordnet_pos)[:max_senses]
    return tuple(
        SenseOption(
            sense_key=synset.name(),
            definition=synset.definition() or "",
            pos=synset.pos(),
            examples=list(synset.examples()),
            lemma_names=[name.replace("_", " ") for name in synset.lemma_names()],
        )
        for synset in synsets
    )


@lru_cache(maxsize=2048)
def count_available_senses(lemma: str, spacy_pos: str) -> int:
    """How many senses WordNet holds, before any cap is applied."""
    wordnet_pos = to_wordnet_pos(spacy_pos)
    if wordnet_pos is None:
        return 0
    try:
        wordnet = _wordnet()
    except WordNetUnavailableError:
        return 0
    return len(wordnet.synsets(lemma, pos=wordnet_pos))


def gloss_text(
    sense: SenseOption,
    *,
    include_examples: bool = True,
    include_lemma_names: bool = False,
) -> str:
    """Build the text used to represent a sense's meaning.

    Measured on this project's own examples:

    * **definition only** - the financial sense of "bank" ranks 4th for
      "I deposited money at the bank."
    * **definition + examples** - it ranks 1st. WordNet's examples are short
      sentences of real usage, which is exactly the kind of context the
      comparison needs.
    * **+ lemma names** - it drops back to 3rd, because near-synonym names such
      as "savings bank" pull unrelated senses towards the context.

    Hence the defaults. Both flags remain configurable so the effect can be
    demonstrated rather than merely asserted.
    """
    parts = [sense.definition]
    if include_examples:
        parts.extend(sense.examples)
    if include_lemma_names:
        parts.extend(sense.lemma_names)
    return " ".join(part for part in parts if part).strip()
