"""The traditional NLP analysis layer, built on spaCy.

Responsibilities and why each is needed by a later layer:

============================  =================================================
Technique                     Consumed by
============================  =================================================
Tokenisation                  every detector (span boundaries, offsets)
Sentence segmentation         referential detector (antecedent window)
POS tagging                   lexical detector (sense lookup is POS-specific)
Lemmatisation                 WordNet lookup, stoplist matching
Dependency parsing            syntactic + scope detectors (attachment, clauses)
Named entity recognition      referential detector (antecedent candidates)
Noun-chunk extraction         referential detector (antecedent candidates)
Word vectors                  semantic layer (context-vs-gloss similarity)
============================  =================================================

This layer makes **no judgement about ambiguity**. It only describes the
sentence. Keeping description and judgement apart is what allows the detectors
to be unit-tested without spaCy mocks and the LLM to be given evidence rather
than conclusions.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ambisense.logging_setup import get_logger
from ambisense.schemas import (
    EntityInfo,
    LinguisticAnalysis,
    NounChunkInfo,
    SentenceInfo,
    TokenInfo,
)

logger = get_logger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the configured spaCy model is not installed."""


# Heuristic animacy lexicon. spaCy's NER only labels *named* people ("John"),
# so a role noun such as "the manager" needs this list to become a candidate
# antecedent for "he"/"she". Documented as a heuristic in the README.
ANIMATE_NOUNS: frozenset[str] = frozenset({
    "man", "woman", "boy", "girl", "child", "person", "people", "student",
    "teacher", "manager", "developer", "engineer", "doctor", "nurse", "friend",
    "brother", "sister", "father", "mother", "parent", "son", "daughter",
    "colleague", "employee", "employer", "worker", "driver", "officer",
    "player", "author", "writer", "artist", "scientist", "researcher",
    "professor", "lecturer", "client", "customer", "patient", "visitor",
    "guest", "neighbour", "neighbor", "leader", "member", "team", "staff",
})

_PLURAL_TAGS = frozenset({"NNS", "NNPS"})
_ANIMATE_ENT_LABELS = frozenset({"PERSON", "ORG", "NORP"})


@lru_cache(maxsize=4)
def load_spacy_model(model_name: str) -> Any:
    """Load and cache a spaCy pipeline.

    Cached because loading ``en_core_web_md`` costs roughly a second - far too
    slow to repeat on every Streamlit re-run.

    Raises:
        ModelLoadError: with installation instructions if the model is absent.
    """
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - dependency guaranteed
        raise ModelLoadError(
            "spaCy is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise ModelLoadError(
            f"spaCy model '{model_name}' is not installed.\n"
            f"Install it with:  python -m spacy download {model_name}"
        ) from exc

    logger.info("Loaded spaCy model '%s'", model_name)
    return nlp


class LinguisticAnalyzer:
    """Wraps a spaCy pipeline and converts its output into plain schemas.

    Converting ``Doc`` objects into Pydantic models immediately means the rest
    of the codebase never touches spaCy internals: detectors work on simple,
    serialisable data and their tests need no spaCy model at all.
    """

    def __init__(self, model_name: str = "en_core_web_md") -> None:
        self.model_name = model_name
        self._nlp = load_spacy_model(model_name)

    @property
    def nlp(self) -> Any:
        """The underlying spaCy pipeline (used by the embeddings backend)."""
        return self._nlp

    @property
    def has_vectors(self) -> bool:
        """Whether the loaded model ships word vectors (``md``/``lg`` do)."""
        return bool(self._nlp.vocab.vectors.shape[0])

    def analyze(self, text: str) -> LinguisticAnalysis:
        """Run the full spaCy pipeline and return a serialisable analysis."""
        doc = self._nlp(text)

        sentences: list[SentenceInfo] = []
        sentence_of_token: dict[int, int] = {}
        for sent_index, sent in enumerate(doc.sents):
            sentences.append(
                SentenceInfo(
                    index=sent_index,
                    text=sent.text,
                    char_start=sent.start_char,
                    char_end=sent.end_char,
                )
            )
            for token in sent:
                sentence_of_token[token.i] = sent_index

        tokens = [
            TokenInfo(
                index=token.i,
                text=token.text,
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                head_index=token.head.i,
                head_text=token.head.text,
                is_stop=token.is_stop,
                is_alpha=token.is_alpha,
                char_start=token.idx,
                char_end=token.idx + len(token.text),
                sentence_index=sentence_of_token.get(token.i, 0),
            )
            for token in doc
        ]

        entities = [
            EntityInfo(
                text=ent.text,
                label=ent.label_,
                char_start=ent.start_char,
                char_end=ent.end_char,
                sentence_index=sentence_of_token.get(ent.start, 0),
            )
            for ent in doc.ents
        ]

        noun_chunks = [
            NounChunkInfo(
                text=chunk.text,
                root_text=chunk.root.text,
                root_lemma=chunk.root.lemma_.lower(),
                root_pos=chunk.root.pos_,
                root_dep=chunk.root.dep_,
                char_start=chunk.start_char,
                char_end=chunk.end_char,
                sentence_index=sentence_of_token.get(chunk.root.i, 0),
                is_plural=self._is_plural(chunk.root),
                is_animate_candidate=self._is_animate_candidate(chunk.root),
            )
            for chunk in doc.noun_chunks
        ]

        logger.debug(
            "Analysed %d tokens / %d sentences / %d entities / %d noun chunks",
            len(tokens), len(sentences), len(entities), len(noun_chunks),
        )

        return LinguisticAnalysis(
            text=text,
            tokens=tokens,
            sentences=sentences,
            entities=entities,
            noun_chunks=noun_chunks,
        )

    @staticmethod
    def _is_plural(token: Any) -> bool:
        """Number agreement, used to filter pronoun antecedents."""
        if token.tag_ in _PLURAL_TAGS:
            return True
        return "Plur" in token.morph.get("Number")

    @staticmethod
    def _is_animate_candidate(token: Any) -> bool:
        """Animacy heuristic: NER label, or membership of ANIMATE_NOUNS."""
        if token.ent_type_ in _ANIMATE_ENT_LABELS:
            return True
        return token.lemma_.lower() in ANIMATE_NOUNS
