"""Context-based WordNet sense ranking (Phase 3).

Pipeline for one ambiguous word::

    AmbiguityCandidate ("bank")
        -> retrieve WordNet senses for lemma + POS
        -> build a gloss vector per sense   (definition [+ examples])
        -> build one context vector         (surrounding content words)
        -> cosine similarity, context vs each gloss
        -> sort, rank, report

What this layer does **not** do
-------------------------------
It never concludes that a word *means* something. The strongest statement it
makes is "this sense's gloss is the most similar of those retrieved". Deciding
what the writer meant needs judgement this layer does not have, and is the job
of the Phase 4 LLM, which will receive these rankings as evidence.

Relationship to the Phase 2 semantic *detector*
-----------------------------------------------
Different things, despite the shared word. ``ambiguity/semantic.py`` is a rule
that flags semantic-role ambiguity ("ready to eat"). This module is the
semantic *analysis* layer that ranks word senses by context. They are not
connected.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ambisense.config import SemanticAnalysisConfig
from ambisense.logging_setup import get_logger
from ambisense.semantic.embeddings import (
    EmbeddingBackend,
    cosine_similarity,
)
from ambisense.semantic.wordnet_senses import (
    count_available_senses,
    gloss_text,
    retrieve_senses,
)
from ambisense.schemas import (
    AmbiguityCandidate,
    AmbiguityType,
    LinguisticAnalysis,
    SemanticAnalysisStatus,
    SenseOption,
    SenseRanking,
)

logger = get_logger(__name__)


class SemanticAnalyzer:
    """Ranks a word's WordNet senses against its surrounding context."""

    def __init__(
        self,
        config: Optional[SemanticAnalysisConfig] = None,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        """Args:
        config: The ``semantic_analysis`` configuration block.
        backend: Vector source. Injected rather than constructed so tests can
            supply a deterministic fake and skip loading spaCy.
        """
        self.config = config or SemanticAnalysisConfig()
        self.backend = backend

    # -- public API ------------------------------------------------------

    def analyze_candidates(
        self,
        candidates: Sequence[AmbiguityCandidate],
        analysis: LinguisticAnalysis,
        context_analysis: Optional[LinguisticAnalysis] = None,
    ) -> list[SenseRanking]:
        """Rank senses for every candidate that WordNet can speak to.

        Only **lexical** candidates are analysed. WordNet sense ranking says
        nothing useful about a prepositional-phrase attachment, an unresolved
        pronoun or a quantifier's scope, so forcing it onto those candidates
        would produce evidence that looks meaningful and is not.
        """
        if not self.config.enable_context_sense_ranking:
            return []

        rankings: list[SenseRanking] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            if candidate.type_hint is not AmbiguityType.LEXICAL:
                continue
            lemma = str(candidate.evidence.get("lemma", "")).lower()
            pos = str(candidate.evidence.get("pos", ""))
            if not lemma or not pos or (lemma, pos) in seen:
                continue
            seen.add((lemma, pos))
            rankings.append(
                self.analyze_word(
                    word=candidate.span_text,
                    lemma=lemma,
                    spacy_pos=pos,
                    analysis=analysis,
                    context_analysis=context_analysis,
                    char_start=candidate.char_start,
                    char_end=candidate.char_end,
                )
            )
        return rankings

    def analyze_word(
        self,
        *,
        word: str,
        lemma: str,
        spacy_pos: str,
        analysis: LinguisticAnalysis,
        context_analysis: Optional[LinguisticAnalysis] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
    ) -> SenseRanking:
        """Rank one word's senses. Always returns a ranking, never raises."""
        base = SenseRanking(
            word=word, lemma=lemma, pos=spacy_pos,
            char_start=char_start, char_end=char_end,
        )

        if not self.config.enable_context_sense_ranking:
            base.status = SemanticAnalysisStatus.DISABLED
            base.note = "Context-based sense ranking is disabled in configuration."
            return base

        senses = retrieve_senses(
            lemma, spacy_pos, self.config.max_senses_considered
        )
        base.senses_available = count_available_senses(lemma, spacy_pos)
        if not senses:
            base.status = SemanticAnalysisStatus.NO_SENSES
            base.note = (
                f"WordNet holds no senses for '{lemma}' as a "
                f"{spacy_pos.lower()}."
            )
            return base

        context_words = self._context_words(
            analysis, lemma, char_start, context_analysis
        )
        base.context_words = context_words

        if len(context_words) < self.config.min_context_words:
            base.status = SemanticAnalysisStatus.NO_CONTEXT_VECTOR
            base.note = (
                "Too few content words around the target to build a context "
                "representation, so no similarity was computed."
            )
            base.senses = list(senses)
            return base

        if self.backend is None:
            base.status = SemanticAnalysisStatus.NO_CONTEXT_VECTOR
            base.note = "No embedding backend was supplied."
            base.senses = list(senses)
            return base

        context_vector = self.backend.vector_for_words(context_words)
        if context_vector is None:
            base.status = SemanticAnalysisStatus.NO_CONTEXT_VECTOR
            base.note = (
                "None of the surrounding content words has a vector in the "
                "loaded model, so no similarity was computed."
            )
            base.senses = list(senses)
            return base

        scored = self._score_senses(senses, context_vector)
        if not scored:
            base.status = SemanticAnalysisStatus.NO_GLOSS_VECTORS
            base.note = (
                "No retrieved sense had a usable gloss vector, so no "
                "similarity was computed."
            )
            base.senses = list(senses)
            return base

        return self._finalise(base, scored)

    # -- internals -------------------------------------------------------

    def _context_words(
        self,
        analysis: LinguisticAnalysis,
        lemma: str,
        char_start: Optional[int],
        context_analysis: Optional[LinguisticAnalysis],
    ) -> list[str]:
        """Content words representing the context the target word sits in.

        Taken from the existing :class:`LinguisticAnalysis` - the sentence is
        never re-parsed. Function words and punctuation are excluded because
        they carry no topical meaning, and the target word itself is excluded
        because it would contribute the same vector to every comparison and
        only dilute the differences between senses.
        """
        allowed_pos = {pos.upper() for pos in self.config.context_content_pos}
        sentence_index = self._sentence_of(analysis, char_start)

        words: list[str] = []
        for token in analysis.tokens:
            if sentence_index is not None and token.sentence_index != sentence_index:
                continue
            if not self._is_context_word(token, allowed_pos, lemma):
                continue
            words.append(token.text)

        if self.config.include_user_context and context_analysis is not None:
            for token in context_analysis.tokens:
                if self._is_context_word(token, allowed_pos, lemma):
                    words.append(token.text)
        return words

    def _is_context_word(self, token, allowed_pos: set[str], lemma: str) -> bool:
        if token.pos.upper() not in allowed_pos:
            return False
        if token.is_stop or not token.is_alpha:
            return False
        if self.config.exclude_target_word and token.lemma.lower() == lemma:
            return False
        return True

    @staticmethod
    def _sentence_of(
        analysis: LinguisticAnalysis, char_start: Optional[int]
    ) -> Optional[int]:
        """Which sentence the target sits in, so context stays local."""
        if char_start is None:
            return None
        for token in analysis.tokens:
            if token.char_start == char_start:
                return token.sentence_index
        return None

    def _score_senses(
        self, senses: Sequence[SenseOption], context_vector
    ) -> list[SenseOption]:
        """Attach a similarity to each sense whose gloss has a vector."""
        scored: list[SenseOption] = []
        for sense in senses:
            text = gloss_text(
                sense,
                include_examples=self.config.gloss_includes_examples,
                include_lemma_names=self.config.gloss_includes_lemma_names,
            )
            gloss_vector = self.backend.vector_for_text(text)
            similarity = cosine_similarity(context_vector, gloss_vector)
            if similarity is None:
                # No usable gloss vector: omit rather than score it 0.0.
                continue
            scored.append(sense.model_copy(update={"context_similarity": similarity}))
        return scored

    def _finalise(
        self, base: SenseRanking, scored: list[SenseOption]
    ) -> SenseRanking:
        """Sort, rank, apply the reporting threshold and describe the result."""
        # Sort by similarity, then by sense key so ties are still deterministic.
        scored.sort(key=lambda s: (-(s.context_similarity or 0.0), s.sense_key))
        for position, sense in enumerate(scored, start=1):
            sense.rank = position

        top = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None
        margin = (
            None if runner_up is None
            else round((top.context_similarity or 0.0)
                       - (runner_up.context_similarity or 0.0), 4)
        )
        resolved = margin is not None and margin > self.config.sense_resolution_margin

        visible = [
            sense for sense in scored
            if (sense.context_similarity or 0.0) >= self.config.min_sense_similarity
        ] or scored[:1]

        base.status = SemanticAnalysisStatus.RANKED
        base.senses = visible
        base.top_sense_key = top.sense_key
        base.margin = margin
        base.resolved_by_context = resolved
        base.note = self._describe(base, top, margin, resolved)
        return base

    def _describe(
        self,
        ranking: SenseRanking,
        top: SenseOption,
        margin: Optional[float],
        resolved: bool,
    ) -> str:
        """Phrase the outcome as evidence, never as a decision."""
        definition = top.definition[:70]
        if margin is None:
            return (
                f"Only one sense of '{ranking.word}' could be scored "
                f"('{definition}'), so there is nothing to compare it against."
            )
        if resolved:
            return (
                f"Among the senses retrieved, '{definition}' has the highest "
                f"contextual similarity, leading the next sense by "
                f"{margin:.3f}. The context favours this reading; it is not "
                f"proof of the writer's intent."
            )
        return (
            f"'{definition}' ranks highest, but only by {margin:.3f}, which is "
            f"below the configured margin of "
            f"{self.config.sense_resolution_margin}. The context leans towards "
            f"this sense without clearly selecting it."
        )
