"""Tests for the Phase 3 semantic analysis layer.

Two styles, as in the detector tests:

* **Integration tests** use the real spaCy vectors and the real WordNet
  corpus, and assert on *relative ranking* - never on exact similarity values,
  which would break on any model update.
* **Unit tests** inject a tiny deterministic fake embedding backend, so the
  ranking, threshold and failure-path logic can be exercised without spaCy.

No test makes a network call or requires an API key.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from ambisense.config import SemanticAnalysisConfig, load_settings
from ambisense.preprocessing import LinguisticAnalyzer
from ambisense.schemas import (
    LinguisticAnalysis,
    SemanticAnalysisStatus,
    SenseOption,
    SenseRanking,
)
from ambisense.semantic import (
    SemanticAnalyzer,
    SpacyEmbeddingBackend,
    cosine_similarity,
    count_available_senses,
    gloss_text,
    retrieve_senses,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer() -> LinguisticAnalyzer:
    return LinguisticAnalyzer("en_core_web_md")


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def backend(analyzer) -> SpacyEmbeddingBackend:
    return SpacyEmbeddingBackend(analyzer.nlp)


@pytest.fixture(scope="module")
def semantic(settings, backend) -> SemanticAnalyzer:
    return SemanticAnalyzer(settings.semantic_analysis, backend)


@pytest.fixture(scope="module")
def rank_word(analyzer, semantic):
    """Rank one word's senses inside a sentence, via the real pipeline."""
    def _rank(sentence: str, word: str, pos: str = "NOUN") -> SenseRanking:
        analysis = analyzer.analyze(sentence)
        token = next(
            t for t in analysis.tokens if t.text.lower() == word.lower()
        )
        return semantic.analyze_word(
            word=token.text, lemma=token.lemma, spacy_pos=token.pos,
            analysis=analysis, char_start=token.char_start,
            char_end=token.char_end,
        )
    return _rank


def rank_of(ranking: SenseRanking, sense_key: str) -> int | None:
    for sense in ranking.senses:
        if sense.sense_key == sense_key:
            return sense.rank
    return None


class FakeBackend:
    """Deterministic 3-dimensional backend: no spaCy, no WordNet, no surprises.

    Each known word maps to a fixed unit vector. Unknown words have no vector,
    which is how the missing-vector paths are exercised.
    """

    VECTORS = {
        "money": np.array([1.0, 0.0, 0.0]),
        "deposit": np.array([1.0, 0.0, 0.0]),
        "river": np.array([0.0, 1.0, 0.0]),
        "water": np.array([0.0, 1.0, 0.0]),
        "unrelated": np.array([0.0, 0.0, 1.0]),
    }

    @property
    def dimension(self) -> int:
        return 3

    def _mean(self, words):
        vectors = [self.VECTORS[w.lower()] for w in words
                   if w.lower() in self.VECTORS]
        return np.mean(vectors, axis=0) if vectors else None

    def vector_for_words(self, words):
        return self._mean(words)

    def vector_for_text(self, text):
        return self._mean(text.split())


class ZeroBackend(FakeBackend):
    """Returns a zero vector - a vector with no direction."""

    def vector_for_text(self, text):
        return np.zeros(3)


class GradedBackend(FakeBackend):
    """Gives every distinct gloss a distinct similarity.

    ``FakeBackend`` maps several real WordNet glosses onto the same vector
    (many bank glosses contain "money"), which produces a margin of exactly
    0.0 and makes threshold behaviour untestable. This backend derives a
    stable angle from a checksum of the gloss text, so similarities are
    distinct and strictly ordered while remaining fully deterministic.
    """

    def vector_for_words(self, words):
        return np.array([1.0, 0.0, 0.0])

    def vector_for_text(self, text):
        angle = (zlib.crc32(text.encode("utf-8")) % 1000) * (np.pi / 4000)
        return np.array([float(np.cos(angle)), float(np.sin(angle)), 0.0])


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vector = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a, b = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_magnitude_does_not_matter(self):
        a, b = np.array([1.0, 1.0]), np.array([5.0, 5.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "a,b",
        [
            (None, np.array([1.0, 0.0])),
            (np.array([1.0, 0.0]), None),
            (None, None),
            (np.zeros(3), np.array([1.0, 0.0, 0.0])),
            (np.array([1.0, 0.0, 0.0]), np.zeros(3)),
        ],
    )
    def test_missing_or_zero_vectors_return_none_not_zero(self, a, b):
        """A zero vector has no direction, so the angle is undefined.

        Returning 0.0 would be inventing a similarity score, which Phase 3
        must never do.
        """
        assert cosine_similarity(a, b) is None

    def test_result_stays_within_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            a, b = rng.normal(size=8), rng.normal(size=8)
            assert -1.0 <= cosine_similarity(a, b) <= 1.0


# ---------------------------------------------------------------------------
# WordNet sense retrieval
# ---------------------------------------------------------------------------


class TestSenseRetrieval:
    def test_retrieves_noun_senses(self):
        senses = retrieve_senses("bank", "NOUN", 8)
        assert senses
        assert all(sense.pos == "n" for sense in senses)
        assert all(sense.definition for sense in senses)

    def test_pos_restricts_the_inventory(self):
        """Verb senses of 'bank' must not pollute the noun analysis."""
        nouns = {s.sense_key for s in retrieve_senses("bank", "NOUN", 20)}
        verbs = {s.sense_key for s in retrieve_senses("bank", "VERB", 20)}
        assert nouns and verbs
        assert not (nouns & verbs)

    def test_cap_is_respected(self):
        assert len(retrieve_senses("bank", "NOUN", 3)) == 3

    def test_available_count_ignores_the_cap(self):
        assert count_available_senses("bank", "NOUN") > 3

    def test_unmappable_pos_returns_nothing(self):
        assert retrieve_senses("the", "DET", 8) == ()

    def test_unknown_word_returns_nothing(self):
        assert retrieve_senses("zzzqqxnotaword", "NOUN", 8) == ()

    def test_senses_are_immutable_and_cacheable(self):
        first = retrieve_senses("bank", "NOUN", 5)
        second = retrieve_senses("bank", "NOUN", 5)
        assert isinstance(first, tuple)
        assert first is second  # served from the lru_cache


class TestGlossText:
    @pytest.fixture
    def sense(self):
        return SenseOption(
            sense_key="x.n.01", definition="a financial institution",
            pos="n", examples=["he cashed a cheque"],
            lemma_names=["savings bank"],
        )

    def test_definition_only(self, sense):
        text = gloss_text(sense, include_examples=False,
                          include_lemma_names=False)
        assert text == "a financial institution"

    def test_examples_are_appended(self, sense):
        text = gloss_text(sense, include_examples=True,
                          include_lemma_names=False)
        assert "cashed a cheque" in text
        assert "savings bank" not in text

    def test_lemma_names_are_appended_when_requested(self, sense):
        text = gloss_text(sense, include_examples=False,
                          include_lemma_names=True)
        assert "savings bank" in text

    def test_empty_definition_does_not_produce_stray_whitespace(self):
        empty = SenseOption(sense_key="x.n.01", definition="", pos="n")
        assert gloss_text(empty) == ""


# ---------------------------------------------------------------------------
# Ranking with real vectors (relative assertions only)
# ---------------------------------------------------------------------------


class TestRealVectorRanking:
    def test_financial_context_favours_the_financial_sense(self, rank_word):
        ranking = rank_word("I deposited money at the bank.", "bank")
        assert ranking.status is SemanticAnalysisStatus.RANKED
        assert ranking.top_sense_key == "depository_financial_institution.n.01"

    def test_river_context_favours_the_riverbank_sense(self, rank_word):
        ranking = rank_word("The fisherman sat on the bank of the river.", "bank")
        assert ranking.top_sense_key == "bank.n.01"
        assert "sloping land" in ranking.senses[0].definition

    def test_context_changes_the_relative_ranking(self, rank_word):
        """The same word, two contexts, two different orderings."""
        financial = rank_word("I deposited money at the bank.", "bank")
        river = rank_word("The fisherman sat on the bank of the river.", "bank")
        assert financial.top_sense_key != river.top_sense_key

        riverbank_in_financial = rank_of(financial, "bank.n.01")
        riverbank_in_river = rank_of(river, "bank.n.01")
        assert riverbank_in_river < riverbank_in_financial

    def test_bird_context_favours_the_bird_sense_of_crane(self, rank_word):
        ranking = rank_word("The crane flew across the lake.", "crane")
        assert ranking.top_sense_key == "crane.n.05"
        assert "bird" in ranking.senses[0].definition

    def test_machine_context_raises_the_machine_sense_similarity(self, rank_word):
        """A documented partial failure, pinned by a test.

        The machine sense of "crane" does NOT reach rank 1 for
        "The crane lifted the heavy container." - the bird sense still wins,
        because averaged static vectors favour the longer, more generic bird
        gloss. What the context *does* change is the machine sense's absolute
        similarity, which rises substantially.

        This asserts the true, weaker claim. Asserting the machine sense ranks
        first would be asserting something the implementation does not do.
        See the README limitations section.
        """
        bird_ctx = rank_word("The crane flew across the lake.", "crane")
        machine_ctx = rank_word("The crane lifted the heavy container.", "crane")

        def similarity(ranking, key):
            return next(s.context_similarity for s in ranking.senses
                        if s.sense_key == key)

        assert similarity(machine_ctx, "crane.n.04") > \
               similarity(bird_ctx, "crane.n.04")

    def test_ranks_are_sequential_and_sorted(self, rank_word):
        ranking = rank_word("I deposited money at the bank.", "bank")
        assert [s.rank for s in ranking.senses] == \
               list(range(1, len(ranking.senses) + 1))
        similarities = [s.context_similarity for s in ranking.senses]
        assert similarities == sorted(similarities, reverse=True)

    def test_margin_matches_the_top_two(self, rank_word):
        ranking = rank_word("The fisherman sat on the bank of the river.", "bank")
        expected = (ranking.senses[0].context_similarity
                    - ranking.senses[1].context_similarity)
        assert ranking.margin == pytest.approx(expected, abs=1e-3)

    def test_target_word_is_excluded_from_its_own_context(self, rank_word):
        ranking = rank_word("I deposited money at the bank.", "bank")
        assert "bank" not in [w.lower() for w in ranking.context_words]

    def test_similarities_are_valid_cosines(self, rank_word):
        ranking = rank_word("I deposited money at the bank.", "bank")
        assert all(-1.0 <= s.context_similarity <= 1.0 for s in ranking.senses)

    def test_analysis_is_deterministic(self, rank_word):
        first = rank_word("I deposited money at the bank.", "bank")
        second = rank_word("I deposited money at the bank.", "bank")
        assert [(s.sense_key, s.rank, s.context_similarity) for s in first.senses] \
               == [(s.sense_key, s.rank, s.context_similarity) for s in second.senses]
        assert first.margin == second.margin

    def test_result_is_a_valid_pydantic_model(self, rank_word):
        ranking = rank_word("I deposited money at the bank.", "bank")
        assert isinstance(ranking, SenseRanking)
        restored = SenseRanking.model_validate_json(ranking.model_dump_json())
        assert restored.top_sense_key == ranking.top_sense_key


# ---------------------------------------------------------------------------
# Integration with Phase 2 candidates
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry(settings):
    from ambisense.ambiguity import DetectorRegistry
    return DetectorRegistry(settings.detectors)


class TestCandidateIntegration:
    def test_lexical_candidates_are_analysed(self, analyzer, registry, semantic):
        analysis = analyzer.analyze("I deposited money at the bank.")
        rankings = semantic.analyze_candidates(registry.run(analysis), analysis)
        assert any(r.word.lower() == "bank" for r in rankings)

    def test_non_lexical_candidates_are_skipped(
        self, analyzer, registry, semantic
    ):
        """WordNet sense ranking says nothing about PP attachment."""
        analysis = analyzer.analyze("I saw the man with the telescope.")
        candidates = registry.run(analysis)
        assert candidates, "expected the syntactic detector to fire"
        rankings = semantic.analyze_candidates(candidates, analysis)
        assert all("telescope" not in r.word.lower() for r in rankings)

    def test_user_context_widens_the_context_words(self, analyzer, semantic):
        analysis = analyzer.analyze("The crane is ready.")
        context = analyzer.analyze("The crane flew across the lake.")
        token = next(t for t in analysis.tokens if t.text == "crane")
        without = semantic.analyze_word(
            word="crane", lemma="crane", spacy_pos="NOUN", analysis=analysis,
            char_start=token.char_start,
        )
        with_context = semantic.analyze_word(
            word="crane", lemma="crane", spacy_pos="NOUN", analysis=analysis,
            context_analysis=context, char_start=token.char_start,
        )
        assert len(with_context.context_words) > len(without.context_words)
        assert "lake" in with_context.context_words

    def test_no_lexical_candidates_yields_no_rankings(
        self, analyzer, semantic
    ):
        analysis = analyzer.analyze("I saw the man with the telescope.")
        assert semantic.analyze_candidates([], analysis) == []


# ---------------------------------------------------------------------------
# Failure paths, using the fake backend
# ---------------------------------------------------------------------------


class TestFailurePaths:
    @pytest.fixture
    def analysis(self, analyzer):
        return analyzer.analyze("I deposited money at the bank.")

    def test_disabled_configuration_reports_disabled(self, analysis):
        analyzer_ = SemanticAnalyzer(
            SemanticAnalysisConfig(enable_context_sense_ranking=False),
            FakeBackend(),
        )
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN", analysis=analysis
        )
        assert ranking.status is SemanticAnalysisStatus.DISABLED
        assert ranking.senses == []

    def test_disabled_configuration_skips_candidates(self, analysis):
        analyzer_ = SemanticAnalyzer(
            SemanticAnalysisConfig(enable_context_sense_ranking=False),
            FakeBackend(),
        )
        assert analyzer_.analyze_candidates([], analysis) == []

    def test_unknown_word_reports_no_senses(self, analysis):
        analyzer_ = SemanticAnalyzer(SemanticAnalysisConfig(), FakeBackend())
        ranking = analyzer_.analyze_word(
            word="zzzqqx", lemma="zzzqqxnotaword", spacy_pos="NOUN",
            analysis=analysis,
        )
        assert ranking.status is SemanticAnalysisStatus.NO_SENSES
        assert ranking.senses == []
        assert "no senses" in ranking.note.lower()

    def test_missing_wordnet_corpus_reports_wordnet_unavailable(
        self, analysis, monkeypatch
    ):
        """An absent corpus must be distinguished from a word with no senses."""
        import ambisense.semantic.context_resolver as context_resolver

        monkeypatch.setattr(
            context_resolver, "retrieve_senses", lambda *a, **k: ()
        )
        monkeypatch.setattr(
            context_resolver, "count_available_senses", lambda *a, **k: 0
        )
        monkeypatch.setattr(
            context_resolver, "wordnet_available", lambda: False
        )
        analyzer_ = SemanticAnalyzer(SemanticAnalysisConfig(), FakeBackend())
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN", analysis=analysis,
        )
        assert ranking.status is SemanticAnalysisStatus.WORDNET_UNAVAILABLE
        assert "not installed" in ranking.note.lower()

    def test_missing_backend_reports_no_context_vector(self, analysis):
        analyzer_ = SemanticAnalyzer(SemanticAnalysisConfig(), backend=None)
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN", analysis=analysis
        )
        assert ranking.status is SemanticAnalysisStatus.NO_CONTEXT_VECTOR
        assert all(s.context_similarity is None for s in ranking.senses)

    def test_context_without_vectors_reports_no_context_vector(self, analyzer):
        """Every content word is unknown to the backend."""
        analysis = analyzer.analyze("Qqzz wibbly frobnicate the bank.")
        analyzer_ = SemanticAnalyzer(SemanticAnalysisConfig(), FakeBackend())
        token = next(t for t in analysis.tokens if t.text == "bank")
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN", analysis=analysis,
            char_start=token.char_start,
        )
        assert ranking.status is SemanticAnalysisStatus.NO_CONTEXT_VECTOR
        assert "no similarity was computed" in ranking.note.lower()

    def test_zero_gloss_vectors_report_no_gloss_vectors(self, analysis):
        """A zero vector must not become a 0.0 similarity."""
        analyzer_ = SemanticAnalyzer(SemanticAnalysisConfig(), ZeroBackend())
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN", analysis=analysis
        )
        assert ranking.status is SemanticAnalysisStatus.NO_GLOSS_VECTORS
        assert all(s.context_similarity is None for s in ranking.senses)

    def test_min_context_words_threshold_is_enforced(self, analysis):
        analyzer_ = SemanticAnalyzer(
            SemanticAnalysisConfig(min_context_words=99), FakeBackend()
        )
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN", analysis=analysis
        )
        assert ranking.status is SemanticAnalysisStatus.NO_CONTEXT_VECTOR

    def test_empty_analysis_does_not_crash(self):
        analyzer_ = SemanticAnalyzer(SemanticAnalysisConfig(), FakeBackend())
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN",
            analysis=LinguisticAnalysis(text=""),
        )
        assert ranking.status is SemanticAnalysisStatus.NO_CONTEXT_VECTOR

    def test_a_failure_still_returns_a_valid_model(self, analysis):
        analyzer_ = SemanticAnalyzer(SemanticAnalysisConfig(), backend=None)
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN", analysis=analysis
        )
        assert SenseRanking.model_validate_json(ranking.model_dump_json())


# ---------------------------------------------------------------------------
# Threshold behaviour, isolated from real vectors
# ---------------------------------------------------------------------------


class TestThresholds:
    @pytest.fixture
    def analysis(self, analyzer):
        return analyzer.analyze("I deposited money at the bank.")

    def test_large_margin_is_reported_as_favoured(self, analysis):
        analyzer_ = SemanticAnalyzer(
            SemanticAnalysisConfig(sense_resolution_margin=0.0),
            GradedBackend(),
        )
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN", analysis=analysis
        )
        assert ranking.status is SemanticAnalysisStatus.RANKED
        assert ranking.resolved_by_context is True

    def test_impossible_margin_is_never_resolved(self, analysis):
        analyzer_ = SemanticAnalyzer(
            SemanticAnalysisConfig(sense_resolution_margin=1.99),
            GradedBackend(),
        )
        ranking = analyzer_.analyze_word(
            word="bank", lemma="bank", spacy_pos="NOUN", analysis=analysis
        )
        assert ranking.resolved_by_context is False
        assert "below the configured margin" in ranking.note

    def test_note_never_claims_certainty(self, rank_word):
        """Phase 3 reports evidence, not decisions."""
        for sentence in ["I deposited money at the bank.",
                         "The fisherman sat on the bank of the river."]:
            note = rank_word(sentence, "bank").note.lower()
            assert "definitely" not in note
            assert "means" not in note
            assert any(word in note for word in
                       ("similarity", "favours", "ranks", "leans"))
