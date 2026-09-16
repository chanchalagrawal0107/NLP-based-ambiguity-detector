"""Tests for input validation and the spaCy analysis layer."""

from __future__ import annotations

import pytest

from ambisense.config import NLPConfig
from ambisense.preprocessing import (
    InputValidationError,
    LinguisticAnalyzer,
    clean_and_validate,
    normalise_text,
)


@pytest.fixture(scope="module")
def nlp_config() -> NLPConfig:
    return NLPConfig()


@pytest.fixture(scope="module")
def analyzer() -> LinguisticAnalyzer:
    """One shared analyzer: loading the spaCy model is slow."""
    return LinguisticAnalyzer("en_core_web_md")


class TestNormalisation:
    def test_smart_quotes_are_flattened(self):
        assert normalise_text("“the crane’s”") == '"the crane\'s"'

    def test_whitespace_is_collapsed(self):
        assert normalise_text("  I   saw \t the man.  ") == "I saw the man."


class TestValidation:
    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_empty_input_is_rejected(self, bad, nlp_config):
        with pytest.raises(InputValidationError, match="No text"):
            clean_and_validate(bad, None, nlp_config)

    def test_too_short_input_is_rejected(self, nlp_config):
        with pytest.raises(InputValidationError, match="too short"):
            clean_and_validate("a", None, nlp_config)

    def test_non_english_input_is_rejected(self, nlp_config):
        with pytest.raises(InputValidationError, match="English"):
            clean_and_validate("यह एक वाक्य है जो हिंदी में लिखा गया है।",
                               None, nlp_config)

    def test_overlong_input_is_truncated_not_rejected(self, nlp_config):
        long_text = "The crane is ready. " * 300
        cleaned = clean_and_validate(long_text, None, nlp_config)
        assert cleaned.was_truncated is True
        assert len(cleaned.text) <= nlp_config.max_input_length
        assert cleaned.warnings

    def test_context_is_normalised_when_supplied(self, nlp_config):
        cleaned = clean_and_validate(
            "The crane is ready.", "  The crane flew   across the lake. ",
            nlp_config,
        )
        assert cleaned.context == "The crane flew across the lake."


class TestLinguisticAnalyzer:
    def test_model_has_vectors(self, analyzer):
        """The semantic layer depends on en_core_web_md shipping vectors."""
        assert analyzer.has_vectors is True

    def test_basic_annotation(self, analyzer):
        analysis = analyzer.analyze("I saw the man with the telescope.")
        saw = next(t for t in analysis.tokens if t.text == "saw")
        assert saw.pos == "VERB"
        assert saw.lemma == "see"
        assert saw.dep == "ROOT"

    def test_sentence_segmentation(self, analyzer):
        analysis = analyzer.analyze(
            "The crane is ready. It lifted the steel beam."
        )
        assert len(analysis.sentences) == 2
        assert analysis.sentences[1].text.startswith("It lifted")

    def test_offsets_map_back_to_the_original_string(self, analyzer):
        text = "I saw the man with the telescope."
        analysis = analyzer.analyze(text)
        for token in analysis.tokens:
            assert text[token.char_start:token.char_end] == token.text

    def test_animate_heuristic_flags_role_nouns(self, analyzer):
        """Needed so 'the manager' can be an antecedent for 'he'.

        Renamed from ``is_person`` in Phase 2: the underlying check also
        accepts ORG and NORP entities, so 'person' was too narrow a name.
        The detection logic is unchanged.
        """
        analysis = analyzer.analyze(
            "The manager told the developer that he needed to fix the bug."
        )
        animate = {
            chunk.root_lemma
            for chunk in analysis.noun_chunks
            if chunk.is_animate_candidate
        }
        assert {"manager", "developer"} <= animate

    def test_organisations_are_animate_candidates(self, analyzer):
        """Documents *why* the field is not called ``is_person``."""
        analysis = analyzer.analyze("Microsoft said it would release the patch.")
        animate = {
            chunk.root_text
            for chunk in analysis.noun_chunks
            if chunk.is_animate_candidate
        }
        assert "Microsoft" in animate

    def test_named_entities_are_extracted(self, analyzer):
        analysis = analyzer.analyze("John told David that he was late.")
        labels = {e.text for e in analysis.entities}
        assert "John" in labels and "David" in labels

    def test_children_of_walks_the_dependency_tree(self, analyzer):
        analysis = analyzer.analyze("I saw the man with the telescope.")
        with_token = next(t for t in analysis.tokens if t.text == "with")
        children = analysis.children_of(with_token.index)
        assert any(child.dep == "pobj" for child in children)
