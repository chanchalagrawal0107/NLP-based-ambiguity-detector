"""Tests for the six rule-based detectors.

Two styles are used deliberately:

* **Integration tests** run real spaCy output through a detector. These prove
  the rules work against the parses the system actually sees.
* **Unit tests** build a ``LinguisticAnalysis`` by hand. These need no spaCy
  model and are what the Doc-isolation design decision buys us: they run in
  microseconds and pin down edge cases that are hard to elicit from real text.

No test makes a network call or requires an API key.
"""

from __future__ import annotations

import pytest

from ambisense.ambiguity import (
    DETECTOR_CLASSES,
    DetectorRegistry,
    LexicalDetector,
    PragmaticDetector,
    ReferentialDetector,
    ScopeDetector,
    SemanticDetector,
    SyntacticDetector,
    deduplicate,
)
from ambisense.ambiguity.base import (
    clause_root_of,
    is_content_token,
    span_of_tokens,
    subtree_tokens,
)
from ambisense.ambiguity.wordnet_support import (
    sense_profile,
    to_wordnet_pos,
    wordnet_available,
)
from ambisense.config import DetectorsConfig, load_settings
from ambisense.preprocessing import LinguisticAnalyzer
from ambisense.schemas import (
    AmbiguityCandidate,
    AmbiguityType,
    LinguisticAnalysis,
    NounChunkInfo,
    SentenceInfo,
    TokenInfo,
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
def registry(settings) -> DetectorRegistry:
    return DetectorRegistry(settings.detectors)


@pytest.fixture(scope="module")
def analyse(analyzer):
    """Convenience: text -> LinguisticAnalysis."""
    return analyzer.analyze


def types_of(candidates) -> set[AmbiguityType]:
    return {candidate.type_hint for candidate in candidates}


def spans_of(candidates) -> list[str]:
    return [candidate.span_text for candidate in candidates]


# ---------------------------------------------------------------------------
# Hand-built analyses (no spaCy needed)
# ---------------------------------------------------------------------------


def make_token(index, text, lemma, pos, tag, dep, head, start, **kwargs):
    return TokenInfo(
        index=index, text=text, lemma=lemma, pos=pos, tag=tag, dep=dep,
        head_index=head, head_text="", char_start=start,
        char_end=start + len(text), **kwargs,
    )


@pytest.fixture
def cyclic_analysis() -> LinguisticAnalysis:
    """Two tokens that are each other's head - a malformed parse.

    Real spaCy output is acyclic, but the traversal helpers must not hang on
    hand-constructed or corrupted data.
    """
    return LinguisticAnalysis(
        text="a b",
        tokens=[
            make_token(0, "a", "a", "NOUN", "NN", "dep", 1, 0),
            make_token(1, "b", "b", "NOUN", "NN", "dep", 0, 2),
        ],
        sentences=[SentenceInfo(index=0, text="a b", char_start=0, char_end=3)],
    )


# ---------------------------------------------------------------------------
# WordNet support
# ---------------------------------------------------------------------------


class TestWordNetSupport:
    def test_pos_mapping(self):
        assert to_wordnet_pos("NOUN") == "n"
        assert to_wordnet_pos("VERB") == "v"
        assert to_wordnet_pos("PROPN") == "n"
        assert to_wordnet_pos("DET") is None
        assert to_wordnet_pos("ADP") is None

    def test_wordnet_is_installed(self):
        assert wordnet_available() is True

    def test_polysemous_word_profile(self):
        profile = sense_profile("bank", "NOUN")
        assert profile.sense_count >= 4
        assert profile.domain_count >= 3
        assert profile.is_known

    def test_monosemous_word_profile(self):
        profile = sense_profile("telescope", "NOUN")
        assert profile.sense_count == 1
        assert profile.domain_count == 1

    def test_unmappable_pos_returns_empty_profile(self):
        profile = sense_profile("the", "DET")
        assert profile.sense_count == 0
        assert profile.is_known is False

    def test_unknown_lemma_returns_empty_profile(self):
        profile = sense_profile("zzzqqxnotaword", "NOUN")
        assert profile.is_known is False


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------


class TestBaseHelpers:
    def test_span_of_empty_token_list(self):
        analysis = LinguisticAnalysis(text="")
        assert span_of_tokens([], analysis) == ("", 0, 0)

    def test_subtree_terminates_on_a_cyclic_parse(self, cyclic_analysis):
        """Guards against an infinite loop on malformed linguistic data."""
        tokens = subtree_tokens(cyclic_analysis, 0)
        assert len(tokens) == 2

    def test_clause_root_terminates_on_a_cyclic_parse(self, cyclic_analysis):
        token = cyclic_analysis.tokens[0]
        assert clause_root_of(cyclic_analysis, token) is not None

    def test_subtree_of_missing_token_is_empty(self, cyclic_analysis):
        assert subtree_tokens(cyclic_analysis, 99) == []

    def test_is_content_token(self):
        word = make_token(0, "bank", "bank", "NOUN", "NN", "pobj", 0, 0)
        punct = make_token(1, ".", ".", "PUNCT", ".", "punct", 0, 4,
                           is_alpha=False)
        assert is_content_token(word) is True
        assert is_content_token(punct) is False


# ---------------------------------------------------------------------------
# Detector: lexical
# ---------------------------------------------------------------------------


class TestLexicalDetector:
    @pytest.fixture
    def detector(self, settings):
        return LexicalDetector(settings.detectors.settings_for("lexical"))

    def test_flags_bank(self, detector, analyse):
        """The canonical lexical-ambiguity example."""
        candidates = detector.detect(analyse("I went to the bank."))
        assert "bank" in spans_of(candidates)
        assert types_of(candidates) == {AmbiguityType.LEXICAL}

    def test_does_not_flag_monosemous_nouns(self, detector, analyse):
        candidates = detector.detect(analyse("I bought a telescope."))
        assert "telescope" not in spans_of(candidates)

    def test_does_not_flag_function_words(self, detector, analyse):
        candidates = detector.detect(analyse("I went to the bank."))
        for span in spans_of(candidates):
            assert span.lower() not in {"the", "to", "i"}

    def test_stoplisted_words_are_skipped(self, detector, analyse):
        """'go' has 30 WordNet senses but is stoplisted as practically clear."""
        candidates = detector.detect(analyse("I went to the bank."))
        assert "went" not in spans_of(candidates)

    def test_evidence_records_the_thresholds_used(self, detector, analyse):
        candidate = detector.detect(analyse("I went to the bank."))[0]
        assert candidate.evidence["rule"] == "wordnet_polysemy"
        assert candidate.evidence["sense_count"] >= 4
        assert "min_senses" in candidate.evidence["thresholds"]
        assert candidate.evidence["domains"]

    def test_repeated_lemma_yields_one_candidate(self, detector, analyse):
        text = "The bank near the other bank was closed."
        banks = [c for c in detector.detect(analyse(text)) if c.span_text == "bank"]
        assert len(banks) == 1

    def test_per_sentence_cap_is_applied(self, analyse):
        detector = LexicalDetector({"max_candidates_per_sentence": 1,
                                    "min_senses": 2, "min_domains": 1})
        candidates = detector.detect(analyse("The cat sat on the mat."))
        assert len(candidates) <= 1

    def test_prior_is_deterministic(self, detector, analyse):
        first = detector.detect(analyse("I went to the bank."))
        second = detector.detect(analyse("I went to the bank."))
        assert [c.prior for c in first] == [c.prior for c in second]

    def test_prior_is_within_bounds(self, detector, analyse):
        for candidate in detector.detect(analyse("Can you open the window?")):
            assert 0.0 <= candidate.prior <= 1.0


# ---------------------------------------------------------------------------
# Detector: syntactic
# ---------------------------------------------------------------------------


class TestSyntacticDetector:
    @pytest.fixture
    def detector(self, settings):
        return SyntacticDetector(settings.detectors.settings_for("syntactic"))

    def test_flags_pp_attachment(self, detector, analyse):
        candidates = detector.detect(analyse("I saw the man with the telescope."))
        assert AmbiguityType.SYNTACTIC in types_of(candidates)
        assert any("telescope" in span for span in spans_of(candidates))

    def test_pp_evidence_names_both_attachment_sites(self, detector, analyse):
        candidate = detector.detect(
            analyse("I saw the man with the telescope.")
        )[0]
        assert candidate.evidence["alternative_verb_site"] == "saw"
        assert candidate.evidence["alternative_noun_site"] == "man"
        assert candidate.evidence["rule"] == "pp_attachment"

    def test_does_not_flag_single_attachment_site(self, detector, analyse):
        """No noun sits between 'sat' and 'on', so there is no alternative.

        This is the control that stops the rule degenerating into
        'any preposition near a noun'.
        """
        candidates = detector.detect(analyse("The cat sat on the mat."))
        pp = [c for c in candidates if c.evidence.get("rule") == "pp_attachment"]
        assert pp == []

    def test_flags_coordination_scope(self, detector, analyse):
        candidates = detector.detect(analyse("The old men and women sat outside."))
        coordination = [
            c for c in candidates
            if c.evidence.get("rule") == "coordination_scope"
        ]
        assert coordination
        assert coordination[0].evidence["modifier"] == "old"
        assert coordination[0].evidence["second_conjunct"] == "women"

    def test_coordination_without_modifier_is_not_flagged(self, detector, analyse):
        candidates = detector.detect(analyse("The men and women sat outside."))
        assert not [
            c for c in candidates
            if c.evidence.get("rule") == "coordination_scope"
        ]

    def test_rules_can_be_disabled_individually(self, analyse):
        detector = SyntacticDetector({"detect_pp_attachment": False,
                                      "detect_coordination_scope": False})
        assert detector.detect(analyse("I saw the man with the telescope.")) == []


# ---------------------------------------------------------------------------
# Detector: referential
# ---------------------------------------------------------------------------


class TestReferentialDetector:
    @pytest.fixture
    def detector(self, settings):
        return ReferentialDetector(
            settings.detectors.settings_for("referential")
        )

    def test_flags_ambiguous_pronoun(self, detector, analyse):
        candidates = detector.detect(analyse("John told David that he was late."))
        assert AmbiguityType.REFERENTIAL in types_of(candidates)
        assert "he" in spans_of(candidates)

    def test_evidence_lists_both_antecedents(self, detector, analyse):
        candidate = detector.detect(
            analyse("John told David that he was late.")
        )[0]
        antecedents = candidate.evidence["antecedents"]
        assert candidate.evidence["antecedent_count"] == 2
        assert any("John" in a for a in antecedents)
        assert any("David" in a for a in antecedents)

    def test_flags_role_nouns_not_just_names(self, detector, analyse):
        candidates = detector.detect(analyse(
            "The manager told the developer that he needed to fix the bug."
        ))
        assert "he" in spans_of(candidates)

    def test_single_antecedent_is_not_flagged(self, detector, analyse):
        candidates = detector.detect(analyse("John said he was late."))
        assert candidates == []

    def test_animacy_filter_excludes_objects(self, detector, analyse):
        """'he' must not take 'the report' as an antecedent."""
        candidates = detector.detect(
            analyse("John read the report and he left.")
        )
        for candidate in candidates:
            assert "report" not in " ".join(candidate.evidence["antecedents"])

    def test_number_filter_excludes_plurals(self, detector, analyse):
        candidates = detector.detect(
            analyse("The students met the teacher and he smiled.")
        )
        for candidate in candidates:
            assert "students" not in " ".join(candidate.evidence["antecedents"])

    def test_demonstrative_that_is_not_treated_as_a_pronoun(
        self, detector, analyse
    ):
        """spaCy tags 'that' as SCONJ here; including it caused false hits."""
        candidates = detector.detect(analyse("John told David that he was late."))
        assert "that" not in spans_of(candidates)

    def test_prior_rises_with_more_antecedents(self, detector, analyse):
        two = detector.detect(analyse("John told David that he was late."))[0]
        three = detector.detect(
            analyse("John told David and Peter that he was late.")
        )
        if three:
            assert three[0].prior >= two.prior


# ---------------------------------------------------------------------------
# Detector: semantic
# ---------------------------------------------------------------------------


class TestSemanticDetector:
    @pytest.fixture
    def detector(self, settings):
        return SemanticDetector(settings.detectors.settings_for("semantic"))

    def test_flags_tough_construction(self, detector, analyse):
        candidates = detector.detect(analyse("The chicken is ready to eat."))
        tough = [
            c for c in candidates
            if c.evidence.get("rule") == "tough_construction"
        ]
        assert tough
        assert tough[0].type_hint is AmbiguityType.SEMANTIC
        assert "ready to eat" in tough[0].span_text

    def test_evidence_explains_the_construction(self, detector, analyse):
        candidate = [
            c for c in detector.detect(analyse("The chicken is ready to eat."))
            if c.evidence.get("rule") == "tough_construction"
        ][0]
        assert candidate.evidence["adjective"] == "ready"
        assert candidate.evidence["infinitive_lemma"] == "eat"
        assert candidate.evidence["infinitive_has_object"] is False
        assert "alternative semantic role" in candidate.evidence["reason"]

    def test_explicit_object_removes_the_ambiguity(self, detector, analyse):
        """'ready to eat the corn' fixes the roles, so it must not fire."""
        candidates = detector.detect(
            analyse("The chicken is ready to eat the corn.")
        )
        assert not [
            c for c in candidates
            if c.evidence.get("rule") == "tough_construction"
        ]

    def test_intransitive_verb_is_excluded(self, detector, analyse):
        """'ready to go' has no passive reading."""
        candidates = detector.detect(analyse("The team is ready to go."))
        assert not [
            c for c in candidates
            if c.evidence.get("rule") == "tough_construction"
        ]

    def test_non_tough_adjective_is_ignored(self, detector, analyse):
        candidates = detector.detect(analyse("The chicken is reluctant to eat."))
        assert not [
            c for c in candidates
            if c.evidence.get("rule") == "tough_construction"
        ]

    def test_noun_compound_rule_can_be_disabled(self, analyse):
        detector = SemanticDetector({"detect_noun_compounds": False,
                                     "detect_tough_constructions": True})
        candidates = detector.detect(analyse("The student protest continued."))
        assert not [
            c for c in candidates if c.evidence.get("rule") == "noun_compound"
        ]


# ---------------------------------------------------------------------------
# Detector: scope
# ---------------------------------------------------------------------------


class TestScopeDetector:
    @pytest.fixture
    def detector(self, settings):
        return ScopeDetector(settings.detectors.settings_for("scope"))

    def test_flags_quantifier_negation(self, detector, analyse):
        candidates = detector.detect(
            analyse("Every student didn't submit the assignment.")
        )
        assert AmbiguityType.SCOPE in types_of(candidates)
        quantifier_negation = [
            c for c in candidates
            if c.evidence.get("rule") == "quantifier_negation"
        ]
        assert quantifier_negation
        assert quantifier_negation[0].evidence["quantifier"].lower() == "every"

    def test_plain_negation_is_not_flagged(self, detector, analyse):
        candidates = detector.detect(analyse("The student didn't submit it."))
        assert not [
            c for c in candidates
            if c.evidence.get("rule") == "quantifier_negation"
        ]

    def test_plain_quantifier_is_not_flagged(self, detector, analyse):
        candidates = detector.detect(analyse("Every student submitted the work."))
        assert not [
            c for c in candidates
            if c.evidence.get("rule") == "quantifier_negation"
        ]

    def test_single_word_cannot_satisfy_both_roles(self, analyse):
        """'no' is in both word lists; it must not trigger on itself."""
        detector = ScopeDetector({
            "quantifiers": ["no"], "negations": ["no"],
            "require_distinct_tokens": True, "detect_quantifier_pairs": False,
        })
        assert detector.detect(analyse("No student arrived.")) == []

    def test_cross_clause_pairs_are_not_flagged(self, detector, analyse):
        """Different clauses cannot interact in scope."""
        candidates = detector.detect(
            analyse("Every student passed, but I didn't attend.")
        )
        assert not [
            c for c in candidates
            if c.evidence.get("rule") == "quantifier_negation"
        ]

    def test_evidence_records_the_clause_head(self, detector, analyse):
        candidate = [
            c for c in detector.detect(
                analyse("Every student didn't submit the assignment.")
            )
            if c.evidence.get("rule") == "quantifier_negation"
        ][0]
        assert candidate.evidence["clause_head"] == "submit"


# ---------------------------------------------------------------------------
# Detector: pragmatic
# ---------------------------------------------------------------------------


class TestPragmaticDetector:
    @pytest.fixture
    def detector(self, settings):
        return PragmaticDetector(settings.detectors.settings_for("pragmatic"))

    def test_flags_modal_request(self, detector, analyse):
        candidates = detector.detect(analyse("Can you open the window?"))
        assert AmbiguityType.PRAGMATIC in types_of(candidates)
        assert candidates[0].evidence["rule"] == "modal_indirect_request"

    def test_evidence_states_both_forces_without_choosing(self, detector, analyse):
        candidate = detector.detect(analyse("Can you open the window?"))[0]
        assert candidate.evidence["literal_force"] == "question about ability"
        assert "request" in candidate.evidence["possible_indirect_force"]
        assert candidate.evidence["action_verb"] == "open"

    def test_statement_is_not_flagged(self, detector, analyse):
        assert detector.detect(analyse("You can open the window.")) == []

    def test_third_person_question_is_not_flagged(self, detector, analyse):
        assert detector.detect(analyse("Can he open the window?")) == []

    def test_stative_verb_is_not_flagged(self, detector, analyse):
        """'Can you see it?' asks about perception, not a performable act."""
        assert detector.detect(analyse("Can you see the window?")) == []

    def test_question_mark_requirement_is_configurable(self, analyse):
        detector = PragmaticDetector({"require_question_mark": False})
        assert detector.detect(analyse("Can you open the window")) != []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_six_detectors_are_registered(self):
        assert set(DETECTOR_CLASSES) == set(DetectorsConfig.DETECTOR_NAMES)

    def test_builds_every_enabled_detector(self, registry):
        assert len(registry.detectors) == 6

    def test_disabled_detector_is_not_built(self):
        registry = DetectorRegistry(DetectorsConfig(lexical=False))
        assert "lexical" not in registry.detector_names
        assert len(registry.detectors) == 5

    def test_disabled_detector_produces_no_candidates(self, analyse):
        registry = DetectorRegistry(DetectorsConfig(
            lexical=False, syntactic=True, referential=False,
            semantic=False, scope=False, pragmatic=False,
        ))
        candidates = registry.run(analyse("I went to the bank."))
        assert AmbiguityType.LEXICAL not in types_of(candidates)

    def test_all_detectors_disabled_yields_nothing(self, analyse):
        registry = DetectorRegistry(DetectorsConfig(
            lexical=False, syntactic=False, referential=False,
            semantic=False, scope=False, pragmatic=False,
        ))
        assert registry.run(analyse("I saw the man with the telescope.")) == []

    def test_multiple_detectors_can_fire_on_one_sentence(self, registry, analyse):
        candidates = registry.run(analyse("The chicken is ready to eat."))
        assert len(types_of(candidates)) >= 2

    def test_results_are_sorted_by_position(self, registry, analyse):
        candidates = registry.run(
            analyse("Every student didn't submit the assignment.")
        )
        starts = [c.char_start for c in candidates]
        assert starts == sorted(starts)

    def test_output_is_deterministic_across_runs(self, registry, analyse):
        analysis = analyse("The chicken is ready to eat.")
        first = registry.run(analysis)
        second = registry.run(analysis)
        assert [(c.detector_name, c.char_start, c.prior) for c in first] == \
               [(c.detector_name, c.char_start, c.prior) for c in second]

    def test_a_failing_detector_does_not_abort_the_run(self, analyse, monkeypatch):
        """One broken rule must not cost the user every other finding."""
        def explode(self, analysis):
            raise RuntimeError("simulated detector failure")

        monkeypatch.setattr(LexicalDetector, "detect", explode)
        registry = DetectorRegistry(DetectorsConfig())
        candidates = registry.run(analyse("I saw the man with the telescope."))
        assert AmbiguityType.SYNTACTIC in types_of(candidates)

    def test_empty_analysis_is_handled(self, registry):
        assert registry.run(LinguisticAnalysis(text="")) == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _candidate(start, end, type_hint, detector, rule, prior=0.5):
    return AmbiguityCandidate(
        span_text="x", char_start=start, char_end=end, type_hint=type_hint,
        detector_name=detector, prior=prior, evidence={"rule": rule},
    )


class TestDeduplication:
    def test_identical_candidates_are_merged(self):
        pair = [
            _candidate(0, 4, AmbiguityType.LEXICAL, "lexical", "r1", 0.5),
            _candidate(0, 4, AmbiguityType.LEXICAL, "lexical", "r1", 0.9),
        ]
        merged = deduplicate(pair)
        assert len(merged) == 1
        assert merged[0].prior == 0.9

    def test_same_span_different_type_is_kept_separate(self):
        """A span can be both a lexical and a semantic candidate."""
        pair = [
            _candidate(0, 4, AmbiguityType.LEXICAL, "lexical", "r1"),
            _candidate(0, 4, AmbiguityType.SEMANTIC, "semantic", "r2"),
        ]
        assert len(deduplicate(pair)) == 2

    def test_same_span_same_type_different_rule_is_kept(self):
        pair = [
            _candidate(0, 4, AmbiguityType.SYNTACTIC, "syntactic", "pp"),
            _candidate(0, 4, AmbiguityType.SYNTACTIC, "syntactic", "coord"),
        ]
        assert len(deduplicate(pair)) == 2

    def test_different_spans_are_kept(self):
        pair = [
            _candidate(0, 4, AmbiguityType.LEXICAL, "lexical", "r1"),
            _candidate(5, 9, AmbiguityType.LEXICAL, "lexical", "r1"),
        ]
        assert len(deduplicate(pair)) == 2

    def test_agreement_between_detectors_is_recorded(self):
        pair = [
            _candidate(0, 4, AmbiguityType.LEXICAL, "lexical", "r1", 0.9),
            _candidate(0, 4, AmbiguityType.LEXICAL, "semantic", "r1", 0.4),
        ]
        merged = deduplicate(pair)
        assert merged[0].evidence["also_detected_by"] == ["semantic"]

    def test_empty_input(self):
        assert deduplicate([]) == []


# ---------------------------------------------------------------------------
# End-to-end behaviour and edge cases
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("I went to the bank.", AmbiguityType.LEXICAL),
            ("I saw the man with the telescope.", AmbiguityType.SYNTACTIC),
            ("John told David that he was late.", AmbiguityType.REFERENTIAL),
            ("The chicken is ready to eat.", AmbiguityType.SEMANTIC),
            ("Every student didn't submit the assignment.", AmbiguityType.SCOPE),
            ("Can you open the window?", AmbiguityType.PRAGMATIC),
            ("The old men and women sat outside.", AmbiguityType.SYNTACTIC),
        ],
    )
    def test_required_examples_produce_the_expected_type(
        self, registry, analyse, text, expected
    ):
        assert expected in types_of(registry.run(analyse(text)))

    @pytest.mark.parametrize(
        "text",
        [
            "The cat sat on the mat.",
            "The train departed at noon.",
            "Water boils at one hundred degrees Celsius.",
        ],
    )
    def test_unambiguous_sentences_avoid_structural_detectors(
        self, registry, analyse, text
    ):
        """Structural detectors must stay silent on clear sentences.

        Each control sentence has no noun between its verb and its
        preposition, so no alternative attachment site exists.

        The lexical detector is excluded from this assertion: WordNet's
        fine-grained senses mean it still fires on ordinary nouns, which is a
        known and documented property of the high-recall design.
        """
        structural = {
            AmbiguityType.SYNTACTIC, AmbiguityType.REFERENTIAL,
            AmbiguityType.SCOPE, AmbiguityType.PRAGMATIC,
        }
        assert not (types_of(registry.run(analyse(text))) & structural)

    def test_known_false_positive_verb_object_pp(self, registry, analyse):
        """Documents a deliberate false positive rather than hiding it.

        "She bought three apples at the market." has the same V-NP-PP shape as
        the telescope sentence, so the rule fires: "[the apples at the market]"
        is a structurally available noun phrase. A human reads the PP as
        modifying the buying, but nothing *structural* rules out the other
        reading, and this layer only reports structure.

        Rejecting pragmatically implausible readings is the Phase 4 LLM's job.
        If this test ever starts failing, the syntactic rule has been made
        narrower - check that the telescope sentence still fires.
        """
        candidates = registry.run(analyse("She bought three apples at the market."))
        pp = [c for c in candidates if c.evidence.get("rule") == "pp_attachment"]
        assert pp, "expected the documented false positive to still occur"
        assert pp[0].evidence["alternative_verb_site"] == "bought"

    def test_long_input_is_handled(self, registry, analyse):
        text = " ".join(["I saw the man with the telescope."] * 25)
        candidates = registry.run(analyse(text))
        assert candidates
        assert all(0.0 <= c.prior <= 1.0 for c in candidates)

    def test_every_candidate_has_evidence_and_an_explanation(
        self, registry, analyse
    ):
        for text in ["I went to the bank.", "Can you open the window?",
                     "The chicken is ready to eat."]:
            for candidate in registry.run(analyse(text)):
                assert candidate.explanation.strip()
                assert candidate.evidence.get("rule")
                assert candidate.evidence.get("detector") == candidate.detector_name

    def test_spans_map_back_to_the_source_text(self, registry, analyse):
        """Offsets must be usable for highlighting without re-tokenising."""
        text = "I saw the man with the telescope."
        for candidate in registry.run(analyse(text)):
            assert text[candidate.char_start:candidate.char_end] == \
                   candidate.span_text

    def test_candidates_serialise_to_json(self, registry, analyse):
        """Evidence must survive serialisation for the Phase 4 prompt."""
        for candidate in registry.run(analyse("The chicken is ready to eat.")):
            assert candidate.model_dump_json()
