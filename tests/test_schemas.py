"""Tests for the shared schemas.

These focus on the *defensive* behaviour: the validators exist because small
LLMs return near-miss types, percentage confidences and single strings where a
list is required. If these tests pass, a sloppy model response still produces
a valid report instead of a crash.
"""

from __future__ import annotations

import pytest

from ambisense.schemas import (
    AmbiguityFinding,
    AmbiguityType,
    ClarityVerdict,
    LLMAnalysis,
    clamp_confidence,
)


class TestAmbiguityTypeCoercion:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("lexical", AmbiguityType.LEXICAL),
            ("SYNTACTIC", AmbiguityType.SYNTACTIC),
            ("structural", AmbiguityType.SYNTACTIC),
            ("attachment", AmbiguityType.SYNTACTIC),
            ("word-sense", AmbiguityType.LEXICAL),
            ("anaphora", AmbiguityType.REFERENTIAL),
            ("quantifier scope", AmbiguityType.SCOPE),
            ("speech act", AmbiguityType.PRAGMATIC),
        ],
    )
    def test_known_and_aliased_types(self, raw, expected):
        assert AmbiguityType.coerce(raw) is expected

    @pytest.mark.parametrize("raw", ["", None, "completely-made-up", 42])
    def test_unrecognised_types_become_unknown(self, raw):
        """The system must be able to say 'I do not know' (brief section 1)."""
        assert AmbiguityType.coerce(raw) is AmbiguityType.UNKNOWN


class TestClarityVerdictCoercion:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("vague", ClarityVerdict.VAGUE),
            ("unambiguous", ClarityVerdict.CLEAR),
            ("insufficient context", ClarityVerdict.INSUFFICIENT_CONTEXT),
            ("needs-context", ClarityVerdict.INSUFFICIENT_CONTEXT),
            ("subjective", ClarityVerdict.VAGUE),
        ],
    )
    def test_coercion(self, raw, expected):
        assert ClarityVerdict.coerce(raw) is expected


class TestConfidenceClamping:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0.75, 0.75),
            ("0.75", 0.75),
            (85, 0.85),        # model answered in percent
            (100, 1.0),        # percent, upper bound
            (1.4, 1.0),        # overshoot, NOT 1.4 percent
            (9.0, 1.0),        # still overshoot: below the percent floor
            (250, 1.0),        # nonsense, clamped
            (-0.2, 0.0),
            ("not a number", 0.5),
            (None, 0.5),
        ],
    )
    def test_clamping(self, raw, expected):
        assert clamp_confidence(raw) == pytest.approx(expected)


class TestAmbiguityFinding:
    def test_messy_llm_payload_still_validates(self):
        finding = AmbiguityFinding(
            text_span="with the telescope",
            type="attachment",
            confidence="92",
            rewrites="I used a telescope to see the man.",
        )
        assert finding.type is AmbiguityType.SYNTACTIC
        assert finding.confidence == pytest.approx(0.92)
        assert finding.rewrites == ["I used a telescope to see the man."]

    def test_blank_rewrites_are_dropped(self):
        finding = AmbiguityFinding(
            text_span="bank", rewrites=["  ", "", "A river bank."]
        )
        assert finding.rewrites == ["A river bank."]


class TestLLMAnalysis:
    def test_parses_a_well_formed_response(self):
        payload = {
            "ambiguous": "true",
            "overall_confidence": 0.9,
            "clarity_verdict": "ambiguous",
            "ambiguities": [
                {
                    "text_span": "with the telescope",
                    "type": "syntactic",
                    "confidence": 0.9,
                    "interpretations": [
                        {"meaning": "The observer used a telescope.",
                         "explanation": "PP attaches to the verb phrase."},
                        {"meaning": "The man had a telescope.",
                         "explanation": "PP attaches to the noun phrase."},
                    ],
                    "context_resolved": False,
                    "rewrites": ["I used a telescope to see the man."],
                }
            ],
        }
        analysis = LLMAnalysis.model_validate(payload)
        assert analysis.ambiguous is True
        assert analysis.clarity_verdict is ClarityVerdict.AMBIGUOUS
        assert len(analysis.ambiguities[0].interpretations) == 2

    def test_empty_response_defaults_to_clear(self):
        analysis = LLMAnalysis.model_validate({})
        assert analysis.ambiguous is False
        assert analysis.clarity_verdict is ClarityVerdict.CLEAR
        assert analysis.ambiguities == []
