"""Tests for the shared schemas.

These focus on the *defensive* behaviour: the validators exist because small
LLMs return near-miss types, percentage confidences and single strings where a
list is required. If these tests pass, a sloppy model response still produces
a valid report instead of a crash.
"""

from __future__ import annotations

import pytest

from ambisense.schemas import AmbiguityType


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
