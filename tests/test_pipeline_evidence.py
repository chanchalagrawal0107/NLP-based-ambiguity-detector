"""``gather_evidence`` accepts and reuses an injected ``LinguisticAnalyzer``.

This is what lets the Streamlit app (Phase 6) build the spaCy pipeline once
via ``st.cache_resource`` instead of reloading it from disk on every rerun.
The default (no ``analyzer`` passed) must keep building its own, so every
existing caller is unaffected.
"""

from __future__ import annotations

from ambisense.config import load_settings
from ambisense.pipeline.evidence import gather_evidence
from ambisense.preprocessing import LinguisticAnalyzer


class TestInjectedAnalyzer:
    def test_default_builds_its_own_analyzer(self):
        settings = load_settings()
        evidence = gather_evidence(settings, "I saw the man with the telescope.", None)
        assert evidence.text == "I saw the man with the telescope."

    def test_injected_analyzer_is_reused_not_rebuilt(self):
        settings = load_settings()
        analyzer = LinguisticAnalyzer(settings.nlp.spacy_model)
        calls = []
        original_analyze = analyzer.analyze

        def spying_analyze(text):
            calls.append(text)
            return original_analyze(text)

        analyzer.analyze = spying_analyze

        evidence = gather_evidence(
            settings, "I saw the man with the telescope.", "Extra context.",
            analyzer=analyzer,
        )

        assert evidence.text == "I saw the man with the telescope."
        # The same analyzer instance did both the main-text and context calls.
        assert calls == ["I saw the man with the telescope.", "Extra context."]
