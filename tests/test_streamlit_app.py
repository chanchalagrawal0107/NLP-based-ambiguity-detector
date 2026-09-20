"""Phase 6: the Streamlit app's pure helpers and a headless smoke render.

``streamlit.testing.v1.AppTest`` executes the script without a browser. The
real Ollama server is never contacted: the health check is stubbed so the
sidebar state is deterministic on every machine.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ambisense.schemas import (
    AdjudicationStatus,
    AmbiguityCandidate,
    AmbiguityType,
    CandidateAdjudication,
    LLMJudgement,
    AdjudicationReport,
)

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("streamlit_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(candidate_id, start, end, text, verdict):
    interpretations = (
        [{"meaning": "one"}, {"meaning": "two"}]
        if verdict == "genuine_ambiguity" else [{"meaning": "one"}]
    )
    return CandidateAdjudication(
        candidate_id=candidate_id,
        candidate=AmbiguityCandidate(
            span_text=text, char_start=start, char_end=end,
            type_hint=AmbiguityType.SYNTACTIC, detector_name="syntactic",
            prior=0.7, explanation="Rule <fired> & more.",
        ),
        status=AdjudicationStatus.ADJUDICATED,
        judgement=LLMJudgement(
            candidate_id=candidate_id, verdict=verdict,
            interpretations=interpretations, explanation="x", confidence=0.5,
        ),
    )


class TestRenderHighlightedText:
    TEXT = "I saw the man with the telescope."

    def test_only_genuine_spans_are_marked(self, app_module):
        report = AdjudicationReport(text=self.TEXT, adjudications=[
            _item("c1", 14, 32, "with the telescope", "genuine_ambiguity"),
            _item("c2", 6, 13, "the man", "not_ambiguous"),
        ])
        out = app_module.render_highlighted_text(self.TEXT, report)
        assert out.count("<mark") == 1
        assert ">with the telescope</mark>" in out

    def test_no_genuine_spans_returns_escaped_text_unchanged(self, app_module):
        report = AdjudicationReport(text="a < b", adjudications=[])
        assert app_module.render_highlighted_text("a < b", report) == "a &lt; b"

    def test_tooltip_and_text_are_html_escaped(self, app_module):
        text = "x <b>y</b> z"
        report = AdjudicationReport(text=text, adjudications=[
            _item("c1", 2, 10, "<b>y</b>", "genuine_ambiguity"),
        ])
        out = app_module.render_highlighted_text(text, report)
        assert "<b>" not in out
        assert "&lt;b&gt;y&lt;/b&gt;</mark>" in out
        assert "Rule &lt;fired&gt; &amp; more." in out

    def test_overlapping_spans_do_not_duplicate_text(self, app_module):
        report = AdjudicationReport(text=self.TEXT, adjudications=[
            _item("c1", 6, 32, "the man with the telescope", "genuine_ambiguity"),
            _item("c2", 14, 32, "with the telescope", "genuine_ambiguity"),
        ])
        out = app_module.render_highlighted_text(self.TEXT, report)
        assert out.count("<mark") == 1
        assert out.count("telescope") == 1  # the overlapping second span is skipped


def test_app_renders_headlessly_with_server_ready(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from ambisense.llm import health

    monkeypatch.setattr(
        health, "check_server",
        lambda config, **kw: health.ServerStatus(
            reachable=True, model_installed=True, installed_models=(config.model,)
        ),
    )
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert not at.exception
    assert [t.value for t in at.title] == ["AmbiSense"]
    assert at.button[0].disabled is False
    assert len(at.selectbox[0].options) > 1  # "(custom)" + demo sentences


def test_app_disables_analysis_when_server_unreachable(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from ambisense.llm import health

    monkeypatch.setattr(
        health, "check_server", lambda config, **kw: health.ServerStatus(reachable=False)
    )
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert not at.exception
    assert at.button[0].disabled is True
    assert any("unreachable" in e.value for e in at.sidebar.error)
