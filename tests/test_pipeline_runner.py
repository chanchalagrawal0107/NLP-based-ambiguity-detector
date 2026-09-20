"""Phase 5 wiring: ``pipeline.runner.analyze`` attaches a sentence summary.

Runs the real Phases 1-3 evidence gathering against a fake LLM provider (no
network, no real Ollama - the autouse ``_block_real_http`` fixture in
conftest.py would fail the test if one were attempted), then checks that the
``summary`` on the returned report is consistent with its own
``adjudications`` list.
"""

from __future__ import annotations

import json
import re

import pytest

from ambisense.config import AdjudicationConfig, LLMConfig, load_settings
from ambisense.llm.adjudicator import LLMAdjudicator
from ambisense.llm.providers.base import LLMResponse
from ambisense.pipeline import analyze
from ambisense.schemas import SentenceVerdict

SECRET = "sk-live-DO-NOT-LEAK-987654321"


def reply(body):
    return LLMResponse(
        content=json.dumps(body), prompt_tokens=10, completion_tokens=10
    )


class ScriptedProvider:
    """Answers every adjudication request the same way; ignores rewrites."""

    name, model = "fake", "fake-model"

    def __init__(self, verdict: str):
        self.verdict = verdict

    def complete(self, request):
        content = request.messages[-1].content
        if "CONFIRMED AMBIGUITIES" in content:
            return reply({"rewrites": []})
        ids = re.findall(r'"candidate_id": "(c\d+)"', content)
        interpretations = (
            [{"meaning": "reading one"}, {"meaning": "reading two"}]
            if self.verdict == "genuine_ambiguity"
            else [{"meaning": "the only reading"}]
        )
        return reply({"judgements": [
            {
                "candidate_id": i,
                "verdict": self.verdict,
                "interpretations": interpretations,
                "explanation": "Justification.",
                "confidence": 0.8,
                "selected_sense": None,
            }
            for i in ids
        ]})


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _adjudicator(verdict: str) -> LLMAdjudicator:
    return LLMAdjudicator(
        LLMConfig(api_key=SECRET, max_retries=1),
        AdjudicationConfig(),
        ScriptedProvider(verdict),
        cache=None,
        sleep=lambda seconds: None,
    )


class TestAnalyzeAttachesSummary:
    def test_genuine_verdicts_roll_up_to_ambiguous(self, settings):
        report = analyze(
            settings,
            "I saw the man with the telescope.",
            None,
            adjudicator=_adjudicator("genuine_ambiguity"),
        )
        assert report.summary.verdict is SentenceVerdict.AMBIGUOUS
        assert report.summary.total_candidates == len(report.adjudications)
        assert report.summary.genuine_count == len(report.adjudications)
        assert report.summary.genuine_candidate_ids == [
            a.candidate_id for a in report.adjudications
        ]

    def test_rejected_verdicts_roll_up_to_not_ambiguous(self, settings):
        report = analyze(
            settings,
            "I saw the man with the telescope.",
            None,
            adjudicator=_adjudicator("not_ambiguous"),
        )
        assert report.summary.verdict is SentenceVerdict.NOT_AMBIGUOUS
        assert report.summary.genuine_count == 0
        assert report.summary.not_ambiguous_count == len(report.adjudications)

    def test_no_candidates_rolls_up_to_no_candidates(self, settings):
        # No detector fires on this sentence (unlike most ordinary sentences,
        # where the high-recall lexical detector flags common words), so no
        # LLM request is made at all.
        report = analyze(
            settings, "The sky is blue.", None, adjudicator=_adjudicator("not_ambiguous")
        )
        assert report.adjudications == []
        assert report.summary.verdict is SentenceVerdict.NO_CANDIDATES
        assert report.summary.total_candidates == 0
