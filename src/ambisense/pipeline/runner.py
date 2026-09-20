"""The full pipeline: Phases 1-3 evidence, Phase 4 LLM adjudication, then the
Phase 5 sentence-level rollup.

One function, used by both the single-sentence ``--analyze`` command and the
multi-case ``--live-llm-test`` loop, so the two never drift apart. The live
test builds one adjudicator up front and passes it in, so the response cache
and diagnostics are shared correctly across cases in that run.
"""

from __future__ import annotations

from typing import Optional

from ambisense.llm import build_adjudicator
from ambisense.llm.adjudicator import LLMAdjudicator
from ambisense.pipeline.evidence import gather_evidence
from ambisense.pipeline.summary import build_sentence_summary
from ambisense.preprocessing import LinguisticAnalyzer
from ambisense.schemas import AdjudicationReport


def analyze(
    settings,
    text: str,
    context: str | None,
    adjudicator: Optional[LLMAdjudicator] = None,
    analyzer: Optional[LinguisticAnalyzer] = None,
) -> AdjudicationReport:
    """Evidence from Phases 1-3, LLM adjudication, then the sentence rollup.

    Args:
        adjudicator: Reused across calls (e.g. by ``--live-llm-test`` across
            its cases) when supplied; built fresh from ``settings`` otherwise.
        analyzer: Reused across calls (e.g. by the Streamlit app across
            reruns) when supplied, so the spaCy model is not reloaded from
            disk on every call; built fresh from ``settings`` otherwise.
    """
    evidence = gather_evidence(settings, text, context, analyzer=analyzer)
    adjudicator = adjudicator or build_adjudicator(settings)
    report = adjudicator.adjudicate(
        evidence.text, evidence.context, evidence.candidates, evidence.rankings
    )
    report.summary = build_sentence_summary(report.adjudications)
    return report
