"""Phases 1-5 orchestration: evidence, LLM adjudication, sentence rollup."""

from __future__ import annotations

from ambisense.pipeline.evidence import Evidence, SetupError, gather_evidence
from ambisense.pipeline.runner import analyze
from ambisense.pipeline.summary import build_sentence_summary

__all__ = [
    "Evidence",
    "SetupError",
    "gather_evidence",
    "analyze",
    "build_sentence_summary",
]
