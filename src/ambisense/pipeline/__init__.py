"""Phases 1-4 orchestration: evidence gathering plus LLM adjudication."""

from __future__ import annotations

from ambisense.pipeline.evidence import Evidence, SetupError, gather_evidence
from ambisense.pipeline.runner import analyze

__all__ = ["Evidence", "SetupError", "gather_evidence", "analyze"]
