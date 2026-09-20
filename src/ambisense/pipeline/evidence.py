"""Phases 1-3, gathered once: validation -> linguistic analysis -> detectors
-> sense ranking.

Shared by every command that needs rule-based candidates and/or semantic
evidence (``--analyze-semantics``, ``--analyze``, ``--live-llm-test``), so the
evidence pipeline is defined in exactly one place. Makes no network call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ambisense.ambiguity import DetectorRegistry
from ambisense.logging_setup import get_logger
from ambisense.preprocessing import LinguisticAnalyzer, clean_and_validate
from ambisense.schemas import AmbiguityCandidate, SenseRanking
from ambisense.semantic import SemanticAnalyzer, SpacyEmbeddingBackend

logger = get_logger(__name__)


class SetupError(RuntimeError):
    """A required resource (e.g. word vectors) is missing."""


@dataclass
class Evidence:
    """Everything Phases 1-3 produce for one input, gathered once."""

    text: str
    context: str | None
    candidates: list[AmbiguityCandidate]
    rankings: list[SenseRanking]


def gather_evidence(
    settings,
    text: str,
    context: str | None,
    *,
    analyzer: Optional[LinguisticAnalyzer] = None,
) -> Evidence:
    """Validation -> linguistic analysis -> detectors -> sense ranking.

    Args:
        analyzer: Reused when supplied, so the spaCy model is not reloaded
            from disk on every call (e.g. across Streamlit reruns); built
            fresh from ``settings`` otherwise.
    """
    cleaned = clean_and_validate(text, context, settings.nlp)
    for warning in cleaned.warnings:
        logger.warning("%s", warning)

    analyzer = analyzer or LinguisticAnalyzer(settings.nlp.spacy_model)
    analysis = analyzer.analyze(cleaned.text)
    context_analysis = (
        analyzer.analyze(cleaned.context) if cleaned.context else None
    )
    candidates = DetectorRegistry(settings.detectors).run(analysis)

    backend = SpacyEmbeddingBackend(analyzer.nlp)
    if not backend.has_vectors:
        raise SetupError(
            f"Semantic analysis needs a spaCy model with word vectors. "
            f"'{settings.nlp.spacy_model}' has none - install en_core_web_md."
        )
    semantic = SemanticAnalyzer(settings.semantic_analysis, backend)
    rankings = semantic.analyze_candidates(candidates, analysis, context_analysis)
    return Evidence(cleaned.text, cleaned.context, candidates, rankings)
