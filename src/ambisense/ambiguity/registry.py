"""Detector registry: builds the enabled detectors and runs them.

The registry is the only place that knows the full set of detectors. It reads
``config.detectors`` to decide which to instantiate, runs each against the
*same* :class:`LinguisticAnalysis`, and returns a deterministically ordered,
deduplicated candidate list.

Determinism matters here. The registry is what a demonstration and the Phase 7
evaluation both call, so identical input must give byte-identical output every
run: detectors execute in declaration order, and results are sorted by a total
ordering with no ties left to chance.
"""

from __future__ import annotations

from typing import Iterable, Optional, Type

from ambisense.ambiguity.base import Detector
from ambisense.ambiguity.lexical import LexicalDetector
from ambisense.ambiguity.pragmatic import PragmaticDetector
from ambisense.ambiguity.referential import ReferentialDetector
from ambisense.ambiguity.scope import ScopeDetector
from ambisense.ambiguity.semantic import SemanticDetector
from ambisense.ambiguity.syntactic import SyntacticDetector
from ambisense.config import DetectorsConfig
from ambisense.logging_setup import get_logger
from ambisense.schemas import AmbiguityCandidate, LinguisticAnalysis

logger = get_logger(__name__)


#: Detector name -> class. The key must match the config field name.
DETECTOR_CLASSES: dict[str, Type[Detector]] = {
    "lexical": LexicalDetector,
    "syntactic": SyntacticDetector,
    "referential": ReferentialDetector,
    "semantic": SemanticDetector,
    "scope": ScopeDetector,
    "pragmatic": PragmaticDetector,
}


class DetectorRegistry:
    """Builds and runs the configured set of detectors."""

    def __init__(self, config: Optional[DetectorsConfig] = None) -> None:
        self.config = config or DetectorsConfig()
        self._detectors: list[Detector] = self._build()

    # -- construction ----------------------------------------------------

    def _build(self) -> list[Detector]:
        detectors: list[Detector] = []
        for name in self.config.enabled_names():
            detector_class = DETECTOR_CLASSES.get(name)
            if detector_class is None:
                logger.warning(
                    "Detector '%s' is enabled in config but has no "
                    "implementation; skipping.", name,
                )
                continue
            detectors.append(detector_class(self.config.settings_for(name)))
        logger.debug(
            "Registry built with %d detector(s): %s",
            len(detectors), ", ".join(d.name for d in detectors) or "none",
        )
        return detectors

    # -- introspection ---------------------------------------------------

    @property
    def detectors(self) -> list[Detector]:
        return list(self._detectors)

    @property
    def detector_names(self) -> list[str]:
        return [detector.name for detector in self._detectors]

    # -- execution -------------------------------------------------------

    def run(self, analysis: LinguisticAnalysis) -> list[AmbiguityCandidate]:
        """Run every enabled detector and return deduplicated candidates.

        A failure inside one detector is logged and skipped rather than
        allowed to abort the whole analysis: a single bad rule should not cost
        the user every other detector's findings.
        """
        candidates: list[AmbiguityCandidate] = []
        for detector in self._detectors:
            try:
                produced = detector.detect(analysis)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "Detector '%s' raised; skipping it for this input.",
                    detector.name,
                )
                continue
            logger.debug(
                "Detector '%s' produced %d candidate(s)",
                detector.name, len(produced),
            )
            candidates.extend(produced)

        return sort_candidates(deduplicate(candidates))


# ---------------------------------------------------------------------------
# Deduplication and ordering
# ---------------------------------------------------------------------------


def _identity(candidate: AmbiguityCandidate) -> tuple:
    """The key that decides whether two candidates are "the same finding".

    Deliberately includes the ambiguity **type** and the detector's own
    ``rule`` marker, not just the character span. Two different kinds of
    ambiguity can legitimately occupy the same span - "bank" can be both a
    lexical candidate and part of a syntactic one - and collapsing them would
    destroy information the LLM needs.
    """
    return (
        candidate.char_start,
        candidate.char_end,
        candidate.type_hint.value,
        str(candidate.evidence.get("rule", "")),
    )


def deduplicate(
    candidates: Iterable[AmbiguityCandidate],
) -> list[AmbiguityCandidate]:
    """Merge candidates that share span + type + rule.

    When two detectors report the same finding, the higher ``prior`` wins and
    the other detector's name is recorded in ``evidence['also_detected_by']``,
    so agreement between rules is preserved rather than discarded.
    """
    merged: dict[tuple, AmbiguityCandidate] = {}
    for candidate in candidates:
        key = _identity(candidate)
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue

        winner, loser = (
            (candidate, existing)
            if candidate.prior > existing.prior
            else (existing, candidate)
        )
        if winner.detector_name != loser.detector_name:
            supporters = set(winner.evidence.get("also_detected_by", []))
            supporters.add(loser.detector_name)
            winner.evidence["also_detected_by"] = sorted(supporters)
        merged[key] = winner
    return list(merged.values())


def sort_candidates(
    candidates: Iterable[AmbiguityCandidate],
) -> list[AmbiguityCandidate]:
    """Order candidates deterministically.

    Document order first (so the report reads left to right), then strongest
    signal, then type and detector name to break any remaining ties. No two
    candidates can compare equal, so the order is fully reproducible.
    """
    return sorted(
        candidates,
        key=lambda c: (
            c.char_start,
            c.char_end,
            -c.prior,
            c.type_hint.value,
            c.detector_name,
        ),
    )
