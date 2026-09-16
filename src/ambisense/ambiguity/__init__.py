"""Rule-based ambiguity candidate detection (Phase 2).

This layer proposes *candidates* - places where linguistic structure permits
more than one reading. It does not decide whether a sentence is genuinely
ambiguous; that judgement belongs to the Phase 4 LLM adjudication layer.

Detectors are deliberately high-recall: they over-flag, and the later layer
filters. No detector performs network access or uses an LLM.
"""

from ambisense.ambiguity.base import Detector
from ambisense.ambiguity.lexical import LexicalDetector
from ambisense.ambiguity.pragmatic import PragmaticDetector
from ambisense.ambiguity.referential import ReferentialDetector
from ambisense.ambiguity.registry import (
    DETECTOR_CLASSES,
    DetectorRegistry,
    deduplicate,
    sort_candidates,
)
from ambisense.ambiguity.scope import ScopeDetector
from ambisense.ambiguity.semantic import SemanticDetector
from ambisense.ambiguity.syntactic import SyntacticDetector

__all__ = [
    "Detector",
    "DetectorRegistry",
    "DETECTOR_CLASSES",
    "deduplicate",
    "sort_candidates",
    "LexicalDetector",
    "SyntacticDetector",
    "ReferentialDetector",
    "SemanticDetector",
    "ScopeDetector",
    "PragmaticDetector",
]
