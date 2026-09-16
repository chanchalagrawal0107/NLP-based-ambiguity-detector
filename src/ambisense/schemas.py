"""Pydantic models shared by every layer of the AmbiSense pipeline.

This module is deliberately the single source of truth for all data that
crosses a layer boundary. Four groups of models live here:

1. **Linguistic models**   - the output of the spaCy analysis layer.
2. **Detection models**    - what the rule-based detectors emit.
3. **Semantic models**     - WordNet senses ranked against the context.
4. **LLM contract models** - the strict schema the LLM must fill in, plus the
   final report handed to every interface.

Keeping them together makes the interfaces between layers easy to inspect and
easy to explain: the whole data flow of the project can be read in one file.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AmbiguityType(str, Enum):
    """The ambiguity categories AmbiSense can report.

    ``UNKNOWN`` is a first-class value on purpose: the system must be able to
    say "something is ambiguous here but I cannot confidently label it"
    rather than forcing every case into a category.
    """

    LEXICAL = "lexical"
    SYNTACTIC = "syntactic"
    REFERENTIAL = "referential"
    SEMANTIC = "semantic"
    SCOPE = "scope"
    PRAGMATIC = "pragmatic"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: Any) -> "AmbiguityType":
        """Map a free-form LLM string onto a known type, defaulting to UNKNOWN.

        Smaller language models frequently answer with near-misses such as
        "structural" or "syntax". Rather than rejecting the whole response we
        normalise what we recognise and fall back to UNKNOWN otherwise, which
        is a truthful answer rather than a guess.
        """
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        for member in cls:
            if text == member.value:
                return member
        aliases = {
            "structural": cls.SYNTACTIC,
            "syntax": cls.SYNTACTIC,
            "grammatical": cls.SYNTACTIC,
            "attachment": cls.SYNTACTIC,
            "word sense": cls.LEXICAL,
            "word-sense": cls.LEXICAL,
            "polysemy": cls.LEXICAL,
            "homonymy": cls.LEXICAL,
            "anaphoric": cls.REFERENTIAL,
            "anaphora": cls.REFERENTIAL,
            "pronoun": cls.REFERENTIAL,
            "coreference": cls.REFERENTIAL,
            "meaning": cls.SEMANTIC,
            "quantifier": cls.SCOPE,
            "quantifier scope": cls.SCOPE,
            "negation": cls.SCOPE,
            "contextual": cls.PRAGMATIC,
            "speech act": cls.PRAGMATIC,
            "discourse": cls.PRAGMATIC,
        }
        return aliases.get(text, cls.UNKNOWN)


class ClarityVerdict(str, Enum):
    """How the system classifies the *overall* clarity of the input.

    Section 10 of the project brief: not every unclear sentence is ambiguous.
    Separating these five outcomes is what stops the system from labelling
    "The movie was good" as ambiguous.
    """

    AMBIGUOUS = "ambiguous"
    VAGUE = "vague"
    UNDERSPECIFIED = "underspecified"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    CLEAR = "clear"

    @classmethod
    def coerce(cls, value: Any) -> "ClarityVerdict":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        for member in cls:
            if text == member.value:
                return member
        aliases = {
            "unambiguous": cls.CLEAR,
            "clear_text": cls.CLEAR,
            "not_ambiguous": cls.CLEAR,
            "imprecise": cls.VAGUE,
            "subjective": cls.VAGUE,
            "incomplete": cls.UNDERSPECIFIED,
            "missing_context": cls.INSUFFICIENT_CONTEXT,
            "needs_context": cls.INSUFFICIENT_CONTEXT,
        }
        return aliases.get(text, cls.CLEAR)


class AnalysisMode(str, Enum):
    """Which components actually contributed to a report."""

    FULL = "full"
    DEGRADED = "degraded"


# ---------------------------------------------------------------------------
# 1. Linguistic models (spaCy analysis layer)
# ---------------------------------------------------------------------------


class TokenInfo(BaseModel):
    """One token with the linguistic features the detectors rely on."""

    index: int
    text: str
    lemma: str
    pos: str = Field(description="Coarse universal POS tag, e.g. NOUN")
    tag: str = Field(description="Fine-grained Penn Treebank tag, e.g. NNS")
    dep: str = Field(description="Dependency relation to its head")
    head_index: int
    head_text: str
    is_stop: bool = False
    is_alpha: bool = True
    char_start: int = 0
    char_end: int = 0
    sentence_index: int = 0


class EntityInfo(BaseModel):
    """A named entity found by spaCy's NER component."""

    text: str
    label: str
    char_start: int
    char_end: int
    sentence_index: int = 0


class NounChunkInfo(BaseModel):
    """A base noun phrase - the candidate pool for referential ambiguity."""

    text: str
    root_text: str
    root_lemma: str
    root_pos: str
    root_dep: str
    char_start: int
    char_end: int
    sentence_index: int = 0
    is_plural: bool = False
    is_animate_candidate: bool = Field(
        default=False,
        description=(
            "Whether this phrase could be the antecedent of a personal "
            "pronoun. True for named people, organisations and national or "
            "religious groups, and for role nouns such as 'the manager'. "
            "Named 'animate candidate' rather than 'person' because "
            "organisations are included: 'The company said it would...' "
            "takes a pronoun, but a company is not a person."
        ),
    )


class SentenceInfo(BaseModel):
    """A single segmented sentence."""

    index: int
    text: str
    char_start: int
    char_end: int


class LinguisticAnalysis(BaseModel):
    """Everything the traditional NLP layer extracted from one input.

    This object is consumed by the detectors and, in condensed form, is
    serialised into the LLM prompt as *evidence*.
    """

    text: str
    tokens: list[TokenInfo] = Field(default_factory=list)
    sentences: list[SentenceInfo] = Field(default_factory=list)
    entities: list[EntityInfo] = Field(default_factory=list)
    noun_chunks: list[NounChunkInfo] = Field(default_factory=list)

    def tokens_in_sentence(self, sentence_index: int) -> list[TokenInfo]:
        return [t for t in self.tokens if t.sentence_index == sentence_index]

    def token_by_index(self, index: int) -> Optional[TokenInfo]:
        for token in self.tokens:
            if token.index == index:
                return token
        return None

    def children_of(self, head_index: int) -> list[TokenInfo]:
        """Dependency children of a token - the core detector primitive."""
        return [
            t for t in self.tokens
            if t.head_index == head_index and t.index != head_index
        ]


# ---------------------------------------------------------------------------
# 2. Detection models (rule-based detector layer)
# ---------------------------------------------------------------------------


class AmbiguityCandidate(BaseModel):
    """A *possible* ambiguity flagged by a rule-based detector.

    Detectors run at high recall and low precision by design; a candidate is a
    hypothesis, not a finding. ``evidence`` carries the structural facts that
    triggered the rule and is what gets shown to the LLM so it can adjudicate.
    """

    model_config = ConfigDict(use_enum_values=False)

    span_text: str
    char_start: int
    char_end: int
    type_hint: AmbiguityType
    detector_name: str
    prior: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the structural signal alone, before the LLM.",
    )
    evidence: dict[str, Any] = Field(default_factory=dict)
    explanation: str = Field(
        default="",
        description="Short human-readable reason the rule fired.",
    )


# ---------------------------------------------------------------------------
# 3. Semantic layer models (WordNet + embeddings)
# ---------------------------------------------------------------------------


class SenseOption(BaseModel):
    """One WordNet sense of a word, optionally scored against the context."""

    sense_key: str
    definition: str
    pos: str
    examples: list[str] = Field(default_factory=list)
    lemma_names: list[str] = Field(default_factory=list)
    context_similarity: Optional[float] = Field(
        default=None,
        description="Cosine similarity between context and this sense's gloss.",
    )


class SenseRanking(BaseModel):
    """Result of ranking a word's senses against the supplied context."""

    word: str
    lemma: str
    pos: str
    senses: list[SenseOption] = Field(default_factory=list)
    top_sense_key: Optional[str] = None
    margin: Optional[float] = Field(
        default=None,
        description="Similarity gap between the best and second-best sense.",
    )
    resolved_by_context: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# 4. LLM contract models (strict output schema)
# ---------------------------------------------------------------------------


class Interpretation(BaseModel):
    """One plausible reading of an ambiguous span."""

    meaning: str = Field(description="The reading, stated as a clear sentence.")
    explanation: str = Field(
        default="",
        description="Why this reading is linguistically available.",
    )


class AmbiguityFinding(BaseModel):
    """One adjudicated ambiguity: the LLM's verdict on a candidate span."""

    text_span: str
    type: AmbiguityType = AmbiguityType.UNKNOWN
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    interpretations: list[Interpretation] = Field(default_factory=list)
    source_of_ambiguity: str = Field(
        default="",
        description="What linguistic property causes the ambiguity.",
    )
    context_resolved: bool = False
    context_effect: str = Field(
        default="",
        description="How the supplied context changes the reading, if at all.",
    )
    rewrites: list[str] = Field(default_factory=list)

    # -- filled in by the pipeline, not by the LLM --
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    detector_support: list[str] = Field(
        default_factory=list,
        description="Names of rule-based detectors that also flagged this span.",
    )
    ambiguity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, value: Any) -> AmbiguityType:
        return AmbiguityType.coerce(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        """Clamp rather than reject: a model returning 1.2 or "0.8" is common."""
        return clamp_confidence(value)

    @field_validator("rewrites", mode="before")
    @classmethod
    def _clean_rewrites(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class LLMAnalysis(BaseModel):
    """The complete structured response required from the LLM.

    Anything the model returns outside this shape is rejected and a repair
    round-trip is attempted (see ``llm/parser.py``).
    """

    ambiguous: bool = False
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    clarity_verdict: ClarityVerdict = ClarityVerdict.CLEAR
    ambiguities: list[AmbiguityFinding] = Field(default_factory=list)
    reasoning_note: str = Field(
        default="",
        description="One-line summary of why this verdict was reached.",
    )

    @field_validator("clarity_verdict", mode="before")
    @classmethod
    def _coerce_verdict(cls, value: Any) -> ClarityVerdict:
        return ClarityVerdict.coerce(value)

    @field_validator("overall_confidence", mode="before")
    @classmethod
    def _clamp_overall(cls, value: Any) -> float:
        return clamp_confidence(value)

    @field_validator("ambiguous", mode="before")
    @classmethod
    def _coerce_bool(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)


# ---------------------------------------------------------------------------
# 5. Final report
# ---------------------------------------------------------------------------


class ScoreBreakdown(BaseModel):
    """Transparent, per-component view of the ambiguity score.

    Shown in the UI so the number is explainable rather than a black box.
    """

    llm_confidence: float = 0.0
    detector_evidence: float = 0.0
    interpretation_count: float = 0.0
    context_uncertainty: float = 0.0
    weighted_total: float = 0.0
    formula: str = ""


class PipelineDiagnostics(BaseModel):
    """Operational facts about a single run - useful in the demo and the viva."""

    mode: AnalysisMode = AnalysisMode.FULL
    llm_used: bool = False
    llm_provider: str = ""
    llm_model: str = ""
    llm_attempts: int = 0
    repair_attempted: bool = False
    cache_hit: bool = False
    elapsed_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    detectors_run: list[str] = Field(default_factory=list)
    candidates_found: int = 0
    candidates_confirmed: int = 0


class AmbiguityReport(BaseModel):
    """The object returned by the pipeline and rendered by every interface."""

    text: str
    context: Optional[str] = None
    is_ambiguous: bool = False
    ambiguity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    clarity_verdict: ClarityVerdict = ClarityVerdict.CLEAR
    findings: list[AmbiguityFinding] = Field(default_factory=list)
    score_breakdown: Optional[ScoreBreakdown] = None
    sense_rankings: list[SenseRanking] = Field(default_factory=list)
    candidates: list[AmbiguityCandidate] = Field(default_factory=list)
    linguistic_analysis: Optional[LinguisticAnalysis] = None
    diagnostics: PipelineDiagnostics = Field(default_factory=PipelineDiagnostics)
    summary: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Values at or above this are interpreted as percentages rather than as
# out-of-range probabilities. See clamp_confidence below.
_PERCENT_FLOOR = 10.0


def clamp_confidence(value: Any) -> float:
    """Coerce any model-produced confidence into a float in [0, 1].

    Handles the three failure modes seen in practice: a string ("0.8"), a
    percentage (85), and a slightly out-of-range float (1.2).

    The percentage branch requires a value of at least ``_PERCENT_FLOOR``.
    Without that floor, a model overshooting to 1.4 would be read as "1.4%"
    and collapse to 0.014 - a confident answer turned into a near-zero one.
    Values between 1 and the floor are treated as overshoot and clamped to 1.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if _PERCENT_FLOOR <= number <= 100.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))
