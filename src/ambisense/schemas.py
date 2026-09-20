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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class SemanticAnalysisStatus(str, Enum):
    """Why a sense ranking did or did not produce scores.

    Phase 3 must never manufacture a similarity score. When a ranking cannot
    be computed the reason is reported explicitly instead.
    """

    RANKED = "ranked"
    NO_SENSES = "no_senses_found"
    NO_CONTEXT_VECTOR = "no_usable_context_vector"
    NO_GLOSS_VECTORS = "no_usable_gloss_vectors"
    WORDNET_UNAVAILABLE = "wordnet_unavailable"
    DISABLED = "disabled"

    @property
    def is_usable(self) -> bool:
        return self is SemanticAnalysisStatus.RANKED


class SenseOption(BaseModel):
    """One WordNet sense of a word, optionally scored against the context."""

    sense_key: str = Field(description="WordNet synset name, e.g. 'bank.n.01'.")
    definition: str
    pos: str
    examples: list[str] = Field(default_factory=list)
    lemma_names: list[str] = Field(default_factory=list)
    context_similarity: Optional[float] = Field(
        default=None,
        description=(
            "Cosine similarity between the context vector and this sense's "
            "gloss vector, in [-1, 1]. This is a similarity score, NOT a "
            "probability that the word carries this sense."
        ),
    )
    rank: Optional[int] = Field(
        default=None,
        description="1-based position after sorting by similarity.",
    )


class SenseRanking(BaseModel):
    """Result of ranking one word's WordNet senses against its context.

    This records *evidence*, not a decision. A top-ranked sense means only
    that its gloss is the most similar of those retrieved - it does not mean
    the word definitely carries that sense.
    """

    word: str
    lemma: str
    pos: str
    status: SemanticAnalysisStatus = SemanticAnalysisStatus.RANKED
    senses: list[SenseOption] = Field(default_factory=list)
    top_sense_key: Optional[str] = None
    margin: Optional[float] = Field(
        default=None,
        description="Similarity gap between the best and second-best sense.",
    )
    resolved_by_context: bool = Field(
        default=False,
        description=(
            "True when the top sense leads the runner-up by more than the "
            "configured margin. An uncalibrated heuristic, not a guarantee."
        ),
    )
    context_words: list[str] = Field(
        default_factory=list,
        description="The content words used to build the context vector.",
    )
    senses_available: int = Field(
        default=0,
        description="Senses WordNet held, before the configured cap.",
    )
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    note: str = ""

    @property
    def top_sense(self) -> Optional[SenseOption]:
        return self.senses[0] if self.senses else None


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


# ---------------------------------------------------------------------------
# 4b. Phase 4 adjudication models (candidate-level LLM contract)
#
# An earlier, sentence-level contract ("is this sentence ambiguous?") was
# replaced by the candidate-level contract below: a narrower, testable
# question per rule-based candidate instead of one verdict for the whole
# sentence.
# ---------------------------------------------------------------------------


class AdjudicationVerdict(str, Enum):
    """The LLM's judgement on one rule-based candidate."""

    GENUINE_AMBIGUITY = "genuine_ambiguity"
    NOT_AMBIGUOUS = "not_ambiguous"
    UNCERTAIN = "uncertain"

    @classmethod
    def parse(cls, value: Any) -> "AdjudicationVerdict":
        """Accept known spellings; **reject** anything unrecognised.

        Unlike ``AmbiguityType.coerce``, an unknown verdict is *not* mapped to
        a default. Turning an unreadable verdict into ``uncertain`` would
        convert a broken response into an answer, so it raises instead and
        the candidate is reported as an invalid response.
        """
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if text == member.value:
                return member
        aliases = {
            "genuine": cls.GENUINE_AMBIGUITY,
            "ambiguous": cls.GENUINE_AMBIGUITY,
            "genuinely_ambiguous": cls.GENUINE_AMBIGUITY,
            "not_genuine": cls.NOT_AMBIGUOUS,
            "not_genuinely_ambiguous": cls.NOT_AMBIGUOUS,
            "unambiguous": cls.NOT_AMBIGUOUS,
            "false_positive": cls.NOT_AMBIGUOUS,
            "unsure": cls.UNCERTAIN,
            "undetermined": cls.UNCERTAIN,
        }
        if text in aliases:
            return aliases[text]
        raise ValueError(
            f"unrecognised verdict {value!r}; expected one of "
            f"{[member.value for member in cls]}"
        )


class AdjudicationStatus(str, Enum):
    """Whether an LLM judgement exists for a candidate, and if not, why.

    A failure is never reported as ``not_ambiguous``: "the API was down" and
    "the sentence is clear" are different facts.
    """

    ADJUDICATED = "adjudicated"
    LLM_NOT_CONFIGURED = "llm_not_configured"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_INVALID_RESPONSE = "llm_invalid_response"


class RewriteStatus(str, Enum):
    """Outcome of the separate rewrite-generation step."""

    GENERATED = "generated"
    NOT_APPLICABLE = "not_applicable"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_INVALID_RESPONSE = "llm_invalid_response"


#: Spellings a model commonly uses to mean "no sense selected".
_NULL_SENSE_SPELLINGS = frozenset({"", "none", "null", "n/a", "not_applicable"})


def parse_optional_confidence(value: Any) -> Optional[float]:
    """Parse a self-reported confidence **without** inventing one.

    ``clamp_confidence`` returns 0.5 for unreadable input, which is acceptable
    for the Phase 1 contract but would manufacture a confidence here. This
    returns ``None`` instead, so a missing number stays visibly missing.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    if _PERCENT_FLOOR <= number <= 100.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


class LLMJudgement(BaseModel):
    """What the LLM returns for one candidate, after validation.

    ``confidence`` is the model's **self-reported** confidence in its verdict.
    It is not a calibrated probability that the verdict is correct.
    """

    candidate_id: str = Field(min_length=1)
    verdict: AdjudicationVerdict
    interpretations: list[Interpretation] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    selected_sense: Optional[str] = Field(
        default=None,
        description=(
            "The Phase 3 sense key (e.g. 'crane.n.04') the model judges to be "
            "the meaning of the span, chosen from the senses it was shown; "
            "None when it selects none of them. Whether the key is one it was "
            "actually shown is checked per candidate by the parser. The model "
            "does NOT report agreement with Phase 3 - that is derived in code."
        ),
    )

    @field_validator("selected_sense", mode="before")
    @classmethod
    def _parse_selected_sense(cls, value: Any) -> Optional[str]:
        """Normalise explicit "no selection" spellings; reject non-strings.

        A number, list or object is malformed and raises, so it becomes a
        validation problem rather than being guessed into None.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(
                f"selected_sense must be a sense key string or null, "
                f"got {type(value).__name__}"
            )
        text = value.strip()
        return None if text.lower() in _NULL_SENSE_SPELLINGS else text

    @field_validator("verdict", mode="before")
    @classmethod
    def _parse_verdict(cls, value: Any) -> AdjudicationVerdict:
        return AdjudicationVerdict.parse(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _parse_confidence(cls, value: Any) -> Optional[float]:
        return parse_optional_confidence(value)

    @field_validator("explanation", mode="before")
    @classmethod
    def _strip_explanation(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("interpretations", mode="before")
    @classmethod
    def _normalise_interpretations(cls, value: Any) -> list[dict[str, str]]:
        """Accept strings or objects; drop blanks and exact duplicates."""
        if not isinstance(value, list):
            return []
        cleaned: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in value:
            if isinstance(item, str):
                item = {"meaning": item}
            if not isinstance(item, dict):
                continue
            meaning = str(item.get("meaning") or "").strip()
            key = meaning.lower()
            if not meaning or key in seen:
                continue
            seen.add(key)
            cleaned.append({
                "meaning": meaning,
                "explanation": str(item.get("explanation") or "").strip(),
            })
        return cleaned

    @model_validator(mode="after")
    def _genuine_needs_two_readings(self) -> "LLMJudgement":
        """A genuine ambiguity with fewer than two readings is self-contradictory."""
        if (self.verdict is AdjudicationVerdict.GENUINE_AMBIGUITY
                and len(self.interpretations) < 2):
            raise ValueError(
                "verdict genuine_ambiguity requires at least two distinct "
                "interpretations"
            )
        return self


class CandidateAdjudication(BaseModel):
    """One candidate, the evidence it was judged on, and the outcome."""

    candidate_id: str
    candidate: AmbiguityCandidate
    sense_ranking: Optional[SenseRanking] = None
    status: AdjudicationStatus
    judgement: Optional[LLMJudgement] = None
    error: str = Field(
        default="",
        description="Why no judgement exists. Never contains secrets.",
    )
    rewrites: list[str] = Field(default_factory=list)
    rewrite_status: RewriteStatus = RewriteStatus.NOT_APPLICABLE
    rewrite_error: str = ""

    # -- application-derived, never taken from the LLM reply --
    phase3_top_sense: Optional[str] = Field(
        default=None,
        description="Phase 3's rank-1 sense key, when usable ranking evidence exists.",
    )
    agrees_with_phase3: Optional[bool] = Field(
        default=None,
        description=(
            "Computed by application code: selected_sense == phase3_top_sense. "
            "None when there is no Phase 3 top sense or the LLM selected no "
            "sense. Not an LLM prediction."
        ),
    )

    @property
    def is_genuine(self) -> bool:
        return (
            self.judgement is not None
            and self.judgement.verdict is AdjudicationVerdict.GENUINE_AMBIGUITY
        )


# ---------------------------------------------------------------------------
# 5. Final report
# ---------------------------------------------------------------------------


class PipelineDiagnostics(BaseModel):
    """Operational facts about a single run - useful in the demo and the viva."""

    mode: AnalysisMode = AnalysisMode.FULL
    llm_used: bool = False
    llm_provider: str = ""
    llm_model: str = ""
    llm_attempts: int = 0
    llm_prompt_tokens: Optional[int] = Field(
        default=None,
        description="Prompt tokens reported by the provider (measured, summed).",
    )
    llm_completion_tokens: Optional[int] = Field(
        default=None,
        description="Completion tokens reported by the provider (measured, summed).",
    )
    repair_attempted: bool = False
    cache_hit: bool = False
    elapsed_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    candidates_found: int = 0
    candidates_confirmed: int = 0


class SentenceVerdict(str, Enum):
    """A deterministic rollup of candidate verdicts - never a fabricated score.

    Every value here is a direct, checkable restatement of the per-candidate
    verdicts already visible in ``AdjudicationReport.adjudications``: nothing
    is computed that could not be recomputed by hand from that list. This is
    a label, not a probability, and it is not a substitute for reading the
    candidates - see ``SentenceSummary.explanation``.
    """

    NO_CANDIDATES = "no_candidates"
    AMBIGUOUS = "ambiguous"
    NOT_AMBIGUOUS = "not_ambiguous"
    UNCERTAIN = "uncertain"
    INCOMPLETE = "incomplete"


class SentenceSummary(BaseModel):
    """A transparent count-based rollup of one sentence's candidates.

    Deliberately contains no numeric score. Every field is a plain count or
    a verdict derived by simple precedence rules from
    ``AdjudicationReport.adjudications`` - see
    ``pipeline.summary.build_sentence_summary`` for the exact rule, which is
    the only place this is computed.
    """

    verdict: SentenceVerdict
    total_candidates: int = 0
    genuine_count: int = 0
    not_ambiguous_count: int = 0
    uncertain_count: int = 0
    unresolved_count: int = 0
    genuine_candidate_ids: list[str] = Field(default_factory=list)
    explanation: str = ""


class AdjudicationReport(BaseModel):
    """The object returned by the pipeline and rendered by every interface.

    Carries a deterministic per-sentence rollup (``summary``) alongside the
    candidate-level detail, but deliberately no fabricated numeric score:
    aggregating candidate-level judgements into one uncalibrated number would
    hide the per-candidate evidence this project is built to keep visible.
    Every field of ``summary`` is recomputable by hand from ``adjudications``.
    """

    text: str
    context: Optional[str] = None
    adjudications: list[CandidateAdjudication] = Field(default_factory=list)
    diagnostics: PipelineDiagnostics = Field(default_factory=PipelineDiagnostics)
    summary: SentenceSummary = Field(
        default_factory=lambda: SentenceSummary(verdict=SentenceVerdict.NO_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Values at or above this are interpreted as percentages rather than as
# out-of-range probabilities. See ``parse_optional_confidence`` above.
_PERCENT_FLOOR = 10.0
