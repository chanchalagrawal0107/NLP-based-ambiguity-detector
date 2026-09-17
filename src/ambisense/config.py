"""Typed configuration loading for AmbiSense.

Two sources, strictly separated:

* ``config/config.yaml`` - all behaviour: model names, thresholds, weights,
  detector switches. Committed to the repository.
* ``.env``               - secrets only (the API key). Never committed.

Nothing in ``src/`` hard-codes a threshold or a model name; everything is read
through the :class:`Settings` object returned by :func:`load_settings`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/ambisense/config.py -> src/ambisense -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT / "data"


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or malformed."""


# ---------------------------------------------------------------------------
# Configuration sections
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout_seconds: float = 45.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    request_json_mode: bool = True
    enable_cache: bool = True
    cache_dir: str = "data/cache"

    # Populated from the environment, never from YAML.
    api_key: Optional[str] = Field(default=None, exclude=True)

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_key.strip()
                    and self.api_key != "your_api_key_here")


class NLPConfig(BaseModel):
    language: str = "en"
    spacy_model: str = "en_core_web_md"
    min_input_length: int = 3
    max_input_length: int = 2000
    max_context_length: int = 2000
    max_sentences: int = 10
    min_ascii_ratio: float = 0.85


class EmbeddingsConfig(BaseModel):
    backend: str = "spacy"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"


class DetectorsConfig(BaseModel):
    lexical: bool = True
    syntactic: bool = True
    referential: bool = True
    semantic: bool = True
    scope: bool = True
    pragmatic: bool = True

    lexical_settings: dict[str, Any] = Field(default_factory=dict)
    syntactic_settings: dict[str, Any] = Field(default_factory=dict)
    referential_settings: dict[str, Any] = Field(default_factory=dict)
    semantic_settings: dict[str, Any] = Field(default_factory=dict)
    scope_settings: dict[str, Any] = Field(default_factory=dict)
    pragmatic_settings: dict[str, Any] = Field(default_factory=dict)

    # Names of every detector the registry knows about, in the order they run.
    # Declared once here so the registry, the CLI and the tests cannot drift
    # out of step with one another.
    DETECTOR_NAMES: ClassVar[tuple[str, ...]] = (
        "lexical", "syntactic", "referential", "semantic", "scope", "pragmatic",
    )

    def is_enabled(self, name: str) -> bool:
        return bool(getattr(self, name, False))

    def settings_for(self, name: str) -> dict[str, Any]:
        """Return the ``<name>_settings`` block, or an empty dict if absent.

        Detectors read their thresholds through this accessor so that a
        missing YAML block degrades to "use the detector's own defaults"
        rather than raising.
        """
        return getattr(self, f"{name}_settings", {}) or {}

    def enabled_names(self) -> list[str]:
        """Enabled detector names, in declaration order (deterministic)."""
        return [name for name in self.DETECTOR_NAMES if self.is_enabled(name)]


class SemanticAnalysisConfig(BaseModel):
    """Phase 3: context-based WordNet sense ranking.

    Defaults for the gloss options were chosen by measurement, not taste -
    see the comments in ``config/config.yaml``.
    """

    enable_context_sense_ranking: bool = True
    max_senses_considered: int = 8

    # Gloss representation
    gloss_includes_examples: bool = True
    gloss_includes_lemma_names: bool = False

    # Context representation
    context_content_pos: list[str] = Field(
        default_factory=lambda: ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
    )
    exclude_target_word: bool = True
    include_user_context: bool = True
    min_context_words: int = 1

    # Reporting
    sense_resolution_margin: float = 0.05
    min_sense_similarity: float = 0.15


class ScoringWeights(BaseModel):
    llm_confidence: float = 0.45
    detector_evidence: float = 0.25
    interpretation_count: float = 0.15
    context_uncertainty: float = 0.15

    def total(self) -> float:
        return (self.llm_confidence + self.detector_evidence
                + self.interpretation_count + self.context_uncertainty)


class ScoringConfig(BaseModel):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    ambiguity_threshold: float = 0.45
    confidence_threshold: float = 0.35
    max_interpretations_for_score: int = 4


class OutputConfig(BaseModel):
    allow_degraded_mode: bool = True
    include_nlp_evidence_in_report: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%H:%M:%S"


class Settings(BaseModel):
    """The whole application configuration as one typed object."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    detectors: DetectorsConfig = Field(default_factory=DetectorsConfig)
    semantic_analysis: SemanticAnalysisConfig = Field(
        default_factory=SemanticAnalysisConfig
    )
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    project_root: Path = PROJECT_ROOT

    model_config = {"arbitrary_types_allowed": True}

    def resolve_path(self, relative: str) -> Path:
        """Turn a config-relative path into an absolute one."""
        candidate = Path(relative)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate

    def prompt_path(self, filename: str) -> Path:
        return self.project_root / "prompts" / filename


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}\n"
            "Expected config/config.yaml at the project root."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config.yaml must contain a top-level mapping.")
    return data


def _apply_env_overrides(settings: Settings) -> Settings:
    """Secrets and optional quick overrides come from the environment."""
    settings.llm.api_key = os.getenv("LLM_API_KEY")

    if provider := os.getenv("LLM_PROVIDER"):
        settings.llm.provider = provider.strip().lower()
    if model := os.getenv("LLM_MODEL"):
        settings.llm.model = model.strip()
    if base_url := os.getenv("LLM_BASE_URL"):
        settings.llm.base_url = base_url.strip()
    if level := os.getenv("LOG_LEVEL"):
        settings.logging.level = level.strip().upper()
    return settings


def load_settings(config_path: Optional[Path | str] = None) -> Settings:
    """Load configuration from YAML plus ``.env``.

    Args:
        config_path: Override for the YAML location (used by tests).

    Returns:
        A fully populated, validated :class:`Settings` instance.

    Raises:
        ConfigError: if the file is missing, malformed, or internally
            inconsistent (for example scoring weights that do not sum to 1).
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _read_yaml(path)

    try:
        settings = Settings.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"config.yaml failed validation: {exc}") from exc

    settings = _apply_env_overrides(settings)
    _validate_consistency(settings)
    return settings


def _validate_consistency(settings: Settings) -> None:
    """Catch configuration mistakes early rather than mid-analysis."""
    weight_total = settings.scoring.weights.total()
    if abs(weight_total - 1.0) > 0.01:
        raise ConfigError(
            f"scoring.weights must sum to 1.0, got {weight_total:.3f}. "
            "Adjust config/config.yaml."
        )
    if settings.embeddings.backend not in {"spacy", "sentence_transformers"}:
        raise ConfigError(
            f"Unknown embeddings.backend: {settings.embeddings.backend!r}. "
            "Use 'spacy' or 'sentence_transformers'."
        )
    if settings.nlp.max_input_length < settings.nlp.min_input_length:
        raise ConfigError("nlp.max_input_length is below nlp.min_input_length.")

    semantic = settings.semantic_analysis
    if semantic.max_senses_considered < 1:
        raise ConfigError(
            "semantic_analysis.max_senses_considered must be at least 1."
        )
    if not -1.0 <= semantic.min_sense_similarity <= 1.0:
        raise ConfigError(
            "semantic_analysis.min_sense_similarity must lie in [-1, 1]; "
            "it is compared against a cosine similarity."
        )
    if not 0.0 <= semantic.sense_resolution_margin <= 2.0:
        raise ConfigError(
            "semantic_analysis.sense_resolution_margin must lie in [0, 2]; "
            "it is a gap between two cosine similarities."
        )
    if not semantic.context_content_pos:
        raise ConfigError(
            "semantic_analysis.context_content_pos must list at least one "
            "part-of-speech tag, or no context vector can ever be built."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so the YAML is parsed once per process."""
    return load_settings()
