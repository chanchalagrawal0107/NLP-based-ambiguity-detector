"""Tests for configuration loading and validation.

These guard two things:

1. The shipped ``config/config.yaml`` actually loads and satisfies every
   consistency rule - a broken config file would break every other layer.
2. ``_validate_consistency`` catches the *semantic* mistakes that YAML parsing
   and type checking cannot, such as weights that are individually valid
   floats but collectively wrong.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from ambisense.config import (
    ConfigError,
    DetectorsConfig,
    DEFAULT_CONFIG_PATH,
    load_settings,
)


@pytest.fixture(scope="module")
def shipped_settings():
    """The real config/config.yaml, as the application loads it."""
    return load_settings()


def _write_config(tmp_path, data: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _minimal_valid() -> dict:
    """The smallest config that passes every consistency rule."""
    return {
        "scoring": {
            "weights": {
                "llm_confidence": 0.4,
                "detector_evidence": 0.3,
                "interpretation_count": 0.2,
                "context_uncertainty": 0.1,
            }
        },
        "embeddings": {"backend": "spacy"},
        "nlp": {"min_input_length": 3, "max_input_length": 100},
    }


class TestShippedConfig:
    def test_it_loads(self, shipped_settings):
        assert shipped_settings.nlp.spacy_model == "en_core_web_md"

    def test_weights_sum_to_one(self, shipped_settings):
        assert shipped_settings.scoring.weights.total() == pytest.approx(1.0)

    def test_every_detector_has_a_settings_block(self, shipped_settings):
        """Phase 2 fix: `semantic` was enabled but had no settings block."""
        for name in DetectorsConfig.DETECTOR_NAMES:
            settings = shipped_settings.detectors.settings_for(name)
            assert settings, f"{name}_settings is missing or empty"

    def test_semantic_settings_are_populated(self, shipped_settings):
        semantic = shipped_settings.detectors.settings_for("semantic")
        assert "tough_adjectives" in semantic
        assert "ready" in semantic["tough_adjectives"]
        assert "intransitive_verbs" in semantic

    def test_enabled_names_are_ordered_and_deterministic(self, shipped_settings):
        first = shipped_settings.detectors.enabled_names()
        second = shipped_settings.detectors.enabled_names()
        assert first == second
        assert first == list(DetectorsConfig.DETECTOR_NAMES)

    def test_demonstratives_excluded_from_pronoun_list(self, shipped_settings):
        """spaCy tags 'that' as SCONJ in 'told David that he was late'."""
        referential = shipped_settings.detectors.settings_for("referential")
        assert "that" not in referential["pronouns"]
        assert "this" not in referential["pronouns"]


class TestDetectorsConfigAccessors:
    def test_settings_for_missing_block_returns_empty_dict(self):
        config = DetectorsConfig()
        assert config.settings_for("lexical") == {}
        assert config.settings_for("nonexistent") == {}

    def test_is_enabled_for_unknown_detector_is_false(self):
        assert DetectorsConfig().is_enabled("telepathic") is False

    def test_disabling_a_detector_removes_it_from_enabled_names(self):
        config = DetectorsConfig(lexical=False, scope=False)
        names = config.enabled_names()
        assert "lexical" not in names and "scope" not in names
        assert "syntactic" in names


class TestConsistencyValidation:
    def test_weights_not_summing_to_one_is_rejected(self, tmp_path):
        data = _minimal_valid()
        data["scoring"]["weights"]["llm_confidence"] = 0.9
        with pytest.raises(ConfigError, match="must sum to 1.0"):
            load_settings(_write_config(tmp_path, data))

    def test_unknown_embeddings_backend_is_rejected(self, tmp_path):
        data = _minimal_valid()
        data["embeddings"]["backend"] = "telepathy"
        with pytest.raises(ConfigError, match="embeddings.backend"):
            load_settings(_write_config(tmp_path, data))

    def test_inverted_length_bounds_are_rejected(self, tmp_path):
        data = _minimal_valid()
        data["nlp"] = {"min_input_length": 500, "max_input_length": 10}
        with pytest.raises(ConfigError, match="max_input_length"):
            load_settings(_write_config(tmp_path, data))

    def test_minimal_valid_config_loads(self, tmp_path):
        settings = load_settings(_write_config(tmp_path, _minimal_valid()))
        assert settings.scoring.weights.total() == pytest.approx(1.0)

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_settings(tmp_path / "does_not_exist.yaml")

    def test_malformed_yaml_is_reported_clearly(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("llm:\n  model: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_settings(path)

    def test_non_mapping_yaml_is_rejected(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="top-level mapping"):
            load_settings(path)


class TestSecretsHandling:
    def test_api_key_never_appears_in_serialised_config(self, shipped_settings):
        """The key is excluded from dumps so it cannot leak into logs."""
        shipped_settings.llm.api_key = "sk-secret-value"
        assert "sk-secret-value" not in shipped_settings.model_dump_json()

    def test_yaml_file_contains_no_api_key(self):
        raw = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        assert "api_key" not in raw
