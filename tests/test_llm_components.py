"""Phase 4 component tests: schemas, config, errors, prompts, evidence,
parser, HTTP provider, retry client, cache and factory.

Nothing here touches the network or needs an API key. The HTTP provider is
exercised through ``httpx.MockTransport``, which runs the real provider code
against scripted HTTP responses.
"""

from __future__ import annotations

import json

import httpx
import pytest
import yaml
from pydantic import ValidationError

from ambisense.config import (
    IMPLEMENTED_PROVIDERS,
    PROMPTS_DIR,
    AdjudicationConfig,
    ConfigError,
    LLMConfig,
    load_settings,
)
from ambisense.llm import evidence as ev
from ambisense.llm.cache import ResponseCache
from ambisense.llm.client import LLMClient
from ambisense.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseFormatError,
    LLMServerError,
    LLMTimeoutError,
    redact,
)
from ambisense.llm.factory import PROVIDER_BUILDERS, build_provider
from ambisense.llm.health import check_server
from ambisense.llm.parser import (
    ResponseParseError,
    extract_json_object,
    parse_judgements,
    parse_rewrites,
)
from ambisense.llm.prompt_builder import (
    PromptTemplateError,
    load_prompt,
    parse_prompt_text,
)
from ambisense.llm.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
)
from ambisense.llm.providers.openai_compatible import OpenAICompatibleProvider
from ambisense.schemas import (
    AdjudicationStatus,
    AdjudicationVerdict,
    AmbiguityCandidate,
    AmbiguityType,
    LLMJudgement,
    SemanticAnalysisStatus,
    SenseOption,
    SenseRanking,
    parse_optional_confidence,
)

SECRET = "sk-test-SECRET-value-1234"


def judgement(candidate_id="c1", verdict="genuine_ambiguity", **overrides):
    item = {
        "candidate_id": candidate_id,
        "verdict": verdict,
        "interpretations": [
            {"meaning": "Reading one.", "explanation": "Licensed by A."},
            {"meaning": "Reading two.", "explanation": "Licensed by B."},
        ],
        "explanation": "Both readings are natural.",
        "confidence": 0.8,
        "selected_sense": None,
    }
    item.update(overrides)
    return item


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TestAdjudicationVerdict:
    @pytest.mark.parametrize("raw,expected", [
        ("genuine_ambiguity", AdjudicationVerdict.GENUINE_AMBIGUITY),
        ("GENUINE AMBIGUITY", AdjudicationVerdict.GENUINE_AMBIGUITY),
        ("genuine", AdjudicationVerdict.GENUINE_AMBIGUITY),
        ("not-ambiguous", AdjudicationVerdict.NOT_AMBIGUOUS),
        ("false_positive", AdjudicationVerdict.NOT_AMBIGUOUS),
        ("uncertain", AdjudicationVerdict.UNCERTAIN),
    ])
    def test_known_spellings(self, raw, expected):
        assert AdjudicationVerdict.parse(raw) is expected

    @pytest.mark.parametrize("raw", ["maybe", "", None, "yes", 1])
    def test_unknown_verdict_is_rejected_not_defaulted(self, raw):
        """An unreadable verdict must not silently become 'uncertain'."""
        with pytest.raises(ValueError):
            AdjudicationVerdict.parse(raw)


class TestOptionalConfidence:
    @pytest.mark.parametrize("raw,expected", [
        (0.82, 0.82), ("0.6", 0.6), (85, 0.85), ("90%", 0.9),
        (1.3, 1.0), (-1, 0.0),
    ])
    def test_readable_values(self, raw, expected):
        assert parse_optional_confidence(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "high", "", True, False, float("nan"), [0.5]])
    def test_unreadable_values_become_none_not_a_guess(self, raw):
        """Unlike clamp_confidence, no 0.5 is manufactured."""
        assert parse_optional_confidence(raw) is None


class TestLLMJudgement:
    def test_valid_judgement(self):
        parsed = LLMJudgement.model_validate(judgement())
        assert parsed.verdict is AdjudicationVerdict.GENUINE_AMBIGUITY
        assert len(parsed.interpretations) == 2
        assert parsed.selected_sense is None

    def test_genuine_with_one_interpretation_is_invalid(self):
        with pytest.raises(ValidationError, match="at least two"):
            LLMJudgement.model_validate(
                judgement(interpretations=[{"meaning": "Only one."}])
            )

    def test_duplicate_interpretations_do_not_count_twice(self):
        with pytest.raises(ValidationError, match="at least two"):
            LLMJudgement.model_validate(judgement(interpretations=[
                {"meaning": "Same reading."}, {"meaning": "same reading."},
            ]))

    def test_not_ambiguous_may_have_one_reading(self):
        parsed = LLMJudgement.model_validate(judgement(
            verdict="not_ambiguous", interpretations=[{"meaning": "Only one."}]
        ))
        assert parsed.verdict is AdjudicationVerdict.NOT_AMBIGUOUS

    def test_string_interpretations_are_accepted(self):
        parsed = LLMJudgement.model_validate(
            judgement(interpretations=["First.", "Second."])
        )
        assert [i.meaning for i in parsed.interpretations] == ["First.", "Second."]

    def test_empty_explanation_is_invalid(self):
        with pytest.raises(ValidationError):
            LLMJudgement.model_validate(judgement(explanation="   "))

    def test_missing_verdict_is_invalid(self):
        item = judgement()
        del item["verdict"]
        with pytest.raises(ValidationError):
            LLMJudgement.model_validate(item)

    def test_malformed_confidence_is_kept_as_none(self):
        parsed = LLMJudgement.model_validate(judgement(confidence="very sure"))
        assert parsed.confidence is None

    def test_llm_has_no_agreement_field(self):
        """Agreement with Phase 3 is derived in code, never taken from the LLM."""
        assert "semantic_evidence_agreement" not in LLMJudgement.model_fields
        assert "agrees_with_phase3" not in LLMJudgement.model_fields
        parsed = LLMJudgement.model_validate(
            judgement(semantic_evidence_agreement="agrees")
        )
        assert not hasattr(parsed, "semantic_evidence_agreement")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestPhase4Config:
    def test_shipped_config_loads_with_adjudication_block(self):
        settings = load_settings()
        assert settings.llm.provider in IMPLEMENTED_PROVIDERS == ("ollama",)
        assert settings.llm.temperature == 0.0
        assert settings.adjudication.batch_candidates is True

    def test_batch_size_is_one_when_batching_disabled(self):
        assert AdjudicationConfig(batch_candidates=False).batch_size == 1
        assert AdjudicationConfig(max_candidates_per_request=4).batch_size == 4

    @pytest.mark.parametrize("section,values,match", [
        ("llm", {"provider": "openai"}, "not implemented"),
        ("llm", {"provider": "groq"}, "not implemented"),
        ("llm", {"max_retries": 9}, "max_retries"),
        ("llm", {"temperature": 3.0}, "temperature"),
        ("adjudication", {"max_candidates_per_request": 0}, "max_candidates_per_request"),
        ("adjudication", {"adjudication_prompt": "missing.txt"}, "missing prompt file"),
    ])
    def test_invalid_values_are_rejected_at_startup(self, tmp_path, section, values, match):
        with pytest.raises(ConfigError, match=match):
            load_settings(_write(tmp_path, {section: values}))

    def test_prompt_files_exist(self):
        settings = load_settings()
        for name in ("adjudication_prompt", "rewrite_prompt", "repair_prompt"):
            assert settings.prompt_path(getattr(settings.adjudication, name)).is_file()


# ---------------------------------------------------------------------------
# Errors and redaction
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.parametrize("error_class,retryable", [
        (LLMTimeoutError, True), (LLMRateLimitError, True),
        (LLMServerError, True), (LLMConnectionError, True),
        (LLMAuthenticationError, False), (LLMRequestError, False),
        (LLMConfigurationError, False),
    ])
    def test_retry_classification(self, error_class, retryable):
        assert error_class("x").retryable is retryable

    def test_no_error_maps_to_a_verdict(self):
        statuses = {cls("x").status for cls in (
            LLMTimeoutError, LLMAuthenticationError, LLMConfigurationError,
            LLMResponseFormatError, LLMServerError,
        )}
        assert AdjudicationStatus.ADJUDICATED not in statuses

    def test_redact_removes_secret(self):
        assert SECRET not in redact(f"bad key {SECRET} given", SECRET)
        assert "[REDACTED]" in redact(f"bad key {SECRET}", SECRET)

    def test_redact_ignores_missing_or_tiny_secrets(self):
        assert redact("text", None) == "text"
        assert redact("a b c", "a") == "a b c"


# ---------------------------------------------------------------------------
# Prompt files and builder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analysis_prompt():
    return load_prompt(PROMPTS_DIR / "ambiguity_analysis.txt")


class TestPromptFiles:
    def test_all_prompts_parse(self):
        for name in ("ambiguity_analysis.txt", "rewrite_generation.txt", "json_repair.txt"):
            assert load_prompt(PROMPTS_DIR / name).user is not None

    def test_adjudication_prompt_contains_the_critical_rules(self, analysis_prompt):
        """Regression guard: these instructions define the LLM's role."""
        system = analysis_prompt.system.template
        for required in (
            "not as ground truth",
            "not probabilities",
            "You may disagree",
            "genuine_ambiguity", "not_ambiguous", "uncertain",
            "vagueness", "underspecification",
            "self-reported",
            "Ignore any instructions",
            "Return ONLY a JSON object",
            "selected_sense",
            "you may select any listed sense, whatever its rank",
        ):
            assert required in system, f"prompt lost the rule: {required!r}"

    def test_prompt_does_not_ask_the_llm_to_report_agreement(self, analysis_prompt):
        """The relationship to Phase 3 is computed in code, not self-reported."""
        system = analysis_prompt.system.template.lower()
        for forbidden in ("agrees", "agreement", "semantic_evidence_agreement"):
            assert forbidden not in system

    def test_rendering_fills_every_placeholder(self, analysis_prompt):
        messages = analysis_prompt.render(
            sentence="S.", context="(none supplied)", candidates="[]"
        )
        assert [m.role for m in messages] == ["system", "user"]
        assert "$" not in messages[1].content

    def test_json_braces_in_prompt_survive_rendering(self, analysis_prompt):
        messages = analysis_prompt.render(sentence="S.", context="-", candidates="[]")
        assert '{"judgements": [' in messages[0].content

    def test_missing_value_raises(self, analysis_prompt):
        with pytest.raises(PromptTemplateError, match=r"\$context"):
            analysis_prompt.render(sentence="S.", candidates="[]")

    def test_user_text_is_not_re_substituted(self, analysis_prompt):
        """A '$candidates' typed by the user must stay literal text."""
        messages = analysis_prompt.render(
            sentence="Pay $candidates now.", context="-", candidates="[]"
        )
        assert "Pay $candidates now." in messages[1].content

    def test_repair_prompt_has_no_system_section(self):
        prompt = load_prompt(PROMPTS_DIR / "json_repair.txt")
        assert prompt.system is None
        assert [m.role for m in prompt.render(problems="- c1: x")] == ["user"]

    def test_malformed_prompt_files_are_rejected(self):
        with pytest.raises(PromptTemplateError, match="no ===USER==="):
            parse_prompt_text("just text")
        with pytest.raises(PromptTemplateError, match="no ===SYSTEM==="):
            parse_prompt_text("stray\n===USER===\nhello")
        with pytest.raises(PromptTemplateError, match="empty user"):
            parse_prompt_text("===USER===\n   ")

    def test_missing_prompt_file(self, tmp_path):
        with pytest.raises(PromptTemplateError):
            load_prompt(tmp_path / "nope.txt")


# ---------------------------------------------------------------------------
# Evidence package
# ---------------------------------------------------------------------------


def lexical_candidate():
    return AmbiguityCandidate(
        span_text="bank", char_start=20, char_end=24,
        type_hint=AmbiguityType.LEXICAL, detector_name="lexical", prior=0.83,
        explanation="'bank' has 10 senses.",
        evidence={"rule": "wordnet_polysemy", "detector": "lexical",
                  "lemma": "bank", "sentence_index": 0,
                  "thresholds": {"min_senses": 4},
                  "sample_definitions": ["x"]},
    )


def ranking(char_start=20, char_end=24, senses=6):
    return SenseRanking(
        word="bank", lemma="bank", pos="NOUN",
        status=SemanticAnalysisStatus.RANKED,
        char_start=char_start, char_end=char_end,
        context_words=["deposited", "money"], margin=0.0322,
        resolved_by_context=True,
        senses=[
            SenseOption(sense_key=f"bank.n.0{i}", definition="word " * 80,
                        pos="n", context_similarity=0.9 - i / 10, rank=i)
            for i in range(1, senses + 1)
        ],
    )


class TestEvidence:
    config = AdjudicationConfig()

    def test_ids_are_stable(self):
        assert ev.candidate_ids(3) == ["c1", "c2", "c3"]

    def test_internals_are_withheld(self):
        compact = ev.compact_rule_evidence(lexical_candidate().evidence)
        assert compact == {"rule": "wordnet_polysemy", "lemma": "bank"}

    def test_semantic_evidence_is_capped_and_truncated(self):
        payload = ev.semantic_payload(ranking(senses=6), self.config)
        assert len(payload["ranked_senses"]) == self.config.max_senses_in_prompt
        assert all(len(s["gloss"]) <= self.config.max_gloss_chars
                   for s in payload["ranked_senses"])
        assert "not a probability" in payload["measure"]

    def test_anchoring_fields_are_not_sent(self):
        """prior and resolved_by_context are withheld on purpose."""
        payload = ev.candidate_payload("c1", lexical_candidate(), ranking(), self.config)
        text = ev.to_json(payload)
        assert '"prior"' not in text
        assert "resolved_by_context" not in text
        assert "_index" not in text
        assert '"margin_top_two": 0.0322' in text

    def test_unusable_ranking_sends_status_only(self):
        bad = SenseRanking(word="x", lemma="x", pos="NOUN",
                           status=SemanticAnalysisStatus.NO_CONTEXT_VECTOR)
        assert ev.semantic_payload(bad, self.config) == {"status": "no_usable_context_vector"}

    def test_zero_sense_budget_sends_nothing(self):
        config = AdjudicationConfig(max_senses_in_prompt=0)
        assert ev.semantic_payload(ranking(), config) is None

    def test_ranking_matches_by_offsets_for_lexical_only(self):
        candidate = lexical_candidate()
        assert ev.match_ranking(candidate, [ranking(0, 4), ranking()]) is not None
        assert ev.match_ranking(candidate, [ranking(0, 4)]) is None
        syntactic = candidate.model_copy(update={"type_hint": AmbiguityType.SYNTACTIC})
        assert ev.match_ranking(syntactic, [ranking()]) is None

    def test_serialisation_is_deterministic(self):
        a = ev.to_json(ev.candidate_payload("c1", lexical_candidate(), ranking(), self.config))
        b = ev.to_json(ev.candidate_payload("c1", lexical_candidate(), ranking(), self.config))
        assert a == b


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_markdown_fences(self):
        assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_surrounding_prose(self):
        assert extract_json_object('Here you go: {"a": 1} Thanks!') == {"a": 1}

    @pytest.mark.parametrize("bad,match", [
        ("", "empty"), ("no json here", "no JSON object"),
        ('{"a": 1,,}', "not valid JSON"), ("[1, 2]", "not a JSON object"),
    ])
    def test_unparseable(self, bad, match):
        with pytest.raises(ResponseParseError, match=match):
            extract_json_object(bad)


class TestParseJudgements:
    def test_all_valid(self):
        text = json.dumps({"judgements": [judgement("c1"), judgement("c2")]})
        result = parse_judgements(text, ["c1", "c2"])
        assert result.complete and set(result.valid) == {"c1", "c2"}

    def test_invalid_json_marks_every_candidate(self):
        result = parse_judgements("not json", ["c1", "c2"])
        assert not result.valid and set(result.problems) == {"c1", "c2"}

    def test_missing_list_key(self):
        result = parse_judgements('{"verdicts": []}', ["c1"])
        assert "no 'judgements' list" in result.problems["c1"]

    def test_one_bad_item_does_not_discard_the_good_one(self):
        text = json.dumps({"judgements": [judgement("c1"), judgement("c2", verdict="perhaps")]})
        result = parse_judgements(text, ["c1", "c2"])
        assert "c1" in result.valid
        assert "unrecognised verdict" in result.problems["c2"]

    def test_missing_candidate_is_reported(self):
        result = parse_judgements(json.dumps({"judgements": [judgement("c1")]}), ["c1", "c2"])
        assert result.problems == {"c2": "no judgement was returned"}

    def test_unknown_ids_are_ignored(self):
        text = json.dumps({"judgements": [judgement("c1"), judgement("c99")]})
        assert set(parse_judgements(text, ["c1"]).valid) == {"c1"}

    def test_duplicate_ids_first_valid_wins(self):
        text = json.dumps({"judgements": [
            judgement("c1", verdict="bogus"),
            judgement("c1", verdict="not_ambiguous"),
            judgement("c1", verdict="uncertain"),
        ]})
        result = parse_judgements(text, ["c1"])
        assert result.valid["c1"].verdict is AdjudicationVerdict.NOT_AMBIGUOUS

    def test_problem_summary_is_readable_and_input_free(self):
        """A repairable defect (here: the candidate never arrived) formats
        cleanly for the repair prompt. See TestRepairableVsFinalProblems for
        the distinction between what is and is not sent back for repair."""
        text = json.dumps({"judgements": []})
        summary = parse_judgements(text, ["c1"]).problem_summary()
        assert summary.startswith("- c1:")

    def test_unrecognised_verdict_is_a_problem_but_not_in_the_repair_summary(self):
        """A contract violation is reported, but never sent back for repair."""
        text = json.dumps({"judgements": [judgement("c1", verdict="zzz-not-a-verdict")]})
        result = parse_judgements(text, ["c1"])
        assert "unrecognised verdict" in result.problems["c1"]
        assert "c1" not in result.repairable
        assert result.problem_summary() == ""


class TestParseRewrites:
    original = "I saw the man with the telescope."

    def test_valid(self):
        text = json.dumps({"rewrites": [{"candidate_id": "c1", "rewrites": ["A.", "B."]}]})
        assert parse_rewrites(text, ["c1"], self.original).valid["c1"] == ["A.", "B."]

    def test_rewrite_identical_to_original_is_dropped(self):
        text = json.dumps({"rewrites": [{"candidate_id": "c1",
                                         "rewrites": [self.original, "  Different. "]}]})
        assert parse_rewrites(text, ["c1"], self.original).valid["c1"] == ["Different."]

    def test_only_identical_rewrites_is_a_problem(self):
        text = json.dumps({"rewrites": [{"candidate_id": "c1", "rewrites": [self.original]}]})
        result = parse_rewrites(text, ["c1"], self.original)
        assert "c1" in result.problems

    def test_non_list_rewrites(self):
        text = json.dumps({"rewrites": [{"candidate_id": "c1", "rewrites": "A."}]})
        assert "must be a list" in parse_rewrites(text, ["c1"], self.original).problems["c1"]


# ---------------------------------------------------------------------------
# HTTP provider, via httpx.MockTransport (real code, no network)
# ---------------------------------------------------------------------------


def make_provider(handler, **config):
    llm_config = LLMConfig(base_url="https://api.test/v1", **config)
    return OpenAICompatibleProvider(
        llm_config, SECRET, transport=httpx.MockTransport(handler)
    )


def ok_body(content='{"judgements": []}', finish="stop"):
    return {
        "model": "llama-test",
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40},
    }


REQUEST = LLMRequest(messages=(ChatMessage("system", "sys"), ChatMessage("user", "JSON please")))


class TestOpenAICompatibleProvider:
    def test_satisfies_the_provider_protocol(self):
        assert isinstance(make_provider(lambda r: httpx.Response(200, json=ok_body())), LLMProvider)

    def test_request_shape(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        make_provider(handler, model="m-1", temperature=0.0).complete(REQUEST)
        assert seen["url"] == "https://api.test/v1/chat/completions"
        assert seen["auth"] == f"Bearer {SECRET}"
        assert seen["body"]["model"] == "m-1"
        assert seen["body"]["temperature"] == 0.0
        assert seen["body"]["response_format"] == {"type": "json_object"}
        assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]

    def test_secret_only_in_the_auth_header(self):
        seen = {}

        def handler(request):
            seen["body"] = request.content.decode()
            return httpx.Response(200, json=ok_body())

        make_provider(handler).complete(REQUEST)
        assert SECRET not in seen["body"]

    def test_json_mode_can_be_disabled(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        make_provider(handler, request_json_mode=False).complete(REQUEST)
        assert "response_format" not in seen["body"]

    def test_success_parses_content_and_usage(self):
        response = make_provider(lambda r: httpx.Response(200, json=ok_body("{}"))).complete(REQUEST)
        assert response.content == "{}"
        assert (response.prompt_tokens, response.completion_tokens) == (120, 40)

    @pytest.mark.parametrize("status,error_class,retryable", [
        (401, LLMAuthenticationError, False),
        (403, LLMAuthenticationError, False),
        (404, LLMRequestError, False),
        (429, LLMRateLimitError, True),
        (500, LLMServerError, True),
        (503, LLMServerError, True),
    ])
    def test_http_errors_are_classified(self, status, error_class, retryable):
        provider = make_provider(lambda r: httpx.Response(status, json={"error": {"message": "x"}}))
        with pytest.raises(error_class) as info:
            provider.complete(REQUEST)
        assert info.value.retryable is retryable

    def test_retry_after_header_is_read(self):
        provider = make_provider(lambda r: httpx.Response(
            429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}
        ))
        with pytest.raises(LLMRateLimitError) as info:
            provider.complete(REQUEST)
        assert info.value.retry_after == 7.0

    def test_timeout(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)
        with pytest.raises(LLMTimeoutError):
            make_provider(handler).complete(REQUEST)

    def test_network_failure(self):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)
        with pytest.raises(LLMConnectionError, match="ollama serve"):
            make_provider(handler).complete(REQUEST)

    def test_no_key_means_no_authorization_header(self):
        """A local Ollama server needs no key, so none is sent."""
        seen = {}

        def handler(request):
            seen["headers"] = dict(request.headers)
            return httpx.Response(200, json=ok_body())

        provider = OpenAICompatibleProvider(
            LLMConfig(base_url="https://api.test/v1"), None,
            transport=httpx.MockTransport(handler),
        )
        provider.complete(REQUEST)
        assert "authorization" not in seen["headers"]
        assert provider.name == "ollama"

    def test_missing_model_gives_an_ollama_pull_hint(self):
        provider = make_provider(
            lambda r: httpx.Response(404, json={"error": {"message": "model 'm-1' not found"}}),
            model="m-1",
        )
        with pytest.raises(LLMRequestError, match="ollama pull m-1") as info:
            provider.complete(REQUEST)
        assert info.value.retryable is False

    def test_reasoning_field_is_ignored_and_content_used(self):
        """qwen3 returns its thinking in a separate 'reasoning' field."""
        body = ok_body('{"judgements": []}')
        body["choices"][0]["message"]["reasoning"] = "Let me think {about braces}..."
        response = make_provider(lambda r: httpx.Response(200, json=body)).complete(REQUEST)
        assert response.content == '{"judgements": []}'

    def test_timeout_message_mentions_local_inference(self):
        def handler(request):
            raise httpx.ReadTimeout("slow", request=request)
        with pytest.raises(LLMTimeoutError, match="timeout_seconds"):
            make_provider(handler).complete(REQUEST)

    @pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": {}}]}])
    def test_malformed_envelope(self, body):
        with pytest.raises(LLMResponseFormatError):
            make_provider(lambda r: httpx.Response(200, json=body)).complete(REQUEST)

    def test_non_json_body(self):
        with pytest.raises(LLMResponseFormatError):
            make_provider(lambda r: httpx.Response(200, text="<html>")).complete(REQUEST)

    def test_empty_content_is_retryable(self):
        with pytest.raises(LLMResponseFormatError) as info:
            make_provider(lambda r: httpx.Response(200, json=ok_body("  "))).complete(REQUEST)
        assert info.value.retryable is True

    def test_truncated_reply_is_not_retryable(self):
        """Hitting max_tokens would truncate identically on a retry."""
        with pytest.raises(LLMResponseFormatError) as info:
            make_provider(lambda r: httpx.Response(200, json=ok_body("{", "length"))).complete(REQUEST)
        assert info.value.retryable is False

    def test_secret_echoed_by_server_is_redacted(self):
        provider = make_provider(lambda r: httpx.Response(
            400, json={"error": {"message": f"key {SECRET} is malformed"}}
        ))
        with pytest.raises(LLMRequestError) as info:
            provider.complete(REQUEST)
        assert SECRET not in str(info.value)


# ---------------------------------------------------------------------------
# Retry client
# ---------------------------------------------------------------------------


class ScriptedProvider:
    name, model = "scripted", "scripted-model"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestRetryClient:
    def client(self, outcomes, **config):
        waits = []
        provider = ScriptedProvider(outcomes)
        client = LLMClient(provider, LLMConfig(**config), sleep=waits.append)
        return client, provider, waits

    def test_success_first_time(self):
        client, provider, waits = self.client([LLMResponse("ok")])
        assert client.complete(REQUEST).attempts == 1
        assert waits == []

    def test_transient_errors_are_retried_with_exponential_backoff(self):
        client, provider, waits = self.client(
            [LLMTimeoutError("t"), LLMServerError("s"), LLMResponse("ok")],
            max_retries=2, retry_backoff_seconds=2.0,
        )
        outcome = client.complete(REQUEST)
        assert outcome.attempts == 3 and outcome.response.content == "ok"
        assert waits == [2.0, 4.0]

    def test_retries_are_bounded(self):
        client, provider, waits = self.client(
            [LLMTimeoutError("t")] * 10, max_retries=2
        )
        with pytest.raises(LLMTimeoutError) as info:
            client.complete(REQUEST)
        assert provider.calls == 3
        assert info.value.attempts == 3

    def test_permanent_errors_are_not_retried(self):
        client, provider, waits = self.client(
            [LLMAuthenticationError("bad key"), LLMResponse("never")], max_retries=2
        )
        with pytest.raises(LLMAuthenticationError):
            client.complete(REQUEST)
        assert provider.calls == 1 and waits == []

    def test_retry_after_is_honoured_but_capped(self):
        client, _, waits = self.client(
            [LLMRateLimitError("r", retry_after=500), LLMResponse("ok")],
            max_retries=1, max_retry_wait_seconds=30,
        )
        client.complete(REQUEST)
        assert waits == [30]

    def test_zero_retries(self):
        client, provider, _ = self.client([LLMTimeoutError("t")], max_retries=0)
        with pytest.raises(LLMTimeoutError):
            client.complete(REQUEST)
        assert provider.calls == 1


# ---------------------------------------------------------------------------
# Cache and factory
# ---------------------------------------------------------------------------


class TestCache:
    def test_round_trip(self, tmp_path):
        cache = ResponseCache(tmp_path)
        cache.put("k", '{"x": 1}')
        assert cache.get("k") == '{"x": 1}'

    def test_miss_and_corrupt_entry(self, tmp_path):
        cache = ResponseCache(tmp_path)
        assert cache.get("absent") is None
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        assert cache.get("bad") is None

    def test_key_changes_with_prompt_text_and_settings(self):
        config = LLMConfig()
        base = ResponseCache.key_for(REQUEST, config, "ollama")
        other_prompt = LLMRequest(messages=(ChatMessage("user", "different"),))
        assert ResponseCache.key_for(other_prompt, config, "ollama") != base
        assert ResponseCache.key_for(REQUEST, LLMConfig(model="other"), "ollama") != base
        assert ResponseCache.key_for(REQUEST, config, "ollama") == base

    def test_key_does_not_depend_on_the_api_key(self):
        assert ResponseCache.key_for(REQUEST, LLMConfig(api_key="a" * 10), "ollama") == \
               ResponseCache.key_for(REQUEST, LLMConfig(api_key="b" * 10), "ollama")

    def test_unwritable_directory_does_not_raise(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        ResponseCache(blocker / "sub").put("k", "v")  # logs, does not raise


class TestFactory:
    def test_registered_providers_match_config(self):
        assert set(PROVIDER_BUILDERS) == set(IMPLEMENTED_PROVIDERS)

    def test_ollama_needs_no_api_key(self):
        provider = build_provider(LLMConfig(api_key=None))
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.name == "ollama"

    def test_optional_key_is_passed_through_for_proxied_servers(self):
        assert LLMConfig(api_key=f"  {SECRET} ").auth_token == SECRET
        assert LLMConfig(api_key="   ").auth_token is None
        assert LLMConfig(api_key=None).auth_token is None

    def test_shipped_default_is_a_local_ollama_server(self):
        config = load_settings().llm
        assert config.provider == "ollama"
        assert config.base_url.startswith("http://localhost:11434")

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMConfigurationError):
            build_provider(LLMConfig(api_key=SECRET, provider="nonexistent"))


# ---------------------------------------------------------------------------
# Ollama server health check
# ---------------------------------------------------------------------------


class TestServerHealth:
    def tags(self, *names):
        return httpx.MockTransport(lambda request: httpx.Response(
            200, json={"models": [{"name": name} for name in names]}
        ))

    def test_ready_when_model_is_installed(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:latest"}]})

        status = check_server(LLMConfig(model="qwen2.5:latest"),
                              transport=httpx.MockTransport(handler))
        assert status.ready
        assert seen["url"] == "http://localhost:11434/api/tags"

    def test_reachable_but_model_missing(self):
        status = check_server(LLMConfig(model="qwen3:14b"),
                              transport=self.tags("llama3.1:8b"))
        assert status.reachable and not status.model_installed
        assert not status.ready

    def test_unreachable_server(self):
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        status = check_server(LLMConfig(), transport=httpx.MockTransport(handler))
        assert not status.reachable and not status.ready
        assert "is Ollama running" in status.error

    def test_server_error_counts_as_unreachable(self):
        status = check_server(LLMConfig(), transport=httpx.MockTransport(
            lambda request: httpx.Response(500, text="boom")
        ))
        assert not status.reachable
