"""Phase 4 adjudicator tests with fake providers.

These exercise the orchestration logic - batching, failure mapping, repair,
the rewrite step and caching - without any network access or API key. A final
group runs the real Phase 1-3 pipeline into the adjudicator (still with a fake
provider) to prove the evidence package the LLM would receive is correct.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ambisense.config import (
    PROJECT_ROOT,
    AdjudicationConfig,
    LLMConfig,
    load_settings,
)
from ambisense.cli.render import render_adjudication
from ambisense.llm.adjudicator import LLMAdjudicator
from ambisense.llm.cache import ResponseCache
from ambisense.llm.errors import (
    LLMAuthenticationError,
    LLMRequestError,
    LLMTimeoutError,
)
from ambisense.llm.providers.base import LLMResponse
from ambisense.pipeline import gather_evidence
from ambisense.schemas import (
    AdjudicationStatus,
    AdjudicationVerdict,
    AmbiguityCandidate,
    AmbiguityType,
    AnalysisMode,
    RewriteStatus,
    SemanticAnalysisStatus,
    SenseOption,
    SenseRanking,
)

SECRET = "sk-live-DO-NOT-LEAK-987654321"
TEXT = "Alpha beta gamma delta."
_ID_RE = re.compile(r'"candidate_id": "(c\d+)"')


# ---------------------------------------------------------------------------
# Fakes and builders
# ---------------------------------------------------------------------------


def judgement(candidate_id, verdict="genuine_ambiguity", **overrides):
    item = {
        "candidate_id": candidate_id,
        "verdict": verdict,
        "interpretations": [{"meaning": f"{candidate_id} reading one."},
                            {"meaning": f"{candidate_id} reading two."}],
        "explanation": "Justification.",
        "confidence": 0.75,
        "selected_sense": None,
    }
    if verdict != "genuine_ambiguity":
        item["interpretations"] = [{"meaning": f"{candidate_id} only reading."}]
    item.update(overrides)
    return item


def reply(body, prompt_tokens=100, completion_tokens=40):
    return LLMResponse(
        content=body if isinstance(body, str) else json.dumps(body),
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )


def is_rewrite_request(request):
    return "CONFIRMED AMBIGUITIES" in request.messages[-1].content or any(
        "CONFIRMED AMBIGUITIES" in m.content for m in request.messages
    )


class RecordingProvider:
    """Scripted outcomes; records every request it receives."""

    name, model = "fake", "fake-model"

    def __init__(self, outcomes=(), responder=None):
        self.outcomes = list(outcomes)
        self.responder = responder
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if self.responder is not None:
            outcome = self.responder(request)
        else:
            outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def adjudication_requests(self):
        return [r for r in self.requests if not is_rewrite_request(r)]

    @property
    def rewrite_requests(self):
        return [r for r in self.requests if is_rewrite_request(r)]


def verdict_responder(verdicts: dict[str, str]):
    """Answer adjudication and rewrite requests for whatever ids are sent."""
    def respond(request):
        ids = _ID_RE.findall(request.messages[-1].content)
        if is_rewrite_request(request):
            return reply({"rewrites": [
                {"candidate_id": i, "rewrites": [f"{i} rewrite A.", f"{i} rewrite B."]}
                for i in ids
            ]})
        return reply({"judgements": [
            judgement(i, verdicts.get(i, "genuine_ambiguity")) for i in ids
        ]})
    return respond


def candidate(start, span="word", type_hint=AmbiguityType.SYNTACTIC):
    return AmbiguityCandidate(
        span_text=span, char_start=start, char_end=start + len(span),
        type_hint=type_hint, detector_name=type_hint.value, prior=0.7,
        explanation=f"Rule fired on {span}.",
        evidence={"rule": "test_rule", "token_index": 3},
    )


def adjudicator(provider, cache=None, sleep=None, llm=None, **config):
    llm = llm or LLMConfig(api_key=SECRET, max_retries=1)
    return LLMAdjudicator(
        llm, AdjudicationConfig(**config), provider,
        cache=cache, sleep=sleep or (lambda seconds: None),
    )


# ---------------------------------------------------------------------------
# No provider / disabled / nothing to do
# ---------------------------------------------------------------------------


class TestNothingSent:
    def test_no_provider_marks_every_candidate_not_configured(self):
        report = adjudicator(None).adjudicate(TEXT, None, [candidate(0), candidate(6)])
        assert {a.status for a in report.adjudications} == {
            AdjudicationStatus.LLM_NOT_CONFIGURED
        }
        assert all(a.judgement is None for a in report.adjudications)
        assert report.diagnostics.mode is AnalysisMode.DEGRADED
        assert report.diagnostics.llm_used is False

    def test_disabled_adjudication_sends_nothing(self):
        provider = RecordingProvider(responder=verdict_responder({}))
        report = adjudicator(provider, enabled=False).adjudicate(TEXT, None, [candidate(0)])
        assert provider.requests == []
        assert "disabled" in report.adjudications[0].error

    def test_no_candidates_means_no_api_call(self):
        provider = RecordingProvider(responder=verdict_responder({}))
        report = adjudicator(provider).adjudicate(TEXT, None, [])
        assert provider.requests == []
        assert report.adjudications == []


# ---------------------------------------------------------------------------
# Happy path, batching and rewrites
# ---------------------------------------------------------------------------


class TestAdjudication:
    def test_batched_by_default(self):
        provider = RecordingProvider(responder=verdict_responder({"c2": "not_ambiguous"}))
        report = adjudicator(provider).adjudicate(
            TEXT, None, [candidate(0), candidate(6), candidate(11)]
        )
        assert len(provider.adjudication_requests) == 1
        assert [a.status for a in report.adjudications] == [AdjudicationStatus.ADJUDICATED] * 3
        assert report.adjudications[1].judgement.verdict is AdjudicationVerdict.NOT_AMBIGUOUS
        assert report.diagnostics.mode is AnalysisMode.FULL
        assert report.diagnostics.candidates_confirmed == 2

    def test_per_candidate_mode_makes_one_call_each(self):
        provider = RecordingProvider(responder=verdict_responder({}))
        adjudicator(provider, batch_candidates=False, generate_rewrites=False).adjudicate(
            TEXT, None, [candidate(0), candidate(6), candidate(11)]
        )
        assert len(provider.adjudication_requests) == 3

    def test_batches_respect_the_size_limit(self):
        provider = RecordingProvider(responder=verdict_responder({}))
        adjudicator(provider, max_candidates_per_request=2, generate_rewrites=False).adjudicate(
            TEXT, None, [candidate(i * 6) for i in range(5)]
        )
        assert len(provider.adjudication_requests) == 3

    def test_rewrites_are_requested_only_for_genuine_candidates(self):
        provider = RecordingProvider(responder=verdict_responder(
            {"c1": "not_ambiguous", "c3": "uncertain"}
        ))
        report = adjudicator(provider).adjudicate(
            TEXT, None, [candidate(0), candidate(6), candidate(11)]
        )
        assert len(provider.rewrite_requests) == 1
        assert _ID_RE.findall(provider.rewrite_requests[0].messages[-1].content) == ["c2"]
        by_id = {a.candidate_id: a for a in report.adjudications}
        assert by_id["c2"].rewrite_status is RewriteStatus.GENERATED
        assert by_id["c2"].rewrites == ["c2 rewrite A.", "c2 rewrite B."]
        assert by_id["c1"].rewrite_status is RewriteStatus.NOT_APPLICABLE
        assert by_id["c1"].rewrites == []

    def test_no_genuine_candidates_means_no_rewrite_call(self):
        provider = RecordingProvider(responder=verdict_responder({"c1": "not_ambiguous"}))
        adjudicator(provider).adjudicate(TEXT, None, [candidate(0)])
        assert provider.rewrite_requests == []

    def test_rewrite_failure_never_changes_the_verdict(self):
        def respond(request):
            if is_rewrite_request(request):
                return LLMAuthenticationError("rewrite call rejected")
            return verdict_responder({})(request)

        report = adjudicator(RecordingProvider(responder=respond)).adjudicate(
            TEXT, None, [candidate(0)]
        )
        item = report.adjudications[0]
        assert item.status is AdjudicationStatus.ADJUDICATED
        assert item.is_genuine
        assert item.rewrite_status is RewriteStatus.LLM_UNAVAILABLE

    def test_invalid_rewrite_reply_is_reported(self):
        def respond(request):
            if is_rewrite_request(request):
                return reply({"rewrites": [{"candidate_id": "c1", "rewrites": [TEXT]}]})
            return verdict_responder({})(request)

        report = adjudicator(RecordingProvider(responder=respond), enable_repair=False).adjudicate(
            TEXT, None, [candidate(0)]
        )
        assert report.adjudications[0].rewrite_status is RewriteStatus.LLM_INVALID_RESPONSE

    def test_rewrites_can_be_disabled(self):
        provider = RecordingProvider(responder=verdict_responder({}))
        adjudicator(provider, generate_rewrites=False).adjudicate(TEXT, None, [candidate(0)])
        assert provider.rewrite_requests == []

    def test_token_usage_is_summed_from_provider_reports(self):
        provider = RecordingProvider(responder=verdict_responder({}))
        report = adjudicator(provider).adjudicate(TEXT, None, [candidate(0)])
        assert report.diagnostics.llm_prompt_tokens == 200
        assert report.diagnostics.llm_completion_tokens == 80

    def test_unreported_tokens_stay_none(self):
        provider = RecordingProvider([LLMResponse(json.dumps({"judgements": [
            judgement("c1", "not_ambiguous")]}))])
        report = adjudicator(provider).adjudicate(TEXT, None, [candidate(0)])
        assert report.diagnostics.llm_prompt_tokens is None


# ---------------------------------------------------------------------------
# Failures are explicit, never a verdict
# ---------------------------------------------------------------------------


class TestFailures:
    def test_timeout_after_retries_is_unavailable_not_unambiguous(self):
        waits = []
        provider = RecordingProvider([LLMTimeoutError("slow")] * 5)
        report = adjudicator(provider, sleep=waits.append).adjudicate(
            TEXT, None, [candidate(0), candidate(6)]
        )
        for item in report.adjudications:
            assert item.status is AdjudicationStatus.LLM_UNAVAILABLE
            assert item.judgement is None
        assert len(provider.requests) == 2  # 1 try + max_retries=1
        assert len(waits) == 1
        assert report.diagnostics.llm_attempts == 2

    def test_authentication_failure_is_not_retried(self):
        provider = RecordingProvider([LLMAuthenticationError("bad key")])
        report = adjudicator(provider).adjudicate(TEXT, None, [candidate(0)])
        assert len(provider.requests) == 1
        assert report.adjudications[0].status is AdjudicationStatus.LLM_UNAVAILABLE

    def test_secret_in_an_error_is_redacted(self):
        provider = RecordingProvider([LLMRequestError(f"key {SECRET} refused")])
        report = adjudicator(provider).adjudicate(TEXT, None, [candidate(0)])
        assert SECRET not in report.adjudications[0].error
        assert SECRET not in report.model_dump_json()

    def test_invalid_reply_without_repair_is_invalid_response(self):
        provider = RecordingProvider([reply("this is not json")])
        report = adjudicator(provider, enable_repair=False).adjudicate(TEXT, None, [candidate(0)])
        item = report.adjudications[0]
        assert item.status is AdjudicationStatus.LLM_INVALID_RESPONSE
        assert item.judgement is None
        assert len(provider.requests) == 1

    def test_a_failed_batch_does_not_affect_another_batch(self):
        provider = RecordingProvider([
            LLMAuthenticationError("first batch fails"),
            reply({"judgements": [judgement("c2", "not_ambiguous")]}),
        ])
        report = adjudicator(provider, batch_candidates=False).adjudicate(
            TEXT, None, [candidate(0), candidate(6)]
        )
        assert report.adjudications[0].status is AdjudicationStatus.LLM_UNAVAILABLE
        assert report.adjudications[1].status is AdjudicationStatus.ADJUDICATED


# ---------------------------------------------------------------------------
# Repair round-trip
# ---------------------------------------------------------------------------


class TestRepair:
    def test_invalid_reply_is_repaired_once(self):
        provider = RecordingProvider([
            reply("Sure! Here is my answer, no JSON."),
            reply({"judgements": [judgement("c1", "not_ambiguous")]}),
        ])
        report = adjudicator(provider, generate_rewrites=False).adjudicate(
            TEXT, None, [candidate(0)]
        )
        assert report.adjudications[0].status is AdjudicationStatus.ADJUDICATED
        assert report.diagnostics.repair_attempted is True
        repair = provider.requests[1]
        assert [m.role for m in repair.messages][-2:] == ["assistant", "user"]
        assert repair.messages[-2].content == "Sure! Here is my answer, no JSON."
        assert "c1:" in repair.messages[-1].content

    def test_repair_merges_with_the_valid_part_of_the_first_reply(self):
        """c2 is missing from the first reply entirely - a format defect,
        so it is repairable, unlike a contract violation (see TestNoRepairForContractViolations)."""
        provider = RecordingProvider([
            reply({"judgements": [judgement("c1")]}),
            reply({"judgements": [judgement("c2", "not_ambiguous")]}),
        ])
        report = adjudicator(provider, generate_rewrites=False).adjudicate(
            TEXT, None, [candidate(0), candidate(6)]
        )
        assert [a.status for a in report.adjudications] == [AdjudicationStatus.ADJUDICATED] * 2
        assert report.adjudications[0].is_genuine
        assert report.diagnostics.repair_attempted is True

    def test_repair_that_still_fails_leaves_only_that_candidate_invalid(self):
        provider = RecordingProvider([
            reply({"judgements": [judgement("c1")]}),  # c2 missing -> repairable
            reply({"judgements": [judgement("c2", verdict="still-wrong")]}),
        ])
        report = adjudicator(provider, generate_rewrites=False).adjudicate(
            TEXT, None, [candidate(0), candidate(6)]
        )
        assert report.adjudications[0].status is AdjudicationStatus.ADJUDICATED
        assert report.adjudications[1].status is AdjudicationStatus.LLM_INVALID_RESPONSE
        assert "unrecognised verdict" in report.adjudications[1].error
        assert len(provider.requests) == 2  # exactly one repair, never a loop


class TestNoRepairForContractViolations:
    """Principle: repairs fix format defects, never re-litigate a bad answer."""

    def test_unknown_verdict_is_not_sent_back_for_repair(self):
        provider = RecordingProvider([
            reply({"judgements": [judgement("c1"), judgement("c2", verdict="perhaps")]}),
        ])
        report = adjudicator(provider, generate_rewrites=False).adjudicate(
            TEXT, None, [candidate(0), candidate(6)]
        )
        assert report.adjudications[0].status is AdjudicationStatus.ADJUDICATED
        assert report.adjudications[1].status is AdjudicationStatus.LLM_INVALID_RESPONSE
        assert "unrecognised verdict" in report.adjudications[1].error
        assert len(provider.requests) == 1  # no repair round-trip at all
        assert report.diagnostics.repair_attempted is False

    def test_selected_sense_not_offered_is_not_sent_back_for_repair(self):
        provider = RecordingProvider([
            reply({"judgements": [
                judgement("c1", selected_sense="not_one_of_the_offered_senses.n.01"),
            ]}),
        ])
        report = adjudicator(provider, generate_rewrites=False).adjudicate(
            TEXT, None, [candidate(0)]
        )
        assert report.adjudications[0].status is AdjudicationStatus.LLM_INVALID_RESPONSE
        assert len(provider.requests) == 1
        assert report.diagnostics.repair_attempted is False

    def test_repair_transport_failure_keeps_the_valid_part(self):
        provider = RecordingProvider([
            reply({"judgements": [judgement("c1")]}),
            LLMAuthenticationError("repair rejected"),
        ])
        report = adjudicator(provider, generate_rewrites=False).adjudicate(
            TEXT, None, [candidate(0), candidate(6)]
        )
        assert report.adjudications[0].status is AdjudicationStatus.ADJUDICATED
        assert report.adjudications[1].status is AdjudicationStatus.LLM_INVALID_RESPONSE
        assert report.diagnostics.warnings


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestCaching:
    def test_second_identical_run_makes_no_api_call(self, tmp_path):
        cache = ResponseCache(tmp_path)
        first = RecordingProvider(responder=verdict_responder({}))
        adjudicator(first, cache=cache).adjudicate(TEXT, None, [candidate(0)])
        assert first.requests

        second = RecordingProvider(responder=verdict_responder({}))
        report = adjudicator(second, cache=cache).adjudicate(TEXT, None, [candidate(0)])
        assert second.requests == []
        assert report.diagnostics.cache_hit is True
        assert report.adjudications[0].status is AdjudicationStatus.ADJUDICATED

    def test_invalid_replies_are_never_cached(self, tmp_path):
        cache = ResponseCache(tmp_path)
        adjudicator(RecordingProvider([reply("garbage")]), cache=cache,
                    enable_repair=False).adjudicate(TEXT, None, [candidate(0)])
        assert list(tmp_path.iterdir()) == []

    def test_cache_disabled_in_config_is_respected(self, tmp_path):
        cache = ResponseCache(tmp_path)
        llm = LLMConfig(api_key=SECRET, enable_cache=False)
        adjudicator(RecordingProvider(responder=verdict_responder({})), cache=cache,
                    llm=llm).adjudicate(TEXT, None, [candidate(0)])
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Prompt content: evidence in, secrets and internals out, deterministic
# ---------------------------------------------------------------------------


class TestPromptContent:
    def ranking(self):
        return SenseRanking(
            word="bank", lemma="bank", pos="NOUN",
            status=SemanticAnalysisStatus.RANKED, char_start=6, char_end=10,
            margin=0.09, resolved_by_context=True,
            senses=[SenseOption(sense_key="bank.n.01", definition="sloping land",
                                pos="n", context_similarity=0.81, rank=1)],
        )

    def test_semantic_evidence_reaches_lexical_candidates_only(self):
        provider = RecordingProvider(responder=verdict_responder({}))
        lexical = candidate(6, "bank", AmbiguityType.LEXICAL)
        adjudicator(provider, generate_rewrites=False).adjudicate(
            TEXT, None, [candidate(0), lexical], [self.ranking()]
        )
        payload = json.loads(
            provider.requests[0].messages[-1].content.split("CANDIDATES (JSON):", 1)[1]
        )
        assert payload[0]["semantic_evidence"] is None
        assert payload[1]["semantic_evidence"]["ranked_senses"][0]["sense"] == "bank.n.01"

    def test_prompt_never_contains_the_secret_or_internals(self):
        provider = RecordingProvider(responder=verdict_responder({}))
        adjudicator(provider).adjudicate(
            TEXT, None, [candidate(6, "bank", AmbiguityType.LEXICAL)], [self.ranking()]
        )
        for request in provider.requests:
            for message in request.messages:
                assert SECRET not in message.content
                assert "token_index" not in message.content
                assert "resolved_by_context" not in message.content

    def test_prompts_are_byte_identical_across_runs(self):
        runs = []
        for _ in range(2):
            provider = RecordingProvider(responder=verdict_responder({}))
            adjudicator(provider).adjudicate(TEXT, "Some context.", [candidate(0), candidate(6)])
            runs.append([[m.content for m in r.messages] for r in provider.requests])
        assert runs[0] == runs[1]

    def test_user_context_is_passed_through(self):
        provider = RecordingProvider(responder=verdict_responder({}))
        adjudicator(provider, generate_rewrites=False).adjudicate(
            TEXT, "The river was high.", [candidate(0)]
        )
        assert "The river was high." in provider.requests[0].messages[-1].content


# ---------------------------------------------------------------------------
# No hardcoded answers (Step 20)
# ---------------------------------------------------------------------------


class TestNoHardcodedAnswers:
    def test_evaluation_sentences_do_not_appear_in_llm_code_or_prompts(self):
        """Production logic and prompts must not special-case the test cases."""
        cases = json.loads(
            (PROJECT_ROOT / "data" / "examples" / "adjudication_cases.json")
            .read_text(encoding="utf-8")
        )["cases"]
        sentences = [case["text"].lower() for case in cases]
        sources = list((PROJECT_ROOT / "src" / "ambisense" / "llm").rglob("*.py"))
        sources += list((PROJECT_ROOT / "prompts").glob("*.txt"))
        assert sources
        for path in sources:
            text = Path(path).read_text(encoding="utf-8").lower()
            for sentence in sentences:
                assert sentence not in text, f"{sentence!r} hardcoded in {path.name}"


# ---------------------------------------------------------------------------
# Integration: real Phase 1-3 evidence -> adjudicator (fake provider)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def main_module():
    import main
    return main


class TestIntegrationWithRealEvidence:
    def test_bank_sentence_sends_rule_and_semantic_evidence(self, settings):
        evidence = gather_evidence(settings, "I deposited money at the bank.", None)
        provider = RecordingProvider(responder=verdict_responder({}))
        report = LLMAdjudicator(
            settings.llm, settings.adjudication, provider, cache=None,
            sleep=lambda s: None,
        ).adjudicate(evidence.text, evidence.context, evidence.candidates, evidence.rankings)

        prompt = provider.requests[0].messages[-1].content
        payload = json.loads(prompt.split("CANDIDATES (JSON):", 1)[1])
        lexical = next(p for p in payload if p["type"] == "lexical")
        assert lexical["span"] == "bank"
        assert lexical["semantic_evidence"]["status"] == "ranked"
        assert "not a probability" in lexical["semantic_evidence"]["measure"]
        assert '"prior"' not in prompt and "_index" not in prompt
        assert all(a.status is AdjudicationStatus.ADJUDICATED for a in report.adjudications)

    def test_render_shows_all_three_layers(self, settings, main_module):
        evidence = gather_evidence(settings, "I deposited money at the bank.", None)
        provider = RecordingProvider(responder=verdict_responder({"c1": "not_ambiguous"}))
        report = LLMAdjudicator(
            settings.llm, settings.adjudication, provider, cache=None,
        ).adjudicate(evidence.text, None, evidence.candidates, evidence.rankings)
        output = render_adjudication(report)
        assert "1. Rule-based candidate (Phase 2)" in output
        assert "2. Semantic evidence (Phase 3)" in output
        assert "3. LLM judgement (Phase 4)" in output
        assert "NOT_AMBIGUOUS" in output and "GENUINE_AMBIGUITY" in output
        assert "self-reported" in output and "NOT a" in output
        assert main_module._exit_code_for(report) == main_module.EXIT_OK

    def test_render_of_failure_states_no_verdict(self, settings, main_module):
        evidence = gather_evidence(settings, "I saw the man with the telescope.", None)
        report = LLMAdjudicator(settings.llm, settings.adjudication, None).adjudicate(
            evidence.text, None, evidence.candidates, evidence.rankings
        )
        output = render_adjudication(report)
        assert "LLM_NOT_CONFIGURED" in output
        assert "never reported as 'not ambiguous'" in output
        assert main_module._exit_code_for(report) == main_module.EXIT_LLM_INCOMPLETE
