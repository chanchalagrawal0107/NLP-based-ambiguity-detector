"""Phase 4: the LLM adjudication layer.

Rules find candidates. Semantic analysis supplies additional evidence. The LLM
adjudicates plausibility.

The adjudicator never asks "is this sentence ambiguous?". For each candidate
it asks "given this specific evidence, is this candidate a genuine ambiguity?"
- a narrower question whose answer can be checked against the evidence.

Flow for one sentence::

    candidates + rankings
      -> evidence package (c1..cN)             evidence.py
      -> batches of <= max_candidates_per_request
         -> prompt from prompts/ambiguity_analysis.txt
         -> cache lookup / provider call with bounded retry
         -> per-candidate validation
         -> one repair round-trip if any candidate is invalid
      -> for candidates judged GENUINE only:
         -> prompt from prompts/rewrite_generation.txt -> validate
      -> AdjudicationReport

Failure is always explicit. A candidate the LLM could not judge gets
``llm_not_configured`` / ``llm_unavailable`` / ``llm_invalid_response`` - never
a silent ``not_ambiguous``. A failed rewrite step never changes a verdict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from ambisense.config import AdjudicationConfig, LLMConfig, PROMPTS_DIR
from ambisense.llm import evidence as ev
from ambisense.llm.agreement import agrees_with_phase3, phase3_top_sense
from ambisense.llm.cache import ResponseCache
from ambisense.llm.client import LLMClient
from ambisense.llm.errors import LLMError, redact
from ambisense.llm.parser import ParseResult, parse_judgements, parse_rewrites
from ambisense.llm.prompt_builder import load_prompt
from ambisense.llm.providers.base import ChatMessage, LLMProvider, LLMRequest
from ambisense.logging_setup import get_logger
from ambisense.schemas import (
    AdjudicationReport,
    AdjudicationStatus,
    AmbiguityCandidate,
    AnalysisMode,
    CandidateAdjudication,
    PipelineDiagnostics,
    RewriteStatus,
    SenseRanking,
)

logger = get_logger(__name__)

NOT_CONFIGURED_MESSAGE = (
    "No LLM provider was supplied, so nothing was sent. The rule-based and "
    "semantic evidence above is unaffected."
)


@dataclass
class _Usage:
    """Running totals for one sentence's diagnostics."""

    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_reported: bool = False
    repair_attempted: bool = False
    cache_hit: bool = False
    calls_made: bool = False
    warnings: list[str] = field(default_factory=list)


class LLMAdjudicator:
    def __init__(
        self,
        llm_config: LLMConfig,
        config: AdjudicationConfig,
        provider: Optional[LLMProvider],
        *,
        prompts_dir: Path = PROMPTS_DIR,
        cache: Optional[ResponseCache] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args:
        provider: ``None`` means no API key is configured; every candidate is
            then reported as ``llm_not_configured`` and nothing is sent.
        cache: Optional validated-response cache.
        sleep: Injected so retry tests run instantly.
        """
        self.llm_config = llm_config
        self.config = config
        self.provider = provider
        self.cache = cache
        self._client = (
            LLMClient(provider, llm_config, sleep=sleep) if provider else None
        )
        self._adjudication_prompt = load_prompt(prompts_dir / config.adjudication_prompt)
        self._rewrite_prompt = load_prompt(prompts_dir / config.rewrite_prompt)
        self._repair_prompt = load_prompt(prompts_dir / config.repair_prompt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adjudicate(
        self,
        text: str,
        context: Optional[str],
        candidates: Sequence[AmbiguityCandidate],
        rankings: Sequence[SenseRanking] = (),
    ) -> AdjudicationReport:
        started = time.perf_counter()
        usage = _Usage()
        ids = ev.candidate_ids(len(candidates))
        adjudications = {}
        for candidate_id, candidate in zip(ids, candidates):
            ranking = ev.match_ranking(candidate, rankings)
            adjudications[candidate_id] = CandidateAdjudication(
                candidate_id=candidate_id,
                candidate=candidate,
                sense_ranking=ranking,
                phase3_top_sense=phase3_top_sense(ranking),
                status=AdjudicationStatus.LLM_NOT_CONFIGURED,
                error=NOT_CONFIGURED_MESSAGE,
            )

        if candidates and self._client is not None and self.config.enabled:
            size = self.config.batch_size
            for start in range(0, len(ids), size):
                self._adjudicate_batch(
                    text, context, [adjudications[i] for i in ids[start:start + size]],
                    usage,
                )
            if self.config.generate_rewrites:
                self._generate_rewrites(
                    text, [a for a in adjudications.values() if a.is_genuine], usage
                )
        elif candidates and not self.config.enabled:
            for adjudication in adjudications.values():
                adjudication.error = "LLM adjudication is disabled in configuration."

        return AdjudicationReport(
            text=text,
            context=context,
            adjudications=[adjudications[i] for i in ids],
            diagnostics=self._diagnostics(adjudications.values(), usage, started),
        )

    # ------------------------------------------------------------------
    # Adjudication
    # ------------------------------------------------------------------

    def _adjudicate_batch(
        self,
        text: str,
        context: Optional[str],
        batch: list[CandidateAdjudication],
        usage: _Usage,
    ) -> None:
        ids = [a.candidate_id for a in batch]
        payload = [
            ev.candidate_payload(a.candidate_id, a.candidate, a.sense_ranking, self.config)
            for a in batch
        ]
        messages = self._adjudication_prompt.render(
            sentence=text,
            context=context or "(none supplied)",
            candidates=ev.to_json(payload),
        )

        allowed_senses = {
            a.candidate_id: ev.prompt_sense_keys(a.sense_ranking, self.config)
            for a in batch
        }
        try:
            result = self._call_and_parse(
                messages,
                lambda reply: parse_judgements(reply, ids, allowed_senses),
                usage,
            )
        except LLMError as error:
            self._mark_failed(batch, error)
            return

        for adjudication in batch:
            judgement = result.valid.get(adjudication.candidate_id)
            if judgement is not None:
                adjudication.status = AdjudicationStatus.ADJUDICATED
                adjudication.judgement = judgement
                adjudication.error = ""
                # Derived in code from stable sense keys; never read from
                # the model's reply.
                adjudication.agrees_with_phase3 = agrees_with_phase3(
                    adjudication.sense_ranking, judgement.selected_sense
                )
            else:
                adjudication.status = AdjudicationStatus.LLM_INVALID_RESPONSE
                adjudication.error = self._clean(
                    "The LLM reply for this candidate failed validation: "
                    + result.problems.get(adjudication.candidate_id, "unknown problem")
                )

    # ------------------------------------------------------------------
    # Rewrites (separate call, genuine candidates only)
    # ------------------------------------------------------------------

    def _generate_rewrites(
        self,
        text: str,
        genuine: list[CandidateAdjudication],
        usage: _Usage,
    ) -> None:
        size = self.config.batch_size
        for start in range(0, len(genuine), size):
            batch = genuine[start:start + size]
            ids = [a.candidate_id for a in batch]
            messages = self._rewrite_prompt.render(
                sentence=text,
                ambiguities=ev.to_json([ev.rewrite_payload(a) for a in batch]),
            )
            try:
                result = self._call_and_parse(
                    messages, lambda reply: parse_rewrites(reply, ids, text), usage
                )
            except LLMError as error:
                for adjudication in batch:
                    adjudication.rewrite_status = (
                        RewriteStatus.LLM_INVALID_RESPONSE
                        if error.status is AdjudicationStatus.LLM_INVALID_RESPONSE
                        else RewriteStatus.LLM_UNAVAILABLE
                    )
                    adjudication.rewrite_error = self._clean(str(error))
                continue

            for adjudication in batch:
                rewrites = result.valid.get(adjudication.candidate_id)
                if rewrites:
                    adjudication.rewrites = rewrites
                    adjudication.rewrite_status = RewriteStatus.GENERATED
                else:
                    adjudication.rewrite_status = RewriteStatus.LLM_INVALID_RESPONSE
                    adjudication.rewrite_error = self._clean(
                        result.problems.get(adjudication.candidate_id, "")
                    )

    # ------------------------------------------------------------------
    # Shared call path: cache -> provider -> parse -> one repair
    # ------------------------------------------------------------------

    def _call_and_parse(
        self,
        messages: list[ChatMessage],
        parse: Callable[[str], ParseResult],
        usage: _Usage,
    ) -> ParseResult:
        request = LLMRequest(messages=tuple(messages), json_mode=True)
        cache_key = self._cache_key(request)

        if cache_key and (cached := self.cache.get(cache_key)) is not None:
            result = parse(cached)
            if result.complete:
                usage.cache_hit = True
                return result
            logger.info("Cached reply no longer validates; calling the LLM")

        content = self._send(request, usage)
        result = parse(content)

        # Only a format/transport defect is repaired. A contract violation
        # (unknown verdict, a sense never offered...) is a bad answer, not
        # bad JSON, so it is final without ever spending a second request.
        if result.repairable and self.config.enable_repair:
            usage.repair_attempted = True
            repair_messages = [
                *messages,
                ChatMessage("assistant", content),
                *self._repair_prompt.render(problems=result.problem_summary()),
            ]
            try:
                repaired_content = self._send(
                    LLMRequest(messages=tuple(repair_messages), json_mode=True), usage
                )
                repaired = parse(repaired_content)
                result = self._merge(result, repaired)
                if repaired.complete:
                    content = repaired_content
            except LLMError as error:
                # Keep whatever the first reply got right.
                usage.warnings.append(
                    self._clean(f"Repair attempt failed: {type(error).__name__}")
                )

        # Cache only a reply that is complete *on its own*. A result assembled
        # by merging an original and a repair has no single reply that would
        # validate if replayed, so it is not cached.
        if cache_key and result.complete and parse(content).complete:
            self.cache.put(cache_key, content)
        return result

    def _send(self, request: LLMRequest, usage: _Usage) -> str:
        usage.calls_made = True
        try:
            outcome = self._client.complete(request)
        except LLMError as error:
            usage.attempts += error.attempts
            raise
        usage.attempts += outcome.attempts
        response = outcome.response
        if response.prompt_tokens is not None:
            usage.prompt_tokens += response.prompt_tokens
            usage.tokens_reported = True
        if response.completion_tokens is not None:
            usage.completion_tokens += response.completion_tokens
            usage.tokens_reported = True
        return response.content

    @staticmethod
    def _merge(first: ParseResult, second: ParseResult) -> ParseResult:
        """Prefer repaired items, but only for ids the first reply left
        repairable. A contract violation from the first reply is final and
        is never overwritten by whatever the retry happens to contain.
        """
        valid = dict(first.valid)
        for candidate_id in first.repairable:
            if candidate_id in second.valid:
                valid[candidate_id] = second.valid[candidate_id]
        merged = ParseResult(valid=valid)
        for candidate_id, problem in {**first.problems, **second.problems}.items():
            if candidate_id not in merged.valid:
                merged.problems[candidate_id] = second.problems.get(candidate_id, problem)
        return merged

    def _cache_key(self, request: LLMRequest) -> Optional[str]:
        if self.cache is None or not self.llm_config.enable_cache:
            return None
        return ResponseCache.key_for(request, self.llm_config, self.provider.name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mark_failed(self, batch: list[CandidateAdjudication], error: LLMError) -> None:
        for adjudication in batch:
            adjudication.status = error.status
            adjudication.judgement = None
            adjudication.error = self._clean(str(error))

    def _clean(self, message: str) -> str:
        return redact(message, self.llm_config.api_key)

    def _diagnostics(self, adjudications, usage: _Usage, started: float) -> PipelineDiagnostics:
        adjudications = list(adjudications)
        all_judged = all(
            a.status is AdjudicationStatus.ADJUDICATED for a in adjudications
        )
        return PipelineDiagnostics(
            mode=AnalysisMode.FULL if all_judged else AnalysisMode.DEGRADED,
            llm_used=usage.calls_made or usage.cache_hit,
            llm_provider=self.provider.name if self.provider else "",
            llm_model=self.llm_config.model if self.provider else "",
            llm_attempts=usage.attempts,
            llm_prompt_tokens=usage.prompt_tokens if usage.tokens_reported else None,
            llm_completion_tokens=(
                usage.completion_tokens if usage.tokens_reported else None
            ),
            repair_attempted=usage.repair_attempted,
            cache_hit=usage.cache_hit,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            warnings=usage.warnings,
            candidates_found=len(adjudications),
            candidates_confirmed=sum(1 for a in adjudications if a.is_genuine),
        )
