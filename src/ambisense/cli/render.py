"""Terminal rendering for every AmbiSense CLI command.

Kept separate from ``main.py`` so the CLI's argument parsing and command
dispatch are not mixed together with presentation. Nothing here computes
anything; each function only formats data another layer already produced.
"""

from __future__ import annotations

import textwrap

from ambisense.llm.health import ServerStatus, ollama_pull_hint, ollama_serve_hint
from ambisense.schemas import (
    AdjudicationReport,
    AmbiguityCandidate,
    LinguisticAnalysis,
    RewriteStatus,
    SenseRanking,
)


def rule(title: str = "", width: int = 78) -> str:
    if not title:
        return "-" * width
    return f"--- {title} " + "-" * max(0, width - len(title) - 5)


def render_nlp_analysis(analysis: LinguisticAnalysis) -> str:
    """Render the linguistic analysis as a readable table."""
    lines: list[str] = []
    lines.append(rule("INPUT"))
    lines.append(analysis.text)
    lines.append("")

    lines.append(rule("SENTENCES"))
    for sentence in analysis.sentences:
        lines.append(f"  [{sentence.index}] {sentence.text}")
    lines.append("")

    lines.append(rule("TOKENS"))
    header = f"  {'#':>3}  {'TEXT':<14} {'LEMMA':<14} {'POS':<6} {'TAG':<6} {'DEP':<12} HEAD"
    lines.append(header)
    lines.append(f"  {'-' * (len(header) - 2)}")
    for token in analysis.tokens:
        lines.append(
            f"  {token.index:>3}  {token.text:<14} {token.lemma:<14} "
            f"{token.pos:<6} {token.tag:<6} {token.dep:<12} {token.head_text}"
        )
    lines.append("")

    lines.append(rule("NAMED ENTITIES"))
    if analysis.entities:
        for entity in analysis.entities:
            lines.append(f"  {entity.text:<22} {entity.label}")
    else:
        lines.append("  (none found)")
    lines.append("")

    lines.append(rule("NOUN CHUNKS"))
    if analysis.noun_chunks:
        lines.append(
            f"  {'CHUNK':<24} {'ROOT':<14} {'DEP':<12} {'PLURAL':<8} ANIMATE"
        )
        for chunk in analysis.noun_chunks:
            lines.append(
                f"  {chunk.text:<24} {chunk.root_text:<14} {chunk.root_dep:<12} "
                f"{str(chunk.is_plural):<8} {chunk.is_animate_candidate}"
            )
    else:
        lines.append("  (none found)")
    lines.append("")
    return "\n".join(lines)


def _format_evidence(evidence: dict, indent: str = "    ") -> list[str]:
    """Render an evidence dict as aligned ``key: value`` lines."""
    lines: list[str] = []
    for key in sorted(evidence):
        value = evidence[key]
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                lines.append(f"{indent}{key}:")
                for item in value:
                    summary = ", ".join(f"{k}={v}" for k, v in item.items())
                    lines.append(f"{indent}  - {summary}")
                continue
            value = ", ".join(str(item) for item in value) or "(none)"
        elif isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items())
        lines.append(f"{indent}{key}: {value}")
    return lines


def render_candidates(
    text: str,
    candidates: list[AmbiguityCandidate],
    detector_names: list[str],
    *,
    show_evidence: bool = False,
    context: str | None = None,
) -> str:
    """Render the rule-based detection result for the terminal."""
    bar = "=" * 70
    lines = [bar, "AmbiSense - Rule-Based Detection (Phase 2)", bar, ""]
    lines.append("Input:")
    lines.append(f"  {text}")
    if context:
        lines.append("")
        lines.append("Context:")
        lines.append(f"  {context}")
    lines.append("")
    lines.append(f"Detectors run: {', '.join(detector_names) or '(none enabled)'}")
    lines.append(f"Candidates:    {len(candidates)}")
    lines.append("")

    if not candidates:
        lines.append("  No ambiguity candidates were found by the rule-based")
        lines.append("  detectors. This is not proof the text is unambiguous -")
        lines.append("  see the Limitations section of the README.")
        lines.append("")
        lines.append(bar)
        return "\n".join(lines)

    for position, candidate in enumerate(candidates, start=1):
        lines.append("-" * 70)
        lines.append(
            f"[{position}] {candidate.type_hint.value.upper()}"
            f"   (detector: {candidate.detector_name},"
            f" signal strength: {candidate.prior:.2f})"
        )
        lines.append("")
        lines.append("  Span:")
        lines.append(f"    {candidate.span_text!r} "
                     f"[chars {candidate.char_start}-{candidate.char_end}]")
        lines.append("")
        lines.append("  Reason:")
        for line in textwrap.wrap(candidate.explanation, width=64):
            lines.append(f"    {line}")
        if show_evidence:
            lines.append("")
            lines.append("  Evidence:")
            lines.extend(_format_evidence(candidate.evidence))
        lines.append("")

    lines.append(bar)
    lines.append(
        "Note: 'signal strength' is the strength of the linguistic evidence,"
    )
    lines.append(
        "NOT a probability that the text is ambiguous. Confirming genuine"
    )
    lines.append("ambiguity is the job of the LLM layer (Phase 4).")
    lines.append(bar)
    return "\n".join(lines)


def render_sense_rankings(
    text: str,
    rankings: list[SenseRanking],
    *,
    context: str | None = None,
) -> str:
    """Render Phase 3 sense rankings for the terminal."""
    bar = "=" * 70
    lines = [bar, "AmbiSense - Semantic Analysis (Phase 3)", bar, ""]
    lines.append("Input:")
    lines.append(f"  {text}")
    if context:
        lines.append("")
        lines.append("Context:")
        lines.append(f"  {context}")
    lines.append("")

    if not rankings:
        lines.append("  No lexical candidates were found, so there are no word")
        lines.append("  senses to rank. Sense ranking applies to lexical")
        lines.append("  ambiguity only - run --detect-only to see all candidates.")
        lines.append("")
        lines.append(bar)
        return "\n".join(lines)

    lines.append(f"Lexical candidates analysed: {len(rankings)}")
    lines.append("")

    for ranking in rankings:
        lines.append("-" * 70)
        lines.append(f"Candidate:  {ranking.word}   "
                     f"(lemma '{ranking.lemma}', {ranking.pos.lower()})")
        lines.append("")

        if not ranking.status.is_usable:
            lines.append(f"  Status: {ranking.status.value}")
            for line in textwrap.wrap(ranking.note, width=64):
                lines.append(f"  {line}")
            lines.append("")
            continue

        lines.append(f"  Context words used: "
                     f"{', '.join(ranking.context_words) or '(none)'}")
        lines.append(f"  WordNet senses considered: {len(ranking.senses)} "
                     f"of {ranking.senses_available} available")
        lines.append("")

        for sense in ranking.senses:
            similarity = sense.context_similarity
            score = "n/a" if similarity is None else f"{similarity:.4f}"
            lines.append(f"  Rank {sense.rank}:  similarity {score}")
            lines.append(f"    {sense.sense_key}")
            for line in textwrap.wrap(sense.definition, width=60):
                lines.append(f"      {line}")
            lines.append("")

        margin = "n/a" if ranking.margin is None else f"{ranking.margin:.4f}"
        lines.append(f"  Margin over runner-up: {margin}")
        lines.append(f"  Context favours top sense: {ranking.resolved_by_context}")
        lines.append("")
        lines.append("  Interpretation:")
        for line in textwrap.wrap(ranking.note, width=64):
            lines.append(f"    {line}")
        lines.append("")

    lines.append(bar)
    lines.append(
        "Note: these are cosine SIMILARITY scores between the context and each"
    )
    lines.append(
        "sense gloss. They are not probabilities, and a top rank is evidence"
    )
    lines.append(
        "about the context - not a decision about what the writer meant."
    )
    lines.append(bar)
    return "\n".join(lines)


def _wrap(text: str, indent: str, width: int = 66) -> list[str]:
    return [f"{indent}{line}" for line in textwrap.wrap(text, width=width)] or [indent]


def _sense_label(item, sense_key: str) -> str:
    """'crane.n.04 - lifts and moves heavy objects...' using Phase 3's glosses."""
    senses = item.sense_ranking.senses if item.sense_ranking else []
    definition = next(
        (s.definition for s in senses if s.sense_key == sense_key), ""
    )
    return f"{sense_key} - {definition[:44]}" if definition else sense_key


def _phase3_comparison_lines(item) -> list[str]:
    """Phase 3 top sense vs the LLM's selection, and the code-derived result."""
    selected = item.judgement.selected_sense
    agrees = item.agrees_with_phase3
    return [
        f"     Phase 3 top sense:   {_sense_label(item, item.phase3_top_sense)}",
        "     LLM selected sense:  "
        + (_sense_label(item, selected) if selected else "none of the offered senses"),
        "     Agrees with Phase 3: "
        + ("n/a (no sense selected)" if agrees is None else str(agrees).lower())
        + "  [computed by comparing sense keys, not reported by the LLM]",
    ]


def render_adjudication(report: AdjudicationReport) -> str:
    """Render Phase 4 output, keeping the three evidence layers visibly apart."""
    bar = "=" * 70
    lines = [bar, "AmbiSense - LLM Adjudication (Phase 4)", bar, ""]
    lines.append("Input:")
    lines.append(f"  {report.text}")
    if report.context:
        lines += ["", "Context:", f"  {report.context}"]
    lines.append("")

    diagnostics = report.diagnostics
    if not report.adjudications:
        lines += [
            "  No rule-based candidates were found, so nothing was sent to the",
            "  LLM. This is not proof the text is unambiguous - see the README",
            "  limitations.", "", bar,
        ]
        return "\n".join(lines)

    lines.append(f"Candidates adjudicated: {len(report.adjudications)}")
    lines.append("")

    for item in report.adjudications:
        candidate = item.candidate
        lines.append("-" * 70)
        lines.append(f"[{item.candidate_id}] {candidate.type_hint.value.upper()}"
                     f"   span: {candidate.span_text!r}")
        lines.append("")

        lines.append("  1. Rule-based candidate (Phase 2):")
        lines += _wrap(candidate.explanation, "     ")
        lines.append("")

        lines.append("  2. Semantic evidence (Phase 3):")
        ranking = item.sense_ranking
        if ranking is None:
            lines.append("     N/A (sense ranking applies to lexical candidates only)")
        elif not ranking.status.is_usable:
            lines.append(f"     unavailable: {ranking.status.value}")
        else:
            for sense in ranking.senses[:3]:
                lines.append(
                    f"     #{sense.rank} {sense.context_similarity:.3f}  "
                    f"{sense.sense_key} - {sense.definition[:44]}"
                )
            margin = "n/a" if ranking.margin is None else f"{ranking.margin:.3f}"
            lines.append(f"     margin over runner-up: {margin} (similarity, not probability)")
        lines.append("")

        lines.append("  3. LLM judgement (Phase 4):")
        judgement = item.judgement
        if judgement is None:
            lines.append(f"     status: {item.status.value.upper()}")
            lines += _wrap(item.error, "     ")
            lines.append("     (No verdict. A failure is never reported as 'not ambiguous'.)")
            lines.append("")
            continue

        lines.append(f"     verdict: {judgement.verdict.value.upper()}")
        confidence = ("not reported" if judgement.confidence is None
                      else f"{judgement.confidence:.2f} (self-reported)")
        lines.append(f"     confidence: {confidence}")
        if item.phase3_top_sense is not None:
            lines += _phase3_comparison_lines(item)
        lines.append("     explanation:")
        lines += _wrap(judgement.explanation, "       ")

        heading = ("interpretations" if item.is_genuine
                   else "readings considered")
        lines.append(f"     {heading}:")
        for number, interpretation in enumerate(judgement.interpretations, start=1):
            lines += _wrap(f"{number}. {interpretation.meaning}", "       ")
            if interpretation.explanation:
                lines += _wrap(f"({interpretation.explanation})", "          ")

        if item.rewrite_status is RewriteStatus.GENERATED:
            lines.append("     suggested rewrites:")
            for number, rewrite in enumerate(item.rewrites, start=1):
                lines += _wrap(f"{number}. {rewrite}", "       ")
        elif item.is_genuine:
            lines.append(f"     rewrites: {item.rewrite_status.value}")
            if item.rewrite_error:
                lines += _wrap(item.rewrite_error, "       ")
        lines.append("")

    lines.append(bar)
    provider = diagnostics.llm_provider or "none"
    lines.append(f"LLM: {provider} {diagnostics.llm_model}".rstrip()
                 + f" | requests: {diagnostics.llm_attempts}"
                 + f" | cache hit: {diagnostics.cache_hit}"
                 + f" | repair: {diagnostics.repair_attempted}")
    if diagnostics.llm_prompt_tokens is not None:
        lines.append(f"Tokens (provider-reported): prompt {diagnostics.llm_prompt_tokens},"
                     f" completion {diagnostics.llm_completion_tokens}")
    lines.append(f"Genuine ambiguities confirmed: {diagnostics.candidates_confirmed}"
                 f" of {diagnostics.candidates_found} candidates")
    lines.append("")
    lines.append("Note: LLM confidence is self-reported by the model and is NOT a")
    lines.append("calibrated probability. Verdicts are judgements over the evidence")
    lines.append("shown above and can be wrong.")
    lines.append(bar)
    return "\n".join(lines)


def render_config_check(
    settings, analyzer_ok: bool, wordnet_ok: bool, server: ServerStatus
) -> str:
    lines = [rule("CONFIGURATION CHECK")]
    lines.append(f"  config file      : {settings.project_root / 'config' / 'config.yaml'}")
    lines.append(f"  LLM provider     : {settings.llm.provider}")
    lines.append(f"  LLM base URL     : {settings.llm.base_url}")
    if server.reachable:
        lines.append(f"  LLM server       : reachable "
                     f"({len(server.installed_models)} models installed)")
        state = ("installed" if server.model_installed
                 else f"NOT INSTALLED - {ollama_pull_hint(settings.llm.model)}")
        lines.append(f"  LLM model        : {settings.llm.model} [{state}]")
    else:
        lines.append(
            f"  LLM server       : NOT REACHABLE - {ollama_serve_hint()}"
        )
        lines.append(f"  LLM model        : {settings.llm.model} [unknown]")
    lines.append(f"  spaCy model      : {settings.nlp.spacy_model} "
                 f"[{'ok' if analyzer_ok else 'NOT INSTALLED'}]")
    lines.append(f"  WordNet corpus   : {'ok' if wordnet_ok else 'NOT DOWNLOADED'}")
    lines.append(f"  embeddings       : {settings.embeddings.backend}")
    enabled = [
        name for name in
        ("lexical", "syntactic", "referential", "semantic", "scope", "pragmatic")
        if settings.detectors.is_enabled(name)
    ]
    lines.append(f"  detectors on     : {', '.join(enabled) or '(none)'}")
    return "\n".join(lines)
