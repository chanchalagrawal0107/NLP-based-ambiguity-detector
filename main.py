"""AmbiSense command-line interface.

Usage examples::

    python main.py --analyze "I saw the man with the telescope."
    python main.py --detect-only "I saw the man with the telescope."
    python main.py --analyze-semantics "I deposited money at the bank."
    python main.py --dump-nlp "I saw the man with the telescope."
    python main.py --live-llm-test
    python main.py --check-config
"""

from __future__ import annotations

import argparse
import json
import sys

from ambisense.ambiguity import DetectorRegistry
from ambisense.ambiguity.wordnet_support import wordnet_available
from ambisense.cli.render import (
    render_adjudication,
    render_candidates,
    render_config_check,
    render_evaluation_report,
    render_nlp_analysis,
    render_sense_rankings,
    rule,
)
from ambisense.config import ConfigError, load_settings
from ambisense.evaluation import DatasetError, load_dataset, run_evaluation
from ambisense.llm import build_adjudicator
from ambisense.llm.health import check_server, ollama_pull_hint, ollama_serve_hint
from ambisense.logging_setup import configure_logging, get_logger
from ambisense.pipeline import SetupError, analyze, gather_evidence
from ambisense.preprocessing import (
    InputValidationError,
    LinguisticAnalyzer,
    ModelLoadError,
    clean_and_validate,
)
from ambisense.schemas import AdjudicationReport, AdjudicationStatus

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_CONFIG_ERROR = 2
#: The report was produced, but at least one candidate has no LLM judgement.
EXIT_LLM_INCOMPLETE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ambisense",
        description=(
            "AmbiSense - hybrid NLP + LLM ambiguity detection and resolution."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="The sentence or paragraph to analyse.",
    )
    parser.add_argument(
        "-c", "--context",
        default=None,
        help="Optional surrounding context used to disambiguate.",
    )
    parser.add_argument(
        "--dump-nlp",
        action="store_true",
        help="Print the traditional NLP analysis (POS, dependencies, NER).",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help=(
            "Run the rule-based detectors and print the candidates. "
            "No LLM call is made."
        ),
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="With --detect-only, print the full evidence for each candidate.",
    )
    parser.add_argument(
        "--analyze-semantics",
        action="store_true",
        help=(
            "Rank the WordNet senses of each lexical candidate against the "
            "sentence context. No LLM call is made."
        ),
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "Full pipeline: rule-based candidates, semantic evidence, then "
            "LLM adjudication. Requires a running Ollama server."
        ),
    )
    parser.add_argument(
        "--live-llm-test",
        action="store_true",
        help=(
            "Opt-in: run data/examples/adjudication_cases.json against the "
            "local Ollama model. Slow on local hardware. Never run by pytest."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Opt-in: run data/evaluation/sentence_labels.json against the "
            "local Ollama model and report accuracy/precision/recall/F1. "
            "Slow on local hardware. Never run by pytest."
        ),
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate config.yaml and the environment, then exit.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternative config.yaml.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override the configured logging level.",
    )
    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_check_config(settings) -> int:
    analyzer_ok = True
    try:
        LinguisticAnalyzer(settings.nlp.spacy_model)
    except ModelLoadError as exc:
        analyzer_ok = False
        logger.error("%s", exc)

    wordnet_ok = wordnet_available()

    # The LLM server is reported but not treated as fatal: every mode except
    # --analyze and --live-llm-test works without it.
    server = check_server(settings.llm)
    print(render_config_check(settings, analyzer_ok, wordnet_ok, server))
    return EXIT_OK if (analyzer_ok and wordnet_ok) else EXIT_CONFIG_ERROR


def command_detect_only(
    settings,
    text: str,
    context: str | None,
    show_evidence: bool,
) -> int:
    """Validation -> linguistic analysis -> rule-based detectors -> report.

    Makes no network call and uses no LLM.
    """
    cleaned = clean_and_validate(text, context, settings.nlp)
    for warning in cleaned.warnings:
        logger.warning("%s", warning)

    analyzer = LinguisticAnalyzer(settings.nlp.spacy_model)
    analysis = analyzer.analyze(cleaned.text)

    registry = DetectorRegistry(settings.detectors)
    candidates = registry.run(analysis)

    print(render_candidates(
        cleaned.text,
        candidates,
        registry.detector_names,
        show_evidence=show_evidence,
        context=cleaned.context,
    ))

    if cleaned.context:
        logger.info(
            "The rule-based detectors do not use context. Context is used by "
            "--analyze-semantics and --analyze."
        )
    return EXIT_OK


def command_analyze_semantics(
    settings,
    text: str,
    context: str | None,
) -> int:
    """Validation -> analysis -> detectors -> WordNet sense ranking.

    Makes no network call and uses no LLM.
    """
    evidence = gather_evidence(settings, text, context)
    print(render_sense_rankings(
        evidence.text, evidence.rankings, context=evidence.context
    ))
    return EXIT_OK


def _exit_code_for(report: AdjudicationReport) -> int:
    """0 when every candidate was judged; 3 when any judgement is missing."""
    if any(a.status is not AdjudicationStatus.ADJUDICATED
           for a in report.adjudications):
        return EXIT_LLM_INCOMPLETE
    return EXIT_OK


def command_analyze(settings, text: str, context: str | None) -> int:
    """Full pipeline: evidence from Phases 1-3, then LLM adjudication."""
    report = analyze(settings, text, context)
    print(render_adjudication(report))
    return _exit_code_for(report)


def command_live_llm_test(settings) -> int:
    """Opt-in: run the documented adjudication cases against the local model.

    Never invoked by pytest. Requires a running Ollama server with the
    configured model installed; checked up front so a missing server fails
    in seconds rather than after the first slow request.
    """
    server = check_server(settings.llm)
    if not server.ready:
        reason = (
            f"model '{settings.llm.model}' is not installed "
            f"({ollama_pull_hint(settings.llm.model)})"
            if server.reachable
            else f"no Ollama server at {settings.llm.base_url} "
                 f"({ollama_serve_hint()})"
        )
        print(f"--live-llm-test cannot run: {reason}. No request was sent.",
              file=sys.stderr)
        return EXIT_CONFIG_ERROR

    cases_path = settings.project_root / "data" / "examples" / "adjudication_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    adjudicator = build_adjudicator(settings)

    worst = EXIT_OK
    for number, case in enumerate(cases, start=1):
        print()
        print(f"#### Case {number}/{len(cases)}: {case['id']}")
        print(f"#### Reviewer note: {case['what_to_check']}")
        report = analyze(
            settings, case["text"], case.get("context"), adjudicator=adjudicator
        )
        print(render_adjudication(report))
        worst = max(worst, _exit_code_for(report))
    return worst


def command_evaluate(settings) -> int:
    """Opt-in: score the labelled dataset against the local model.

    Never invoked by pytest. Mirrors ``command_live_llm_test``'s shape: the
    server is checked up front so a missing model fails in seconds, and one
    adjudicator is shared across cases so the response cache applies.
    """
    server = check_server(settings.llm)
    if not server.ready:
        reason = (
            f"model '{settings.llm.model}' is not installed "
            f"({ollama_pull_hint(settings.llm.model)})"
            if server.reachable
            else f"no Ollama server at {settings.llm.base_url} "
                 f"({ollama_serve_hint()})"
        )
        print(f"--evaluate cannot run: {reason}. No request was sent.",
              file=sys.stderr)
        return EXIT_CONFIG_ERROR

    dataset_path = (
        settings.project_root / "data" / "evaluation" / "sentence_labels.json"
    )
    try:
        dataset = load_dataset(dataset_path)
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    results = run_evaluation(settings, dataset, adjudicator=build_adjudicator(settings))
    print(render_evaluation_report(results))
    return EXIT_OK


def command_dump_nlp(settings, text: str, context: str | None) -> int:
    cleaned = clean_and_validate(text, context, settings.nlp)
    for warning in cleaned.warnings:
        logger.warning("%s", warning)

    analyzer = LinguisticAnalyzer(settings.nlp.spacy_model)
    analysis = analyzer.analyze(cleaned.text)
    print(render_nlp_analysis(analysis))

    if cleaned.context:
        context_analysis = analyzer.analyze(cleaned.context)
        print(rule("CONTEXT ANALYSIS"))
        print(render_nlp_analysis(context_analysis))
    return EXIT_OK


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.log_level:
        settings.logging.level = args.log_level
    configure_logging(settings.logging)

    if args.check_config:
        return command_check_config(settings)

    if args.live_llm_test:
        try:
            return command_live_llm_test(settings)
        except (ModelLoadError, SetupError) as exc:
            print(f"Setup error: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR

    if args.evaluate:
        try:
            return command_evaluate(settings)
        except (ModelLoadError, SetupError) as exc:
            print(f"Setup error: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR

    if not args.text:
        print("No text supplied. Try:\n"
              '  python main.py --detect-only "I saw the man with the telescope."\n'
              '  python main.py --analyze "I saw the man with the telescope."\n'
              '  python main.py --analyze-semantics "I deposited money at the bank."\n'
              '  python main.py --dump-nlp "I saw the man with the telescope."\n'
              "  python main.py --check-config", file=sys.stderr)
        return EXIT_USER_ERROR

    try:
        if args.dump_nlp:
            return command_dump_nlp(settings, args.text, args.context)
        if args.detect_only:
            return command_detect_only(
                settings, args.text, args.context, args.show_evidence
            )
        if args.analyze_semantics:
            return command_analyze_semantics(
                settings, args.text, args.context
            )
        if args.analyze:
            return command_analyze(settings, args.text, args.context)
        print("Choose a mode: --analyze (full pipeline with LLM "
              "adjudication), --detect-only (rule-based candidates), "
              "--analyze-semantics (WordNet sense ranking) or --dump-nlp "
              "(NLP layer output).",
              file=sys.stderr)
        return EXIT_USER_ERROR
    except InputValidationError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR
    except (ModelLoadError, SetupError) as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
