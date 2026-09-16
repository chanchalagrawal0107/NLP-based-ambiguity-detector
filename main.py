"""AmbiSense command-line interface.

Usage examples::

    python main.py --dump-nlp "I saw the man with the telescope."
    python main.py --check-config

Later phases add full analysis, detector-only and sense-ranking modes.
"""

from __future__ import annotations

import argparse
import sys

from ambisense.config import ConfigError, load_settings
from ambisense.logging_setup import configure_logging, get_logger
from ambisense.preprocessing import (
    InputValidationError,
    LinguisticAnalyzer,
    ModelLoadError,
    clean_and_validate,
)
from ambisense.schemas import LinguisticAnalysis

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_CONFIG_ERROR = 2


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
# Rendering helpers
# ---------------------------------------------------------------------------


def _rule(title: str = "", width: int = 78) -> str:
    if not title:
        return "-" * width
    return f"--- {title} " + "-" * max(0, width - len(title) - 5)


def render_nlp_analysis(analysis: LinguisticAnalysis) -> str:
    """Render the linguistic analysis as a readable table."""
    lines: list[str] = []
    lines.append(_rule("INPUT"))
    lines.append(analysis.text)
    lines.append("")

    lines.append(_rule("SENTENCES"))
    for sentence in analysis.sentences:
        lines.append(f"  [{sentence.index}] {sentence.text}")
    lines.append("")

    lines.append(_rule("TOKENS"))
    header = f"  {'#':>3}  {'TEXT':<14} {'LEMMA':<14} {'POS':<6} {'TAG':<6} {'DEP':<12} HEAD"
    lines.append(header)
    lines.append(f"  {'-' * (len(header) - 2)}")
    for token in analysis.tokens:
        lines.append(
            f"  {token.index:>3}  {token.text:<14} {token.lemma:<14} "
            f"{token.pos:<6} {token.tag:<6} {token.dep:<12} {token.head_text}"
        )
    lines.append("")

    lines.append(_rule("NAMED ENTITIES"))
    if analysis.entities:
        for entity in analysis.entities:
            lines.append(f"  {entity.text:<22} {entity.label}")
    else:
        lines.append("  (none found)")
    lines.append("")

    lines.append(_rule("NOUN CHUNKS"))
    if analysis.noun_chunks:
        lines.append(
            f"  {'CHUNK':<24} {'ROOT':<14} {'DEP':<12} {'PLURAL':<8} PERSON"
        )
        for chunk in analysis.noun_chunks:
            lines.append(
                f"  {chunk.text:<24} {chunk.root_text:<14} {chunk.root_dep:<12} "
                f"{str(chunk.is_plural):<8} {chunk.is_person}"
            )
    else:
        lines.append("  (none found)")
    lines.append("")
    return "\n".join(lines)


def render_config_check(settings, analyzer_ok: bool, wordnet_ok: bool) -> str:
    lines = [_rule("CONFIGURATION CHECK")]
    lines.append(f"  config file      : {settings.project_root / 'config' / 'config.yaml'}")
    lines.append(f"  LLM provider     : {settings.llm.provider}")
    lines.append(f"  LLM model        : {settings.llm.model}")
    lines.append(f"  LLM base URL     : {settings.llm.base_url}")
    key_state = "present" if settings.llm.has_credentials else "MISSING (degraded mode only)"
    lines.append(f"  LLM_API_KEY      : {key_state}")
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
    lines.append(f"  score threshold  : {settings.scoring.ambiguity_threshold}")
    lines.append(f"  weights sum      : {settings.scoring.weights.total():.2f}")
    return "\n".join(lines)


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

    wordnet_ok = True
    try:
        from nltk.corpus import wordnet

        wordnet.synsets("bank")
    except Exception:  # LookupError or ImportError
        wordnet_ok = False

    print(render_config_check(settings, analyzer_ok, wordnet_ok))
    return EXIT_OK if (analyzer_ok and wordnet_ok) else EXIT_CONFIG_ERROR


def command_dump_nlp(settings, text: str, context: str | None) -> int:
    cleaned = clean_and_validate(text, context, settings.nlp)
    for warning in cleaned.warnings:
        logger.warning("%s", warning)

    analyzer = LinguisticAnalyzer(settings.nlp.spacy_model)
    analysis = analyzer.analyze(cleaned.text)
    print(render_nlp_analysis(analysis))

    if cleaned.context:
        context_analysis = analyzer.analyze(cleaned.context)
        print(_rule("CONTEXT ANALYSIS"))
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

    if not args.text:
        print("No text supplied. Try:\n"
              '  python main.py --dump-nlp "I saw the man with the telescope."\n'
              "  python main.py --check-config", file=sys.stderr)
        return EXIT_USER_ERROR

    try:
        if args.dump_nlp:
            return command_dump_nlp(settings, args.text, args.context)
        print("Full analysis is implemented in a later phase. "
              "Use --dump-nlp for the current NLP layer output.",
              file=sys.stderr)
        return EXIT_USER_ERROR
    except InputValidationError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_USER_ERROR
    except ModelLoadError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
