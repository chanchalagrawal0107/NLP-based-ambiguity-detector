"""Input validation and normalisation - the first layer of the pipeline.

Runs before any expensive work (spaCy model, WordNet lookups, API calls) so
that bad input fails immediately, cheaply, and with a message the user can act
on. Every guard here is driven by ``config.yaml`` under ``nlp:``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ambisense.config import NLPConfig
from ambisense.logging_setup import get_logger

logger = get_logger(__name__)


class InputValidationError(ValueError):
    """Raised when input cannot be analysed.

    Carries a user-facing message; the UI displays ``str(exc)`` directly
    instead of a traceback.
    """


# Curly quotes and dashes confuse both the parser and span offsets.
_SMART_CHARS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class CleanedInput:
    """Validated, normalised text ready for linguistic analysis."""

    text: str
    context: str | None
    original_text: str
    was_truncated: bool = False
    warnings: tuple[str, ...] = ()


def normalise_text(text: str) -> str:
    """Normalise unicode and collapse whitespace without changing wording.

    Offsets reported later refer to this normalised string, which is also the
    string shown back to the user, so the two can never drift apart.
    """
    text = unicodedata.normalize("NFKC", text)
    for smart, plain in _SMART_CHARS.items():
        text = text.replace(smart, plain)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _ascii_letter_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are ASCII.

    A crude but effective English-only guard: Devanagari, Cyrillic or CJK input
    scores near zero, while English with a few accents still scores high.
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 1.0
    ascii_letters = sum(1 for char in letters if char.isascii())
    return ascii_letters / len(letters)


def clean_and_validate(
    text: str,
    context: str | None,
    config: NLPConfig,
) -> CleanedInput:
    """Validate and normalise user input.

    Args:
        text: The sentence or paragraph to analyse.
        context: Optional surrounding context supplied by the user.
        config: The ``nlp`` section of the configuration.

    Returns:
        A :class:`CleanedInput` with normalised text and any soft warnings.

    Raises:
        InputValidationError: for empty, too-short, or non-English input.
    """
    warnings: list[str] = []

    if text is None or not str(text).strip():
        raise InputValidationError(
            "No text provided. Enter a sentence to analyse."
        )

    original = str(text)
    cleaned = normalise_text(original)

    if len(cleaned) < config.min_input_length:
        raise InputValidationError(
            f"Text is too short to analyse (minimum "
            f"{config.min_input_length} characters)."
        )

    was_truncated = False
    if len(cleaned) > config.max_input_length:
        cleaned = cleaned[: config.max_input_length].rsplit(" ", 1)[0]
        was_truncated = True
        warnings.append(
            f"Input exceeded {config.max_input_length} characters and was "
            "truncated. Analyse shorter passages for best results."
        )

    ratio = _ascii_letter_ratio(cleaned)
    if ratio < config.min_ascii_ratio:
        raise InputValidationError(
            "This text does not appear to be English. AmbiSense analyses "
            "English text only - see the Limitations section of the README."
        )

    cleaned_context: str | None = None
    if context and str(context).strip():
        cleaned_context = normalise_text(str(context))
        if len(cleaned_context) > config.max_context_length:
            cleaned_context = cleaned_context[: config.max_context_length]
            warnings.append("Context was truncated to the configured maximum.")

    logger.debug(
        "Input validated: %d chars, context=%s",
        len(cleaned),
        "yes" if cleaned_context else "no",
    )

    return CleanedInput(
        text=cleaned,
        context=cleaned_context,
        original_text=original,
        was_truncated=was_truncated,
        warnings=tuple(warnings),
    )
