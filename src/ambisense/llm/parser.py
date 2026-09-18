"""Validate LLM replies before anything downstream is allowed to use them.

Pipeline::

    raw text -> JSON object -> per-candidate Pydantic validation

Validation is **per candidate**. In a batched reply, one malformed judgement
must not discard the valid ones beside it, so each item is validated on its
own and problems are recorded against the specific ``candidate_id``.

Nothing here guesses. A missing judgement stays missing; an unreadable verdict
is a problem, not ``uncertain``; an unreadable confidence is ``None``, not 0.5.

Repairable vs final
--------------------
A problem is either **repairable** (a transport/format defect: the reply was
not JSON, an expected item never arrived at all) or **final** (a contract
violation: an unrecognised verdict, ``genuine_ambiguity`` with fewer than two
interpretations, a ``selected_sense`` that was never offered). Only repairable
problems are sent back for the one bounded repair round-trip - a contract
violation is a bad *answer*, not a bad *format*, and re-asking the model to
"fix the JSON" would let it silently change its answer instead. See
``ParseResult.repairable`` and ``LLMAdjudicator._call_and_parse``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from ambisense.schemas import LLMJudgement

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ResponseParseError(ValueError):
    """The reply is not a JSON object at all."""


@dataclass
class ParseResult:
    """Valid items keyed by candidate id, plus a readable list of problems.

    ``repairable`` is the subset of ``problems`` keys that are transport/format
    defects rather than contract violations - see the module docstring.
    """

    valid: dict[str, Any] = field(default_factory=dict)
    problems: dict[str, str] = field(default_factory=dict)
    repairable: set[str] = field(default_factory=set)

    @property
    def complete(self) -> bool:
        return not self.problems

    def problem_summary(self) -> str:
        """The repairable problems only - a contract violation is not sent back."""
        return "\n".join(
            f"- {candidate_id}: {problem}"
            for candidate_id, problem in sorted(self.problems.items())
            if candidate_id in self.repairable
        )

    def _record(self, candidate_id: str, problem: str, *, repairable: bool) -> None:
        self.problems.setdefault(candidate_id, problem)
        if repairable:
            self.repairable.add(candidate_id)

    def _accept(self, candidate_id: str, value: Any) -> None:
        self.valid[candidate_id] = value
        self.problems.pop(candidate_id, None)
        self.repairable.discard(candidate_id)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating markdown fences or stray prose.

    JSON mode should make this unnecessary, but not every model honours it
    perfectly, and a reply wrapped in ```json fences is still recoverable.
    """
    if not text or not text.strip():
        raise ResponseParseError("the reply was empty")
    candidate = _FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ResponseParseError("the reply contained no JSON object")
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ResponseParseError(f"the reply is not valid JSON ({exc.msg})")
    if not isinstance(parsed, dict):
        raise ResponseParseError("the reply is JSON but not a JSON object")
    return parsed


def _summarise(error: ValidationError) -> str:
    """Compact, input-free summary of a Pydantic error (safe to log or resend)."""
    parts = []
    for item in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "item"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)


def _items(parsed: dict[str, Any], list_key: str) -> list[Any]:
    items = parsed.get(list_key)
    if not isinstance(items, list):
        raise ResponseParseError(f"the reply has no '{list_key}' list")
    return items


def _selected_sense_problem(
    judgement: LLMJudgement, allowed: frozenset[str]
) -> str | None:
    """A selected sense must be one of the keys offered for that candidate.

    An unoffered key - a made-up id, a free-text description, or any key for a
    candidate that had no sense evidence - is malformed. It is reported as a
    problem (so the repair round-trip can fix it) rather than being guessed
    into "no selection", which would change the derived agreement silently.
    """
    selected = judgement.selected_sense
    if selected is None or selected in allowed:
        return None
    if not allowed:
        return (f"selected_sense {selected!r} given, but no senses were supplied "
                f"for this candidate; it must be null")
    return (f"selected_sense {selected!r} is not one of the supplied senses "
            f"{sorted(allowed)}")


def parse_judgements(
    text: str,
    expected_ids: Sequence[str],
    allowed_senses: Mapping[str, frozenset[str]] | None = None,
) -> ParseResult:
    """Validate an adjudication reply against the candidates that were sent.

    ``allowed_senses`` maps candidate id -> the sense keys shown for it. A
    candidate absent from the mapping was shown no senses.
    """
    result = ParseResult()
    expected = set(expected_ids)
    allowed_senses = allowed_senses or {}
    try:
        items = _items(extract_json_object(text), "judgements")
    except ResponseParseError as exc:
        for candidate_id in expected_ids:
            result._record(candidate_id, str(exc), repairable=True)
        return result

    for item in items:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id not in expected or candidate_id in result.valid:
            # Unknown ids are ignored; for duplicates the first valid one wins.
            continue
        try:
            judgement = LLMJudgement.model_validate(item)
        except ValidationError as exc:
            # A contract violation (unknown verdict, missing field, a genuine
            # ambiguity with one reading...) is a bad answer, not bad JSON.
            # It is final: never sent back for repair.
            result._record(candidate_id, _summarise(exc), repairable=False)
            continue
        problem = _selected_sense_problem(
            judgement, allowed_senses.get(candidate_id, frozenset())
        )
        if problem:
            # Also a contract violation - the model chose a sense it was
            # never shown - not a format defect.
            result._record(candidate_id, problem, repairable=False)
            continue
        result._accept(candidate_id, judgement)

    for candidate_id in expected_ids:
        if candidate_id not in result.valid and candidate_id not in result.problems:
            # The reply parsed, but this candidate never got an item at all -
            # a format defect (the model dropped it), so it is repairable.
            result._record(candidate_id, "no judgement was returned", repairable=True)
    return result


def parse_rewrites(
    text: str,
    expected_ids: Sequence[str],
    original_sentence: str,
) -> ParseResult:
    """Validate a rewrite reply. ``valid`` maps candidate id -> list[str]."""
    result = ParseResult()
    expected = set(expected_ids)
    original = " ".join(original_sentence.split()).lower()
    try:
        items = _items(extract_json_object(text), "rewrites")
    except ResponseParseError as exc:
        for candidate_id in expected_ids:
            result._record(candidate_id, str(exc), repairable=True)
        return result

    for item in items:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id not in expected or candidate_id in result.valid:
            continue
        raw = item.get("rewrites")
        if not isinstance(raw, list):
            # A format defect: the field has the wrong shape.
            result._record(candidate_id, "rewrites must be a list", repairable=True)
            continue
        rewrites: list[str] = []
        for rewrite in raw:
            cleaned = " ".join(str(rewrite or "").split())
            # A "rewrite" identical to the original disambiguates nothing.
            if (cleaned and cleaned.lower() != original
                    and cleaned.lower() not in {r.lower() for r in rewrites}):
                rewrites.append(cleaned)
        if rewrites:
            result._accept(candidate_id, rewrites)
        else:
            # A contract violation: the model answered, but every rewrite
            # failed to disambiguate anything - not a format defect.
            result._record(
                candidate_id, "no usable rewrite differing from the original",
                repairable=False,
            )

    for candidate_id in expected_ids:
        if candidate_id not in result.valid and candidate_id not in result.problems:
            result._record(candidate_id, "no rewrites were returned", repairable=True)
    return result
