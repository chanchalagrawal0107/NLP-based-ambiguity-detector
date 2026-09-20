"""Phase 7: the labelled sentence dataset consumed by ``--evaluate``.

Ground truth is a single developer's judgement per sentence, at the sentence
level only: does the sentence contain at least one genuine ambiguity? This is
a deliberately narrower question than candidate-level span matching (which
would need a span-overlap and type-matching algorithm this project does not
build) - see the README's Phase 7 section for the honesty tradeoff this
implies. The single-annotator limitation is already documented in the
README's Future Work section and is not repeated here.

Kept separate from ``data/examples/adjudication_cases.json``, which has no
ground-truth field and is documented as demo-only for ``--live-llm-test``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DatasetError(ValueError):
    """The dataset file is missing, malformed, or fails validation."""


class EvaluationCase(BaseModel):
    """One labelled sentence."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    context: Optional[str] = None
    label: Literal["ambiguous", "not_ambiguous"]
    notes: str = ""


def load_dataset(path: Path) -> list[EvaluationCase]:
    """Load and validate the labelled dataset. Never silently drops a case."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(f"could not read evaluation dataset {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc

    cases_raw = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(cases_raw, list) or not cases_raw:
        raise DatasetError(f"{path} must contain a non-empty 'cases' list")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(cases_raw):
        try:
            case = EvaluationCase.model_validate(item)
        except Exception as exc:  # pydantic ValidationError
            raise DatasetError(f"case {index} in {path} is invalid: {exc}") from exc
        if case.id in seen_ids:
            raise DatasetError(f"duplicate case id {case.id!r} in {path}")
        seen_ids.add(case.id)
        cases.append(case)
    return cases
