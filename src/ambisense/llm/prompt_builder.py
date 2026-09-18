"""Load prompt templates from ``prompts/`` and fill them with evidence.

Prompts live in text files, not in Python, so they can be read, reviewed and
edited without touching code - and so the exact wording sent to the model is
part of the repository.

File format
-----------
A prompt file has an optional ``===SYSTEM===`` section (stable instructions)
and a required ``===USER===`` section (the per-request evidence)::

    ===SYSTEM===
    You are ...
    ===USER===
    SENTENCE: $sentence

Why ``string.Template`` and not ``str.format``
----------------------------------------------
The prompts contain literal JSON examples full of ``{`` and ``}``. With
``str.format`` every brace would have to be doubled, and one missed brace
breaks the prompt. ``string.Template`` only reacts to ``$name``, so the JSON
examples can be written naturally. ``substitute`` (not ``safe_substitute``)
is used so a missing value raises instead of silently sending a literal
``$sentence`` to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Optional

from ambisense.llm.providers.base import ChatMessage

SYSTEM_MARKER = "===SYSTEM==="
USER_MARKER = "===USER==="


class PromptTemplateError(ValueError):
    """A prompt file is missing, malformed, or was rendered with missing values."""


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: Optional[Template]
    user: Template

    def render(self, **values: str) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        try:
            if self.system is not None:
                messages.append(
                    ChatMessage("system", self.system.substitute(values).strip())
                )
            messages.append(
                ChatMessage("user", self.user.substitute(values).strip())
            )
        except KeyError as exc:
            raise PromptTemplateError(
                f"Prompt '{self.name}' needs a value for ${exc.args[0]}"
            ) from exc
        except ValueError as exc:
            raise PromptTemplateError(
                f"Prompt '{self.name}' contains an invalid '$' placeholder: {exc}"
            ) from exc
        return messages


def parse_prompt_text(text: str, name: str = "prompt") -> PromptTemplate:
    if USER_MARKER not in text:
        raise PromptTemplateError(f"Prompt '{name}' has no {USER_MARKER} section")
    before_user, user_text = text.split(USER_MARKER, 1)
    system_text: Optional[str] = None
    if SYSTEM_MARKER in before_user:
        system_text = before_user.split(SYSTEM_MARKER, 1)[1].strip()
    elif before_user.strip():
        raise PromptTemplateError(
            f"Prompt '{name}' has text before {USER_MARKER} but no "
            f"{SYSTEM_MARKER} marker"
        )
    if not user_text.strip():
        raise PromptTemplateError(f"Prompt '{name}' has an empty user section")
    return PromptTemplate(
        name=name,
        system=Template(system_text) if system_text else None,
        user=Template(user_text.strip()),
    )


def load_prompt(path: Path) -> PromptTemplate:
    """Read and parse a prompt file.

    Not cached: prompt files are a few kilobytes, and re-reading them means an
    edited prompt takes effect on the next run without a restart.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptTemplateError(f"Cannot read prompt file {path.name}") from exc
    return parse_prompt_text(text, name=path.name)
