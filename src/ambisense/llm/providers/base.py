"""The provider interface: the only thing the rest of AmbiSense knows about an LLM.

A provider does exactly one job - send a list of chat messages, get text back -
and reports failure by raising a typed :class:`~ambisense.llm.errors.LLMError`.
It knows nothing about ambiguity, prompts, retries or validation. Those live in
separate modules, so a provider can be swapped without touching them, and each
of them can be tested with a fake provider and no network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMRequest:
    """One provider call. Immutable, so it can double as a cache key source."""

    messages: tuple[ChatMessage, ...]
    json_mode: bool = True


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can complete a chat request."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the model's reply, or raise an ``LLMError`` subclass."""
        ...
