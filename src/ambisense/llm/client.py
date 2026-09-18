"""Bounded retry around a provider.

The retry policy lives here, not in the provider, so that every provider gets
the same behaviour and the policy can be tested with a fake provider and a
fake clock.

Policy
------
* Only errors marked ``retryable`` are retried (timeouts, 429, 5xx, network,
  transient empty replies). Authentication failures and malformed requests
  fail immediately: repeating them cannot help and wastes quota.
* At most ``llm.max_retries`` extra attempts. Never unbounded.
* Waits grow exponentially (``backoff x 2^n``). A server's ``Retry-After``
  header is honoured, but every wait is capped at
  ``llm.max_retry_wait_seconds`` so a demo cannot stall indefinitely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ambisense.config import LLMConfig
from ambisense.llm.errors import LLMError
from ambisense.llm.providers.base import LLMProvider, LLMRequest, LLMResponse
from ambisense.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CompletionOutcome:
    response: LLMResponse
    attempts: int


class LLMClient:
    """Sends a request through a provider with bounded retries."""

    def __init__(
        self,
        provider: LLMProvider,
        config: LLMConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Args:
        sleep: Injected so tests can verify the wait schedule instantly.
        """
        self.provider = provider
        self.config = config
        self._sleep = sleep

    def wait_before_retry(self, attempt: int, error: LLMError) -> float:
        """Seconds to wait after failed attempt number ``attempt`` (1-based)."""
        backoff = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
        wait = error.retry_after if error.retry_after is not None else backoff
        return max(0.0, min(wait, self.config.max_retry_wait_seconds))

    def complete(self, request: LLMRequest) -> CompletionOutcome:
        max_attempts = 1 + max(0, self.config.max_retries)
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.provider.complete(request)
                return CompletionOutcome(response=response, attempts=attempt)
            except LLMError as error:
                error.attempts = attempt
                if not error.retryable or attempt == max_attempts:
                    logger.warning(
                        "LLM request failed after %d attempt(s): %s",
                        attempt, type(error).__name__,
                    )
                    raise
                wait = self.wait_before_retry(attempt, error)
                logger.info(
                    "LLM attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt, max_attempts, type(error).__name__, wait,
                )
                self._sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover
