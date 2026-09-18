"""Typed LLM errors, and the retry and status policy attached to each.

Every failure the provider layer can hit is mapped to one of these classes.
Two properties drive everything downstream:

* ``retryable`` - whether trying again could plausibly succeed. A timeout or a
  rate limit might clear; an invalid API key will not, so retrying it only
  wastes time and quota.
* ``status`` - which :class:`AdjudicationStatus` the candidate receives. No
  error ever maps to "not ambiguous": an outage is not a verdict.
"""

from __future__ import annotations

from typing import Optional

from ambisense.schemas import AdjudicationStatus


class LLMError(Exception):
    """Base class for all LLM failures."""

    retryable: bool = False
    status: AdjudicationStatus = AdjudicationStatus.LLM_UNAVAILABLE

    def __init__(self, message: str, *, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        #: Filled in by the client: how many requests were made in total.
        self.attempts: int = 0


class LLMConfigurationError(LLMError):
    """No usable API key, or an unknown provider. Nothing was sent."""

    status = AdjudicationStatus.LLM_NOT_CONFIGURED


class LLMAuthenticationError(LLMError):
    """HTTP 401/403 - the key was rejected. Permanent."""


class LLMRequestError(LLMError):
    """Other HTTP 4xx - the request itself is wrong (e.g. unknown model)."""


class LLMRateLimitError(LLMError):
    """HTTP 429 - too many requests or tokens. May clear after a wait."""

    retryable = True


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""

    retryable = True


class LLMConnectionError(LLMError):
    """DNS, TLS or socket failure before a response arrived."""

    retryable = True


class LLMServerError(LLMError):
    """HTTP 5xx - a fault on the provider's side."""

    retryable = True


class LLMResponseFormatError(LLMError):
    """A response arrived but is unusable: empty, truncated or malformed.

    ``retryable`` is set per instance: an empty body may be a transient glitch
    worth retrying, whereas a response truncated at ``max_tokens`` would be
    truncated identically next time.
    """

    status = AdjudicationStatus.LLM_INVALID_RESPONSE

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message, retry_after=retry_after)
        self.retryable = retryable


def redact(text: str, secret: Optional[str]) -> str:
    """Remove a secret from any text that might be logged or displayed.

    Provider error bodies are echoed into reports; this is the last line of
    defence ensuring an API key can never appear in them, even if a provider
    started including it.
    """
    if not text or not secret or len(secret) < 4:
        return text
    return text.replace(secret, "[REDACTED]")
