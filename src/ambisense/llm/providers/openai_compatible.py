"""Provider for an OpenAI-compatible chat API, used to reach a local Ollama server.

Ollama serves ``POST {base_url}/chat/completions`` in the OpenAI format, so
this adapter speaks that protocol with plain ``httpx`` rather than an SDK:

* no new dependency (``httpx`` was already required),
* the actual HTTP request is visible in a few dozen lines, easy to explain,
* every HTTP outcome can be tested offline with ``httpx.MockTransport``.

The model runs on the local machine, so there is no API key and no per-request
cost. Reasoning models such as qwen3 return their thinking in a separate
``reasoning`` field; only ``content`` is used, which holds the JSON answer.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from ambisense.config import LLMConfig
from ambisense.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseFormatError,
    LLMServerError,
    LLMTimeoutError,
    redact,
)
from ambisense.llm.health import ollama_pull_hint, ollama_serve_hint
from ambisense.llm.providers.base import LLMRequest, LLMResponse
from ambisense.logging_setup import get_logger

logger = get_logger(__name__)

#: Longest server error text copied into a report.
_MAX_ERROR_DETAIL = 200


class OpenAICompatibleProvider:
    """Calls ``POST {base_url}/chat/completions``."""

    def __init__(
        self,
        config: LLMConfig,
        api_key: Optional[str] = None,
        *,
        provider_name: str = "ollama",
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        """Args:
        config: Transport settings (model, URL, temperature, timeout...).
        api_key: Optional. Only needed if the server sits behind an
            authenticating proxy; a plain local Ollama server needs none.
        provider_name: Label used in diagnostics.
        transport: Injected by tests to simulate HTTP responses offline.
        """
        self._config = config
        self._api_key = api_key
        self._name = provider_name
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._config.model

    # -- request -----------------------------------------------------------

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if request.json_mode and self._config.request_json_mode:
            # Server-side guarantee that the reply is syntactically valid
            # JSON. It does not guarantee the JSON matches our schema - that is
            # still checked by the parser.
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(self, request: LLMRequest) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            response = self._client.post(
                "/chat/completions",
                json=self.build_payload(request),
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"The {self._name} model '{self._config.model}' did not answer "
                f"within {self._config.timeout_seconds:g}s. Local inference can "
                f"be slow; raise llm.timeout_seconds or use a smaller model."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(
                f"Could not reach {self._name} at {self._config.base_url} "
                f"({type(exc).__name__}). Is the server running? "
                f"{ollama_serve_hint()}"
            ) from exc

        if response.status_code >= 400:
            raise self._error_for(response)
        return self._parse_success(response)

    # -- response ----------------------------------------------------------

    def _parse_success(self, response: httpx.Response) -> LLMResponse:
        try:
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseFormatError(
                f"{self._name} returned a response without the expected "
                f"choices[0].message.content structure",
                retryable=True,
            ) from exc

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise LLMResponseFormatError(
                "The reply was cut off at the max_tokens limit, so it is "
                "incomplete. Reasoning models spend tokens thinking before "
                "answering; increase llm.max_tokens or reduce "
                "adjudication.max_candidates_per_request.",
                retryable=False,
            )
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseFormatError(
                f"{self._name} returned an empty reply", retryable=True
            )

        usage = body.get("usage") or {}
        return LLMResponse(
            content=content,
            model=str(body.get("model") or self._config.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
        )

    def _error_for(self, response: httpx.Response):
        status = response.status_code
        detail = self._error_detail(response)
        message = redact(f"{self._name} HTTP {status}: {detail}", self._api_key)

        if status in (401, 403):
            return LLMAuthenticationError(
                f"The server rejected the request (HTTP {status}). A local "
                f"Ollama server needs no key; if yours is behind an "
                f"authenticating proxy, set LLM_API_KEY in .env."
            )
        if status == 404 and "not found" in detail.lower():
            return LLMRequestError(
                f"Model '{self._config.model}' is not installed. "
                f"{ollama_pull_hint(self._config.model)}"
            )
        if status == 429:
            return LLMRateLimitError(message, retry_after=self._retry_after(response))
        if status >= 500:
            return LLMServerError(message, retry_after=self._retry_after(response))
        return LLMRequestError(message)

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            error = response.json().get("error") or {}
            detail = str(error.get("message") or "")
        except (ValueError, AttributeError):
            detail = response.text or ""
        return detail[:_MAX_ERROR_DETAIL] or "no detail"

    @staticmethod
    def _retry_after(response: httpx.Response) -> Optional[float]:
        value = response.headers.get("retry-after")
        try:
            return max(0.0, float(value)) if value is not None else None
        except ValueError:
            return None

    def close(self) -> None:
        self._client.close()
