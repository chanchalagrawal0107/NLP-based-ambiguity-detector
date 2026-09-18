"""Is the local Ollama server up, and is the configured model installed?

Used by ``--check-config`` and ``--live-llm-test`` so a missing server or model
is reported clearly *before* an analysis starts, instead of surfacing as a
connection error halfway through a demo.

This uses Ollama's native ``GET /api/tags`` endpoint, which lists installed
models. It is separate from the provider because it is a diagnostic, not part
of the adjudication path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import httpx

from ambisense.config import LLMConfig


def ollama_serve_hint() -> str:
    """Remediation text for an unreachable Ollama server."""
    return "Start it with: ollama serve"


def ollama_pull_hint(model: str) -> str:
    """Remediation text for a server reachable but missing the model."""
    return f"Run: ollama pull {model}"


@dataclass(frozen=True)
class ServerStatus:
    reachable: bool
    model_installed: bool = False
    installed_models: tuple[str, ...] = field(default_factory=tuple)
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.reachable and self.model_installed


def _server_root(base_url: str) -> str:
    """``http://localhost:11434/v1`` -> ``http://localhost:11434``."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def check_server(
    config: LLMConfig,
    *,
    timeout: float = 5.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> ServerStatus:
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.get(f"{_server_root(config.base_url)}/api/tags")
            response.raise_for_status()
            models = tuple(
                sorted(str(item.get("name", "")) for item in response.json().get("models", []))
            )
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        return ServerStatus(
            reachable=False,
            error=f"{type(exc).__name__}: is Ollama running at {config.base_url}?",
        )
    return ServerStatus(
        reachable=True,
        model_installed=config.model in models,
        installed_models=models,
    )
