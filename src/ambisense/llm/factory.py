"""Build the configured provider - the single place that maps names to classes."""

from __future__ import annotations

from typing import Callable, Optional

from ambisense.config import IMPLEMENTED_PROVIDERS, LLMConfig, Settings
from ambisense.llm.adjudicator import LLMAdjudicator
from ambisense.llm.cache import ResponseCache
from ambisense.llm.errors import LLMConfigurationError
from ambisense.llm.providers.base import LLMProvider
from ambisense.llm.providers.openai_compatible import OpenAICompatibleProvider

#: provider name -> constructor. Every name here must also appear in
#: ``config.IMPLEMENTED_PROVIDERS``; a test enforces that the two agree.
PROVIDER_BUILDERS: dict[str, Callable[[LLMConfig], LLMProvider]] = {
    "ollama": lambda config: OpenAICompatibleProvider(
        config, config.auth_token, provider_name="ollama"
    ),
}


def build_provider(config: LLMConfig) -> LLMProvider:
    """Return the configured provider.

    No API key is required: the model is served locally. Whether the server is
    actually running is discovered on the first request (or up front with
    ``llm.health.check_server``).
    """
    builder = PROVIDER_BUILDERS.get(config.provider)
    if builder is None:
        raise LLMConfigurationError(
            f"LLM provider {config.provider!r} is not implemented "
            f"(implemented: {', '.join(IMPLEMENTED_PROVIDERS)})"
        )
    return builder(config)


def build_adjudicator(
    settings: Settings,
    provider: Optional[LLMProvider] = None,
    *,
    use_provider_from_config: bool = True,
) -> LLMAdjudicator:
    """Assemble an adjudicator from settings.

    Tests pass ``provider`` explicitly (a fake) and
    ``use_provider_from_config=False`` so no real provider is built.
    """
    if provider is None and use_provider_from_config:
        provider = build_provider(settings.llm)
    cache = (
        ResponseCache(settings.resolve_path(settings.llm.cache_dir))
        if settings.llm.enable_cache else None
    )
    return LLMAdjudicator(settings.llm, settings.adjudication, provider, cache=cache)
