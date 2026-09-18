"""On-disk cache of validated LLM replies.

Why it exists: an identical request (same provider, model, settings and exact
prompt text) at temperature 0 should get the same answer, so paying for it
twice is waste - and a viva demo re-runs the same sentences many times.

Two rules keep it honest:

* The key is a SHA-256 of **everything** that shapes the reply, including the
  full prompt text. Editing a prompt file therefore changes the key, and a
  stale answer from an older prompt can never be served.
* Only replies that passed validation are stored. Caching a malformed reply
  would make a transient failure permanent.

The API key is never part of the key or the stored entry. ``data/cache`` is
git-ignored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from ambisense.config import LLMConfig
from ambisense.llm.providers.base import LLMRequest
from ambisense.logging_setup import get_logger

logger = get_logger(__name__)


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key_for(request: LLMRequest, config: LLMConfig, provider_name: str) -> str:
        material = json.dumps(
            {
                "provider": provider_name,
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "json_mode": request.json_mode and config.request_json_mode,
                "messages": [[m.role, m.content] for m in request.messages],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return str(json.loads(path.read_text(encoding="utf-8"))["content"])
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Ignoring unreadable cache entry %s", path.name)
            return None

    def put(self, key: str, content: str) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(
                json.dumps({"content": content}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            # A cache that cannot be written must never break an analysis.
            logger.warning("Could not write LLM cache entry: %s", exc)
