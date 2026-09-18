"""Suite-wide safety net: the test suite must never reach a real LLM API.

Every Phase 4 test uses a fake provider or ``httpx.MockTransport``. This
fixture guarantees it: any attempt to open a real HTTP connection - including
to a local Ollama server that happens to be running - fails the test
immediately, so the suite stays fast, deterministic and machine-independent.

``httpx.MockTransport`` is a different transport class, so the offline
provider tests are unaffected. Live checks belong to the opt-in
``python main.py --live-llm-test`` command, never to pytest.
"""

from __future__ import annotations

import httpx
import pytest


@pytest.fixture(autouse=True)
def _block_real_http(monkeypatch):
    def refuse(self, request):
        raise AssertionError(
            f"Test attempted a real HTTP request to {request.url.host}. "
            "Use a fake provider or httpx.MockTransport."
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
