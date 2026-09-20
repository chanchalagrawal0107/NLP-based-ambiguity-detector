"""CLI mode dispatch: every advertised flag must reach its command handler.

``--evaluate`` was once parsed but never dispatched, so it fell through to the
"No text supplied" message. These tests pin each opt-in command to its handler
without making any LLM call (the handlers are replaced by stubs).
"""

from __future__ import annotations

import pytest

import main as cli
from ambisense.llm.health import ServerStatus


@pytest.mark.parametrize(
    "flag, handler",
    [
        ("--evaluate", "command_evaluate"),
        ("--live-llm-test", "command_live_llm_test"),
        ("--check-config", "command_check_config"),
    ],
)
def test_textless_flags_reach_their_handler(monkeypatch, flag, handler):
    calls = []
    monkeypatch.setattr(cli, handler, lambda settings: calls.append(flag) or 0)

    assert cli.main([flag]) == 0
    assert calls == [flag]


def test_evaluate_refuses_without_ollama_server(monkeypatch, capsys):
    """With no reachable server, --evaluate exits with the config-error code
    and runs nothing."""
    monkeypatch.setattr(
        cli, "check_server", lambda config: ServerStatus(reachable=False)
    )
    assert cli.main(["--evaluate"]) == cli.EXIT_CONFIG_ERROR
    assert "--evaluate cannot run" in capsys.readouterr().err
