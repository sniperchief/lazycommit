"""Optional end-to-end test against a real local Ollama model.

Enable with:  LAZYCOMMIT_TEST_MODEL=qwen2.5-coder:7b pytest -m ollama
"""

from __future__ import annotations

import os

import pytest
from conftest import FAKE_AWS_KEY, Repo, run_cli, snapshot

from lazycommit.ollama import DEFAULT_HOST, ModelError, OllamaClient

MODEL = os.environ.get("LAZYCOMMIT_TEST_MODEL")
HOST = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)


def _ollama_ready() -> str | None:
    if not MODEL:
        return "set LAZYCOMMIT_TEST_MODEL to run"
    try:
        OllamaClient(HOST, timeout=5).ensure_model(MODEL)
    except ModelError as exc:
        return f"Ollama not ready: {exc}"
    return None


@pytest.mark.ollama
def test_real_model_produces_valid_proposal(repo: Repo) -> None:
    reason = _ollama_ready()
    if reason:
        pytest.skip(reason)
    repo.write("src/app.py", "def main():\n    return compute(2)\n\n\ndef compute(x):\n    return x * 21\n")
    repo.write("docs/usage.md", "# Usage\n\nRun `python -m app`.\n")
    repo.write("src/settings.py", f'AWS_KEY = "{FAKE_AWS_KEY}"\n')
    before = snapshot(repo.root)

    code, out = run_cli(repo, "--timeout", "600", host=HOST, model=MODEL)

    print(out)
    assert code == 0, out
    assert "PROPOSED COMMIT PLAN" in out
    assert "src/settings.py: possible secret" in out
    assert FAKE_AWS_KEY not in out
    assert snapshot(repo.root) == before
