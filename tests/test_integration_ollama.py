"""Optional end-to-end test against a real local Ollama model.

Enable with:  LAZYCOMMIT_TEST_MODEL=qwen2.5-coder:7b pytest -m ollama
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from conftest import FAKE_AWS_KEY, Repo, run_cli, run_cmd, snapshot

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


@pytest.mark.ollama
def test_real_model_drafts_truthful_pr_description(remote_repo: tuple[Repo, Path]) -> None:
    reason = _ollama_ready()
    if reason:
        pytest.skip(reason)
    repo, _ = remote_repo
    repo.git("switch", "-q", "-c", "feature/retry")
    repo.write("src/app.py", "import time\n\n\ndef main(retries=3):\n    for i in range(retries):\n"
                             "        try:\n            return 1\n        except OSError:\n"
                             "            time.sleep(2 ** i)\n")
    repo.commit_all("feat(app): retry main on OSError")
    repo.write("src/settings.py", f'AWS_KEY = "{FAKE_AWS_KEY}"\n')
    repo.commit_all("chore: add settings")

    code, out = run_cmd(repo, "pr", "--model", MODEL or "", "--timeout", "600")

    print(out)
    assert code == 0, out
    assert "src/settings.py: possible secret" in out and FAKE_AWS_KEY not in out
    body = out.split("PULL REQUEST DESCRIPTION")[1]
    testing = body.split("## Testing")[1].split("##")[0]
    assert "did not run tests" in testing
    without_testing = body.replace(testing, "")
    assert not re.search(r"(?i)tests? (?:pass|passed|ran)|all tests", without_testing)
