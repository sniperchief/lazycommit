"""The CLI must never change the index, working tree, refs, or history."""

from __future__ import annotations

import os
import subprocess
from typing import Any

import pytest
from conftest import FAKE_AWS_KEY, FakeOllama, Repo, group, plan_reply, run_cli, snapshot

from lazycommit import gitrepo


def _messy(repo: Repo) -> None:
    repo.write("LICENSE", "unchanged tracked file\n")
    repo.commit_all("add license")
    repo.write("src/app.py", "def main():\n    return 2\n")
    repo.git("add", "src/app.py")
    repo.write("src/app.py", "def main():\n    return 3\n")
    repo.write("README.md", "# changed\n")
    repo.write("new/file.txt", "new\n")
    repo.write("src/config.py", f'KEY = "{FAKE_AWS_KEY}"\n')
    repo.write(".env", "A=b\n")


@pytest.mark.parametrize("extra_args", [[], ["--show-input"]])
def test_repository_state_unchanged(repo: Repo, fake_ollama: FakeOllama, extra_args: list[str]) -> None:
    _messy(repo)
    fake_ollama.reply = plan_reply(group(["src/app.py", "README.md", "new/file.txt"], "feat", "x"))
    # The test's own git calls may refresh the index, so run them before the snapshot.
    status_before = repo.git("status", "--porcelain=v2")
    log_before = repo.git("log", "--all", "--format=%H")
    # Make the index's cached stat info stale (same content, new mtime) so a
    # careless `git status` would rewrite .git/index and fail this test.
    unchanged = repo.root / "LICENSE"
    st = unchanged.stat()
    os.utime(unchanged, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    before = snapshot(repo.root)

    code, _ = run_cli(repo, *extra_args, host=fake_ollama.url)

    assert code == 0
    assert snapshot(repo.root) == before  # every byte of .git and worktree
    assert repo.git("status", "--porcelain=v2") == status_before
    assert repo.git("log", "--all", "--format=%H") == log_before


def test_state_unchanged_on_model_failure(repo: Repo, fake_ollama: FakeOllama) -> None:
    _messy(repo)
    fake_ollama.reply = "garbage"
    before = snapshot(repo.root)
    code, _ = run_cli(repo, host=fake_ollama.url)
    assert code == 1
    assert snapshot(repo.root) == before


def test_state_unchanged_during_merge_conflict(repo: Repo) -> None:
    repo.git("checkout", "-q", "-b", "other")
    repo.write("README.md", "# other\n")
    repo.commit_all("other")
    repo.git("checkout", "-q", "main")
    repo.write("README.md", "# main\n")
    repo.commit_all("main")
    repo.git("merge", "other")
    before = snapshot(repo.root)
    code, _ = run_cli(repo, host="http://127.0.0.1:9")
    assert code == 3
    assert snapshot(repo.root) == before


def test_only_read_only_git_commands_are_executed(
    repo: Repo, fake_ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    _messy(repo)
    fake_ollama.reply = plan_reply()
    calls: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(list(cmd))
        assert kwargs.get("shell") is False
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    run_cli(repo, host=fake_ollama.url)

    assert calls, "expected git to be invoked"
    for cmd in calls:
        assert cmd[0] == "git" and "--no-optional-locks" in cmd
        sub = next(a for a in cmd[1:] if not a.startswith("-") and "=" not in a)
        assert sub in gitrepo.READ_ONLY_SUBCOMMANDS, cmd
        if sub == "diff":
            assert "--no-ext-diff" in cmd and "--no-textconv" in cmd
