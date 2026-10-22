"""Phase 3, local parts: branch verification, PR drafting, outgoing secret scan, gh detection.

Everything here is real: real Git repositories, a real bare repository as the
remote, the real `gh` binary (or its real absence). Nothing is pushed to
GitHub; the remote-publishing tests live in test_pr_github.py.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FAKE_AWS_KEY, FAKE_ENV_VALUE, Repo, Script, run_cmd, snapshot

from lazycommit import branch, cli, github, prwriter
from lazycommit.branch import BranchError
from lazycommit.gitrepo import GitError, run_git

SANDBOX_URL = "https://github.com/sniperchief/lazycommit-sandbox.git"


def _feature(repo: Repo, name: str = "feature/retry-logic") -> None:
    repo.git("switch", "-q", "-c", name)
    repo.write("src/app.py", "def main():\n    return retry(2)\n\n\ndef retry(n):\n    return n\n")
    repo.commit_all("feat(app): add retry helper")
    repo.write("tests/test_app.py", "from src.app import retry\n\n\ndef test_retry():\n    assert retry(1) == 1\n")
    repo.git("add", "-A")
    repo.git("commit", "-q", "-m", "test: cover retry helper", "-m", "Retries were untested, which hid a bug.")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


# --- verification of repository, branch, remote, base ---------------------------


def test_no_remote(repo: Repo) -> None:
    _feature(repo)
    code, out = run_cmd(repo, "pr", "--no-model")
    assert code == 1
    assert "no Git remote is configured" in out and "git remote add origin" in out


def test_detached_head(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    repo.git("switch", "-q", "--detach", "HEAD")
    code, out = run_cmd(repo, "pr", "--no-model")
    assert code == 1 and "HEAD is detached" in out


def test_missing_base_branch(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    code, out = run_cmd(repo, "pr", "--no-model", "--base", "release")
    assert code == 1
    assert "base branch 'release' was not found" in out and "git fetch origin" in out


def test_on_base_branch(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    code, out = run_cmd(repo, "pr", "--no-model")
    assert code == 1 and "you are on the base branch 'main'" in out


def test_no_unique_commits(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    repo.git("switch", "-q", "-c", "empty-branch")
    code, out = run_cmd(repo, "pr", "--no-model")
    assert code == 1
    assert "has no commits that are not already in 'main'" in out


def test_commits_with_no_net_change(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    repo.git("switch", "-q", "-c", "undo")
    repo.write("README.md", "# changed\n")
    repo.commit_all("docs: change readme")
    repo.git("revert", "--no-edit", "HEAD")
    code, out = run_cmd(repo, "pr", "--no-model")
    assert code == 1 and "make no net change" in out


def test_invalid_base_value_is_rejected(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    code, out = run_cmd(repo, "pr", "--no-model", "--base=--upload-pack=evil")
    assert code == 2 and "invalid --base value" in out


def test_unsafe_state_is_refused(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    repo.git("switch", "-q", "main")
    repo.write("src/app.py", "conflict\n")
    repo.commit_all("main change")
    repo.git("switch", "-q", "feature/retry-logic")
    repo.git("merge", "main")  # conflicts
    code, out = run_cmd(repo, "pr", "--no-model")
    assert code == 3 and "merge in progress" in out


def test_identifies_only_branch_commits(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    repo.git("switch", "-q", "-c", "feature")
    repo.write("a.txt", "a\n")
    repo.commit_all("feat: add a")
    # main moves on after the branch point (and is pushed); those commits are not ours.
    repo.git("switch", "-q", "main")
    repo.write("main-only.txt", "m\n")
    repo.commit_all("chore: main-only work")
    repo.git("push", "-q", "origin", "main")
    repo.git("switch", "-q", "feature")
    repo.write("b.txt", "b\n")
    repo.commit_all("feat: add b")

    info = branch.resolve(repo.root)
    assert [c.subject for c in info.commits] == ["feat: add a", "feat: add b"]
    assert sorted(f.path for f in info.files) == ["a.txt", "b.txt"]
    assert info.base == "main" and info.base_ref == "refs/remotes/origin/main"


def test_remote_base_preferred_over_stale_local_base(remote_repo: tuple[Repo, Path]) -> None:
    repo, bare = remote_repo
    # Someone else advanced origin/main; local main is stale.
    other = Repo(bare.parent / "other")
    subprocess.run(["git", "clone", "-q", str(bare), str(other.root)], check=True, capture_output=True)
    other.git("config", "user.name", "O")
    other.git("config", "user.email", "o@example.invalid")
    other.write("upstream.txt", "u\n")
    other.commit_all("feat: upstream work")
    other.git("push", "-q", "origin", "main")
    repo.git("fetch", "-q", "origin")
    repo.git("switch", "-q", "-c", "feature", "origin/main")
    repo.write("mine.txt", "x\n")
    repo.commit_all("feat: mine")
    info = branch.resolve(repo.root)
    assert [c.subject for c in info.commits] == ["feat: mine"]


# --- drafting -------------------------------------------------------------------


def test_preview_fallback_is_truthful_and_changes_nothing(remote_repo: tuple[Repo, Path]) -> None:
    repo, bare = remote_repo
    _feature(repo)
    repo.git("status")
    before, before_remote = snapshot(repo.root), snapshot(bare)

    code, out = run_cmd(repo, "pr", "--no-model")

    assert code == 0, out
    assert "preview only - nothing will be pushed or published" in out
    assert "Retry logic" in out  # title from the branch name (2 commits)
    assert "- feat(app): add retry helper" in out and "- test: cover retry helper" in out
    assert "Lazy-Commit did not run tests and has no record of test results" in out
    assert "Test files changed in this branch: `tests/test_app.py`" in out
    assert "passed" not in out.lower()
    assert "Nothing was pushed and no pull request was created" in out
    assert snapshot(repo.root) == before
    assert snapshot(bare) == before_remote


def test_model_unavailable_falls_back(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    code, out = run_cmd(repo, "pr", "--model", "anything", "--ollama-host", f"http://127.0.0.1:{_free_port()}")
    assert code == 0
    assert "cannot reach Ollama" in out
    assert "Showing a description built only from commit messages" in out
    assert "deterministic description used" in out
    assert "## Summary\n- feat(app): add retry helper" in out


def test_no_model_configured_falls_back(remote_repo: tuple[Repo, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAZYCOMMIT_MODEL", raising=False)
    repo, _ = remote_repo
    _feature(repo)
    code, out = run_cmd(repo, "pr", "--ollama-host", f"http://127.0.0.1:{_free_port()}")
    assert code == 0 and "## Summary" in out


def test_json_preview(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    out, err = io.StringIO(), io.StringIO()
    code = cli.run(["pr", "--repo", str(repo.root), "--no-model", "--json"], out=out, err=err)
    data = json.loads(out.getvalue())
    assert code == 0
    assert data["published"] is False and data["pushed"] is False
    assert data["source"] == "fallback"
    assert [c["subject"] for c in data["commits"]] == ["feat(app): add retry helper", "test: cover retry helper"]
    assert "## Testing" in data["body"]


def _info(repo: Repo) -> branch.BranchInfo:
    return branch.resolve(repo.root)


def test_model_output_cannot_invent_tests_or_files(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    raw = json.dumps({
        "title": "Add retry helper",
        "summary": ["Add a retry helper to src/app.py", "All tests pass on CI",
                    "Update src/ghost.py to use retries", "Cover retry with tests"],
        "motivation": "Retries were untested.",
        "details": ["Tested manually against production", "retry() returns its argument"],
        "limitations": [],
    })
    info = _info(repo)
    draft = prwriter.parse_description(raw, info, branch.collect_pr(info))
    assert draft.summary == ["Add a retry helper to src/app.py", "Cover retry with tests"]
    assert draft.details == ["retry() returns its argument"]
    assert draft.motivation == "Retries were untested."  # a commit body explains it
    assert any("testing" in w for w in draft.warnings)
    assert any("src/ghost.py" in w for w in draft.warnings)


def test_motivation_dropped_without_evidence(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    repo.git("switch", "-q", "-c", "f")
    repo.write("a.txt", "a\n")
    repo.commit_all("feat: add a")  # no body: no stated motivation
    raw = json.dumps({"title": "Add a", "summary": ["Add a.txt"], "motivation": "Users asked for it.",
                      "details": [], "limitations": []})
    info = _info(repo)
    draft = prwriter.parse_description(raw, info, branch.collect_pr(info))
    assert draft.motivation == ""
    assert any("no commit message explains why" in w for w in draft.warnings)


def test_limitations_need_evidence(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    repo.git("switch", "-q", "-c", "f")
    repo.write("a.py", "def a():\n    return 1\n")
    repo.commit_all("feat: add a")
    raw = json.dumps({"title": "Add a", "summary": ["Add a()"], "motivation": "", "details": [],
                      "limitations": ["a() is not thread-safe"]})
    info = _info(repo)
    draft = prwriter.parse_description(raw, info, branch.collect_pr(info))
    assert draft.limitations == []
    assert any("no commit message or added line documents one" in w for w in draft.warnings)

    repo.write("a.py", "def a():\n    # TODO: handle negative input\n    return 1\n")
    repo.commit_all("feat: document a gap")
    raw = raw.replace("a() is not thread-safe", "Negative input is not handled yet")
    info = _info(repo)
    draft = prwriter.parse_description(raw, info, branch.collect_pr(info))
    assert draft.limitations == ["Negative input is not handled yet"]


@pytest.mark.parametrize("raw,error", [
    ("not json", "not valid JSON"),
    ('["a"]', "not a JSON object"),
    ('{"summary": ["x"], "motivation": "", "details": [], "limitations": []}', "no title"),
    ('{"title": "t", "summary": "x", "motivation": "", "details": [], "limitations": []}', "must be a list"),
    ('{"title": "t", "summary": ["All tests pass"], "motivation": "", "details": [], "limitations": []}',
     "no usable summary"),
])
def test_malformed_model_output_rejected(remote_repo: tuple[Repo, Path], raw: str, error: str) -> None:
    repo, _ = remote_repo
    _feature(repo)
    info = _info(repo)
    with pytest.raises(prwriter.InvalidDescriptionError, match=error):
        prwriter.parse_description(raw, info, branch.collect_pr(info))


def test_model_cannot_describe_files_it_was_not_shown(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    repo.write("src/settings.py", f'AWS_KEY = "{FAKE_AWS_KEY}"\n')
    repo.commit_all("chore: add settings")
    info = _info(repo)
    pr = branch.collect_pr(info)
    assert prwriter.hidden_paths(info, pr) == ["src/settings.py"]
    raw = json.dumps({
        "title": "Add retry helper",
        "summary": ["Add a retry helper", "Introduce a settings file for retry configuration"],
        "motivation": "", "details": ["src/settings.py configures retries"], "limitations": [],
    })
    draft = prwriter.parse_description(raw, info, pr)
    assert draft.summary == ["Add a retry helper"] and draft.details == []
    assert any("not shown" in w for w in draft.warnings)
    body = prwriter.render_body(draft, info, pr)
    assert "1 changed file(s) were not analyzed by Lazy-Commit" in body
    assert "withheld. Do not describe them" in prwriter.build_prompt(info, pr)


# --- secrets ---------------------------------------------------------------------


def test_secrets_never_reach_the_model_input(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    repo.write("src/config.py", f'AWS_ACCESS_KEY_ID = "{FAKE_AWS_KEY}"\n')
    repo.write(".env", f"DB_PASSWORD={FAKE_ENV_VALUE}\n")
    repo.commit_all("chore: add config")
    repo.git("commit", "-q", "--allow-empty", "-m", f"chore: rotate key {FAKE_AWS_KEY}")

    code, out = run_cmd(repo, "pr", "--show-input")

    assert code == 0
    model_input = out.split("BEGIN SANITIZED MODEL INPUT")[1]
    assert FAKE_AWS_KEY not in out and FAKE_ENV_VALUE not in out
    assert "FILE: src/config.py" not in model_input and "FILE: .env" not in model_input
    assert "src/config.py: possible secret: AWS access key ID" in out
    assert ".env: sensitive file type" in out
    assert "(message withheld)" in model_input


def test_secret_added_then_removed_is_caught_in_outgoing_history(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    repo.write("src/settings.py", f'KEY = "{FAKE_AWS_KEY}"\n')
    repo.commit_all("chore: add settings")
    repo.write("src/settings.py", 'KEY = os.environ["KEY"]\n')
    repo.commit_all("chore: read key from env")

    info = _info(repo)
    pr = branch.collect_pr(info)
    assert "src/settings.py" in pr.paths  # the net diff is clean...
    scan = branch.scan_outgoing(info)  # ...but the history is not
    assert any("src/settings.py contains a possible AWS access key ID" in p for p in scan.problems)
    assert all(FAKE_AWS_KEY not in p for p in scan.problems)

    code, out = run_cmd(repo, "pr", "--no-model")
    assert code == 0
    assert "Possible secrets in the 4 commit(s) that are not on the remote yet" in out
    assert "`--publish` will refuse" in out and FAKE_AWS_KEY not in out


def test_commit_message_secret_is_withheld_from_description(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    repo.git("commit", "-q", "--allow-empty", "-m", f"chore: use {FAKE_AWS_KEY}")
    code, out = run_cmd(repo, "pr", "--no-model")
    assert FAKE_AWS_KEY not in out
    assert "(message withheld: matched a secret pattern)" in out


# --- publish preconditions ----------------------------------------------------------


def test_publish_requires_a_terminal(remote_repo: tuple[Repo, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = remote_repo
    _feature(repo)
    monkeypatch.setattr(sys, "stdin", io.StringIO("c\npush\ncreate\n"))
    out = io.StringIO()
    code = cli.run(["pr", "--repo", str(repo.root), "--publish", "--no-model"], out=out)
    assert code == 2 and "--publish needs an interactive terminal" in out.getvalue()


def test_publish_refuses_non_github_remote(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    _feature(repo)
    code, out = run_cmd(repo, "pr", "--publish", "--no-model", prompt=Script([]))
    assert code == 1 and "does not point to a github.com repository" in out


def test_gh_not_installed(remote_repo: tuple[Repo, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    repo, bare = remote_repo
    _feature(repo)
    repo.git("remote", "set-url", "origin", SANDBOX_URL)
    git_dir = os.path.dirname(shutil.which("git") or "")
    monkeypatch.delenv("LAZYCOMMIT_GH", raising=False)
    monkeypatch.setenv("PATH", git_dir)  # a real environment where gh is absent
    assert github.find_gh() is None
    script = Script(["c", "push", "create"])
    code, out = run_cmd(repo, "pr", "--publish", "--no-model", prompt=script)
    assert code == 1
    assert "GitHub CLI (gh) is not installed" in out and "winget install --id GitHub.cli" in out
    assert "Nothing was pushed and no pull request was created" in out
    assert script.questions == []


@pytest.mark.skipif(github.find_gh() is None, reason="GitHub CLI not installed")
def test_gh_not_logged_in(remote_repo: tuple[Repo, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo, _ = remote_repo
    _feature(repo)
    repo.git("remote", "set-url", "origin", SANDBOX_URL)
    empty = tmp_path / "gh-config"
    empty.mkdir()
    monkeypatch.setenv("LAZYCOMMIT_GH", github.find_gh() or "")
    monkeypatch.setenv("GH_CONFIG_DIR", str(empty))  # real gh with no stored login
    for var in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    script = Script(["c", "push", "create"])
    code, out = run_cmd(repo, "pr", "--publish", "--no-model", prompt=script)
    assert code == 1
    assert "not logged in to github.com" in out and "gh auth login" in out
    assert "gho_" not in out and script.questions == []


# --- guardrails -------------------------------------------------------------------


def test_push_command_never_forces() -> None:
    cmd = github.push_command("origin", "feature/x")
    assert cmd[-2:] == ["origin", "refs/heads/feature/x:refs/heads/feature/x"]
    assert not any(a in cmd for a in ("--force", "-f", "--force-with-lease", "--mirror", "--all", "--tags"))
    assert not any(a.startswith("+") for a in cmd)
    for bad in ("+main", "a:b", "-f", "x..y", "a~1"):
        with pytest.raises(github.GitHubError):
            github.push_command("origin", bad)


def test_read_only_git_wrapper_still_refuses_writes(repo: Repo) -> None:
    for args in (("push", "origin"), ("config", "user.name", "x"), ("config", "--unset", "user.name"),
                 ("remote", "add", "x", "y"), ("fetch",)):
        with pytest.raises(GitError):
            run_git(repo.root, *args)
    assert run_git(repo.root, "config", "--get", "user.name").strip() == b"Test User"


def test_github_url_parsing() -> None:
    assert branch.parse_github_url("https://github.com/acme/widget.git") == ("acme", "widget")
    assert branch.parse_github_url("git@github.com:acme/widget.git") == ("acme", "widget")
    assert branch.parse_github_url("https://user:tok@github.com/acme/widget") == ("acme", "widget")
    assert branch.parse_github_url("https://gitlab.com/acme/widget.git") is None
    assert branch.parse_github_url("https://github.com.evil.io/acme/widget") is None


def test_branch_errors_carry_hints(remote_repo: tuple[Repo, Path]) -> None:
    repo, _ = remote_repo
    with pytest.raises(BranchError) as info:
        branch.resolve(repo.root)
    assert "git switch -c" in info.value.hint
