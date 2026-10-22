"""Phase 3 publishing against a REAL GitHub repository (no mocks).

These tests push real branches and open real pull requests, then close them
and delete the branches. They only run when you name a throwaway repository:

    set LAZYCOMMIT_GITHUB_SANDBOX=sniperchief/lazycommit-sandbox
    pytest -m github

Requires `gh` (logged in) and git credentials that can push to it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import FAKE_AWS_KEY, Repo, Script, run_cmd

from lazycommit.github import find_gh

SANDBOX = os.environ.get("LAZYCOMMIT_GITHUB_SANDBOX", "")
GH = find_gh()

pytestmark = [
    pytest.mark.github,
    pytest.mark.skipif(not SANDBOX or GH is None,
                       reason="set LAZYCOMMIT_GITHUB_SANDBOX=owner/repo and install gh to run"),
]


def gh(*args: str) -> str:
    assert GH
    proc = subprocess.run([GH, *args], capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise AssertionError(f"gh {args[:2]} failed: {proc.stderr.strip()}")
    return proc.stdout


def open_prs(branch: str) -> list[dict[str, object]]:
    rows = json.loads(gh("pr", "list", "--repo", SANDBOX, "--head", branch, "--state", "open",
                         "--json", "number,url,title,body,commits"))
    assert isinstance(rows, list)
    return rows


def remote_oid(repo: Repo, branch: str) -> str | None:
    # Plain git with the user's normal config (the credential helper may be system-level).
    proc = subprocess.run(["git", "ls-remote", "--heads", "origin", f"refs/heads/{branch}"],
                          cwd=repo.root, capture_output=True, text=True, check=True)
    out = proc.stdout.split()
    return out[0] if out else None


@pytest.fixture
def sandbox(tmp_path: Path) -> Iterator[tuple[Repo, str]]:
    root = tmp_path / "sandbox"
    subprocess.run(["git", "clone", "-q", "-c", "core.autocrlf=false",
                    f"https://github.com/{SANDBOX}.git", str(root)], check=True, capture_output=True)
    repo = Repo(root)
    repo.git("config", "user.name", "Lazy-Commit Test")
    repo.git("config", "user.email", "test@example.invalid")
    repo.git("config", "commit.gpgsign", "false")
    name = f"lc-test-{uuid.uuid4().hex[:8]}"
    repo.git("switch", "-q", "-c", name)
    try:
        yield repo, name
    finally:
        for pr in open_prs(name):
            gh("pr", "close", str(pr["number"]), "--repo", SANDBOX, "--delete-branch")
        subprocess.run(["git", "push", "-q", "origin", "--delete", name], cwd=root, capture_output=True)


def _work(repo: Repo, n: int = 1) -> None:
    for i in range(n):
        repo.write(f"notes/{uuid.uuid4().hex[:6]}.md", f"note {i}\n")
        repo.commit_all(f"docs: add note {i}")


def _publish(repo: Repo, script: Script, *extra: str) -> tuple[int, str]:
    return run_cmd(repo, "pr", "--publish", "--no-model", *extra, prompt=script)


def test_publish_end_to_end(sandbox: tuple[Repo, str]) -> None:
    repo, name = sandbox
    _work(repo)

    code, out = _publish(repo, Script(["c", "push", "create"]))

    assert code == 0, out
    url = re.search(r"Created pull request: (https://github\.com/\S+/pull/\d+)", out)
    assert url, out
    assert remote_oid(repo, name) == repo.git("rev-parse", "HEAD").strip()
    view = json.loads(gh("pr", "view", url.group(1), "--json", "state,title,body,headRefName,baseRefName"))
    assert view["state"] == "OPEN" and view["headRefName"] == name and view["baseRefName"] == "main"
    assert view["title"] == "docs: add note 0"
    assert "Lazy-Commit did not run tests and has no record of test results" in view["body"]


def test_existing_pr_is_never_duplicated_or_overwritten(sandbox: tuple[Repo, str]) -> None:
    repo, name = sandbox
    _work(repo)
    code, out = _publish(repo, Script(["c", "push", "create"]))
    assert code == 0, out
    original = open_prs(name)[0]

    # Up to date: explains the options and stops.
    code, out = _publish(repo, Script([]))
    assert code == 0
    assert f"An open pull request already exists for '{name}'" in out
    assert "will not create a second pull request or change this one" in out
    assert "already up to date" in out

    # New local commit: pushing needs explicit approval and only adds commits.
    _work(repo)
    code, out = _publish(repo, Script(["push"]))
    assert code == 0, out
    assert "now includes your latest commits" in out
    prs = open_prs(name)
    assert len(prs) == 1
    assert prs[0]["title"] == original["title"] and prs[0]["body"] == original["body"]
    commits = prs[0]["commits"]
    assert isinstance(commits, list) and len(commits) == 2


def test_declining_push_publishes_nothing(sandbox: tuple[Repo, str]) -> None:
    repo, name = sandbox
    _work(repo)
    code, out = _publish(repo, Script(["c", "no"]))
    assert code == 0
    assert "Not pushed. No pull request was created." in out
    assert remote_oid(repo, name) is None and open_prs(name) == []


def test_declining_create_leaves_only_the_push(sandbox: tuple[Repo, str]) -> None:
    repo, name = sandbox
    _work(repo)
    code, out = _publish(repo, Script(["c", "push", "nope"]))
    assert code == 0
    assert "Not created. The branch was pushed; nothing else changed." in out
    assert remote_oid(repo, name) is not None and open_prs(name) == []


def test_edited_title_and_body_are_what_gets_published(sandbox: tuple[Repo, str]) -> None:
    repo, name = sandbox
    _work(repo)

    def edit_body(question: str) -> None:
        path = re.search(r"Edit the description in (.+?), save it", question)
        assert path
        Path(path.group(1)).write_text("## Summary\n- Hand-written description.\n", encoding="utf-8",
                                       newline="\n")

    script = Script(["t", "Custom title from review", "b", "", "c", "push", "create"],
                    before={"Edit the description in": edit_body})
    code, out = _publish(repo, script)
    assert code == 0, out
    pr = open_prs(name)[0]
    assert pr["title"] == "Custom title from review"
    body = pr["body"]
    assert isinstance(body, str) and "\r" not in body  # no CRLF mangling on the way to GitHub
    assert body.strip() == "## Summary\n- Hand-written description."


def test_branch_change_before_create_stops(sandbox: tuple[Repo, str]) -> None:
    repo, name = sandbox
    _work(repo)
    script = Script(["c", "push", "create"], before={"Type 'create'": lambda _: _work(repo)})
    code, out = _publish(repo, script)
    assert code == 4
    assert "the branch changed since the preview; no pull request was created" in out
    assert "(the branch was pushed to origin)" in out
    assert open_prs(name) == []


def test_outgoing_secret_blocks_push(sandbox: tuple[Repo, str]) -> None:
    repo, name = sandbox
    repo.write("settings.py", f'KEY = "{FAKE_AWS_KEY}"\n')
    repo.commit_all("chore: add settings")
    repo.write("settings.py", 'KEY = os.environ["KEY"]\n')
    repo.commit_all("chore: read key from env")
    script = Script(["c", "push", "create"])

    code, out = _publish(repo, script)

    assert code == 1
    assert "possible secrets would be pushed" in out and FAKE_AWS_KEY not in out
    assert script.questions == []
    assert remote_oid(repo, name) is None and open_prs(name) == []


def test_secret_typed_into_title_blocks_publishing(sandbox: tuple[Repo, str]) -> None:
    repo, name = sandbox
    _work(repo)
    code, out = _publish(repo, Script(["t", f"use {FAKE_AWS_KEY}", "c", "push", "create"]))
    assert code == 1
    assert "matches a secret pattern" in out and FAKE_AWS_KEY not in out
    assert remote_oid(repo, name) is None


def _redirect_origin(repo: Repo, tmp_path: Path, reject: bool) -> None:
    """Keep the GitHub URL but send Git traffic to a local bare copy (optionally rejecting pushes)."""
    bare = tmp_path / "redirect.git"
    subprocess.run(["git", "clone", "-q", "--bare", "--single-branch", "--branch", "main", str(repo.root), str(bare)],
                   check=True, capture_output=True)
    if reject:
        hook = bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\necho 'rejected by test hook' >&2\nexit 1\n", encoding="utf-8")
        hook.chmod(0o755)
    repo.git("config", f"url.{bare.as_posix()}.insteadOf", f"https://github.com/{SANDBOX}.git")


def test_failed_push_is_reported_truthfully(sandbox: tuple[Repo, str], tmp_path: Path) -> None:
    repo, name = sandbox
    _work(repo)
    _redirect_origin(repo, tmp_path, reject=True)
    code, out = _publish(repo, Script(["c", "push", "create"]))
    assert code == 1
    assert "git push failed" in out and "Nothing was force-pushed" in out
    assert "No pull request was created." in out
    assert open_prs(name) == []


def test_failed_pr_creation_is_reported_truthfully(sandbox: tuple[Repo, str], tmp_path: Path) -> None:
    repo, name = sandbox
    _work(repo)
    _redirect_origin(repo, tmp_path, reject=False)  # the push lands locally, so GitHub lacks the branch
    code, out = _publish(repo, Script(["c", "push", "create"]))
    assert code == 1, out
    assert "GitHub did not create the pull request" in out
    assert "(the branch was pushed to origin)" in out
    assert "Created pull request" not in out
    assert open_prs(name) == []
