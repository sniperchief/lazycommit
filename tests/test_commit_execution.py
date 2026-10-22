"""Phase 2: interactive approval and safe commit execution (temp repos, fake model)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import FAKE_AWS_KEY, FAKE_ENV_VALUE, FakeOllama, Repo, group, plan_reply, snapshot

from lazycommit import cli, execute, gitrepo
from lazycommit.collect import collect
from lazycommit.execute import APPROVED, BLOCKED, GroupResult, take_snapshot
from lazycommit.gitrepo import GitError, read_state
from lazycommit.planner import Group

APP_V2 = "def main():\n    return 2\n"


class Script:
    """Scripted terminal answers. Runs ``before_confirm`` just before the final question."""

    def __init__(self, answers: list[str], before_confirm: Callable[[], object] | None = None) -> None:
        self.answers = list(answers)
        self.before_confirm = before_confirm
        self.questions: list[str] = []

    def __call__(self, question: str) -> str:
        self.questions.append(question)
        if "Type 'yes'" in question and self.before_confirm:
            self.before_confirm()
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def review(repo: Repo, fake: FakeOllama, script: Script, *extra: str) -> tuple[int, str]:
    out = io.StringIO()
    argv = ["--repo", str(repo.root), "--ollama-host", fake.url, "--model", "fake-model", "--review", *extra]
    code = cli.run(argv, out=out, prompt=script)
    return code, out.getvalue()


def subjects(repo: Repo) -> list[str]:
    return repo.git("log", "--format=%s").splitlines()


def commit_files(repo: Repo, rev: str = "HEAD") -> set[str]:
    return set(repo.git("diff-tree", "--root", "--no-commit-id", "-r", "--name-only", rev).split())


def status_lines(repo: Repo) -> set[str]:
    return set(repo.git("status", "--porcelain=v1", "--untracked-files=all").splitlines())


def blob(repo: Repo, spec: str) -> str:
    return repo.git("rev-parse", spec).strip()


def _two_changes(repo: Repo) -> None:
    repo.write("src/app.py", APP_V2)
    repo.write("docs/usage.md", "usage\n")


def _two_groups() -> str:
    return plan_reply(
        group(["src/app.py"], "fix", "return 2 from main", scope="app"),
        group(["docs/usage.md"], "docs", "add usage notes"),
    )


# --- approval ---------------------------------------------------------------


def test_approve_one_group_and_reject_another(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.reply = _two_groups()

    code, out = review(repo, fake_ollama, Script(["y", "n", "yes"]))

    assert code == 0, out
    assert subjects(repo) == ["fix(app): return 2 from main", "initial"]
    assert commit_files(repo) == {"src/app.py"}
    assert status_lines(repo) == {"?? docs/usage.md"}
    head = blob(repo, "HEAD")
    assert f"[1] COMMITTED  {head[:12]}  fix(app): return 2 from main" in out
    assert "[2] REJECTED" in out
    assert "1 commit(s) created on 'main'. Nothing was pushed." in out


def test_review_shows_message_files_rationale_and_confidence(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.reply = plan_reply(
        group(["src/app.py"], "fix", "return 2", rationale="main now returns 2", confidence="low",
              notes="unsure about intent")
    )
    script = Script(["n"])
    code, out = review(repo, fake_ollama, script)
    assert code == 0
    assert "[1/1] PROPOSED fix: return 2" in out
    assert "- src/app.py  [unstaged]" in out
    assert "why: main now returns 2" in out
    assert "confidence: low" in out and "notes: unsure about intent" in out
    assert "-> REJECTED" in out
    assert script.questions == ["    Approve this commit? [y]es / [n]o / [q]uit review: "]


@pytest.mark.parametrize(
    "answers",
    [
        ["n", "n"],              # reject everything
        ["y", "y", "no"],        # approve, then decline the final confirmation
        ["y", "y", "y"],         # 'y' is not the explicit 'yes' confirmation
        ["y", "y"],              # input ends at the confirmation
        ["q"],                   # quit the review
        [],                      # input ends immediately
        ["maybe", "sure", "ok"],  # unrecognized answers never count as approval
    ],
)
def test_no_commits_without_explicit_approval(repo: Repo, fake_ollama: FakeOllama, answers: list[str]) -> None:
    _two_changes(repo)
    fake_ollama.reply = _two_groups()
    repo.git("status")  # let the test's git refresh the index before snapshotting
    before = snapshot(repo.root)

    code, out = review(repo, fake_ollama, Script(answers))

    assert code == 0, out
    assert "Nothing was committed" in out
    assert snapshot(repo.root) == before
    assert subjects(repo) == ["initial"]


def test_review_refuses_without_a_terminal(
    repo: Repo, fake_ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_changes(repo)
    fake_ollama.reply = _two_groups()
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\ny\nyes\n"))  # piped "approval"
    before = snapshot(repo.root)
    out = io.StringIO()
    code = cli.run(["--repo", str(repo.root), "--ollama-host", fake_ollama.url, "--model", "x", "--review"], out=out)
    assert code == 2
    assert "needs an interactive terminal" in out.getvalue()
    assert fake_ollama.requests == []
    assert snapshot(repo.root) == before


def test_json_preview_never_commits(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    repo.write("README.md", "# staged\n")
    repo.git("add", "README.md")
    repo.write("README.md", "# staged and unstaged\n")
    fake_ollama.reply = plan_reply(
        group(["src/app.py", "docs/usage.md"], "feat", "add usage"),
        group(["README.md"], "docs", "update readme"),
    )
    repo.git("status")
    before = snapshot(repo.root)
    out, err = io.StringIO(), io.StringIO()
    code = cli.run(["--repo", str(repo.root), "--ollama-host", fake_ollama.url, "--model", "fake-model",
                    "--json"], out=out, err=err)

    assert code == 0
    data = json.loads(out.getvalue())
    assert data["committed"] is False and data["status"] == "proposed"
    first, second = data["groups"]
    assert first["header"] == "feat: add usage" and first["commit_safe"] is True
    assert first["files"][1] == {"path": "docs/usage.md", "changes": ["untracked"]}
    assert second["commit_safe"] is False
    assert "both staged and unstaged" in second["blocked_reasons"][0]
    assert "PROPOSED COMMIT PLAN" in err.getvalue()
    assert snapshot(repo.root) == before


def test_json_reports_early_stop(repo: Repo) -> None:
    out, err = io.StringIO(), io.StringIO()
    code = cli.run(["--repo", str(repo.root), "--json"], out=out, err=err)
    assert code == 0
    assert json.loads(out.getvalue())["status"] == "clean"


# --- staging isolation --------------------------------------------------------


def _messy(repo: Repo) -> None:
    repo.write("src/util.py", "X = 1\n")
    repo.write("notes.txt", "notes\n")
    repo.commit_all("more files")
    # Pre-existing staged work that is not part of any approved group.
    repo.write("src/util.py", "X = 2\n")
    repo.write("src/staged_new.py", "NEW = True\n")
    repo.git("add", "src/util.py", "src/staged_new.py")
    # Unstaged and untracked work that is not part of the plan.
    repo.write("notes.txt", "notes, edited\n")
    repo.write("scratch.txt", "scratch\n")
    # The changes the plan will commit.
    _two_changes(repo)


def test_only_approved_changes_are_committed_and_everything_else_is_preserved(
    repo: Repo, fake_ollama: FakeOllama
) -> None:
    _messy(repo)
    fake_ollama.reply = plan_reply(
        group(["src/app.py", "docs/usage.md"], "feat", "return 2 and document it"),
        group(["src/util.py", "src/staged_new.py"], "refactor", "bump util"),
    )
    staged_util, staged_new = blob(repo, ":src/util.py"), blob(repo, ":src/staged_new.py")
    worktree_before = {p: (repo.root / p).read_bytes() for p in
                       ("notes.txt", "scratch.txt", "src/util.py", "src/staged_new.py", "src/app.py")}

    code, out = review(repo, fake_ollama, Script(["y", "n", "yes"]))

    assert code == 0, out
    assert subjects(repo)[0] == "feat: return 2 and document it"
    assert commit_files(repo) == {"src/app.py", "docs/usage.md"}
    assert repo.git("show", "HEAD:src/app.py") == APP_V2
    # Pre-existing staged changes are still staged, byte for byte.
    assert blob(repo, ":src/util.py") == staged_util
    assert blob(repo, ":src/staged_new.py") == staged_new
    assert status_lines(repo) == {"M  src/util.py", "A  src/staged_new.py", " M notes.txt", "?? scratch.txt"}
    for path, data in worktree_before.items():
        assert (repo.root / path).read_bytes() == data


def test_fully_staged_file_is_committed_exactly_as_staged(repo: Repo, fake_ollama: FakeOllama) -> None:
    repo.write("src/app.py", APP_V2)
    repo.git("add", "src/app.py")
    repo.write("docs/usage.md", "usage\n")
    staged = blob(repo, ":src/app.py")
    fake_ollama.reply = plan_reply(group(["src/app.py"], "fix", "return 2"))

    code, out = review(repo, fake_ollama, Script(["y", "yes"]))

    assert code == 0, out
    assert blob(repo, "HEAD:src/app.py") == staged
    assert status_lines(repo) == {"?? docs/usage.md"}


def test_mixed_staged_and_unstaged_file_is_blocked(repo: Repo, fake_ollama: FakeOllama) -> None:
    repo.write("src/app.py", APP_V2)
    repo.git("add", "src/app.py")
    repo.write("src/app.py", "def main():\n    return 3\n")
    repo.write("docs/usage.md", "usage\n")
    staged = blob(repo, ":src/app.py")
    fake_ollama.reply = plan_reply(
        group(["src/app.py"], "fix", "return 3"),
        group(["docs/usage.md"], "docs", "add usage"),
    )
    script = Script(["y", "yes"])

    code, out = review(repo, fake_ollama, script)

    assert code == 0, out
    assert "src/app.py: has both staged and unstaged changes" in out
    assert "-> BLOCKED" in out
    assert len(script.questions) == 2  # never asked about the blocked group
    assert subjects(repo)[0] == "docs: add usage"
    assert commit_files(repo) == {"docs/usage.md"}
    assert blob(repo, ":src/app.py") == staged
    assert (repo.root / "src/app.py").read_text() == "def main():\n    return 3\n"
    assert status_lines(repo) == {"MM src/app.py"}


def test_only_blocked_groups_commits_nothing(repo: Repo, fake_ollama: FakeOllama) -> None:
    repo.write("src/app.py", APP_V2)
    repo.git("add", "src/app.py")
    repo.write("src/app.py", "def main():\n    return 3\n")
    fake_ollama.reply = plan_reply(group(["src/app.py"], "fix", "return 3"))
    code, out = review(repo, fake_ollama, Script([]))
    assert code == 0
    assert "No groups were approved. Nothing was committed." in out
    assert subjects(repo) == ["initial"]


def test_rename_and_deletion_are_committed_as_separate_groups(repo: Repo, fake_ollama: FakeOllama) -> None:
    repo.git("mv", "src/app.py", "src/main.py")
    (repo.root / "README.md").unlink()
    fake_ollama.reply = plan_reply(
        group(["src/main.py"], "refactor", "rename app module"),
        group(["README.md"], "docs", "drop readme"),
    )

    code, out = review(repo, fake_ollama, Script(["y", "y", "yes"]))

    assert code == 0, out
    assert subjects(repo) == ["docs: drop readme", "refactor: rename app module", "initial"]
    assert commit_files(repo, "HEAD~1") == {"src/app.py", "src/main.py"}
    assert commit_files(repo) == {"README.md"}
    assert status_lines(repo) == set()


def test_first_commit_in_empty_repository(tmp_path: Path, fake_ollama: FakeOllama) -> None:
    r = Repo(tmp_path)
    r.git("init", "-q", "-b", "main")
    r.git("config", "user.name", "Test User")
    r.git("config", "user.email", "test@example.invalid")
    r.git("config", "commit.gpgsign", "false")
    r.write("a.txt", "a\n")
    r.write("b.txt", "b\n")
    fake_ollama.reply = plan_reply(group(["a.txt"], "feat", "add a"))
    code, out = review(r, fake_ollama, Script(["y", "yes"]))
    assert code == 0, out
    assert subjects(r) == ["feat: add a"]
    assert status_lines(r) == {"?? b.txt"}


# --- revalidation -------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.write("src/app.py", "def main():\n    return 99\n"), id="approved-file-edited"),
        pytest.param(lambda r: r.write("README.md", "# edited later\n"), id="other-file-edited"),
        pytest.param(lambda r: r.git("add", "docs/usage.md"), id="index-changed"),
        pytest.param(lambda r: r.write("late.txt", "new\n"), id="new-untracked-file"),
        pytest.param(lambda r: (r.root / "docs/usage.md").unlink(), id="file-removed"),
    ],
)
def test_state_change_after_planning_invalidates_plan(
    repo: Repo, fake_ollama: FakeOllama, mutate: Callable[[Repo], Any]
) -> None:
    _two_changes(repo)
    fake_ollama.reply = _two_groups()

    code, out = review(repo, fake_ollama, Script(["y", "y", "yes"], before_confirm=lambda: mutate(repo)))

    assert code == 4, out
    assert subjects(repo) == ["initial"]
    assert "no longer matches the state the plan was generated from" in out
    assert "0 commit(s) created" in out
    assert "[1] NOT RUN" in out and "[2] NOT RUN" in out
    assert "Run `lazycommit --review` again" in out


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda r: r.git("checkout", "-q", "-b", "other"), "branch changed from 'main' to 'other'"),
        (lambda r: r.git("commit", "-q", "--allow-empty", "-m", "sneaky"), "HEAD moved"),
        (lambda r: r.git("checkout", "-q", "--detach"), "HEAD is detached"),
    ],
    ids=["branch", "head", "detached"],
)
def test_branch_or_head_change_invalidates_plan(
    repo: Repo, fake_ollama: FakeOllama, mutate: Callable[[Repo], Any], expected: str
) -> None:
    _two_changes(repo)
    fake_ollama.reply = _two_groups()
    head = blob(repo, "HEAD")

    code, out = review(repo, fake_ollama, Script(["y", "y", "yes"], before_confirm=lambda: mutate(repo)))

    assert code == 4, out
    assert expected in out
    assert "fix(app): return 2 from main" not in subjects(repo)
    assert repo.git("rev-list", "--count", f"{head}..HEAD").strip() in ("0", "1")


def test_change_between_two_commits_stops_the_second(
    repo: Repo, fake_ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_changes(repo)
    fake_ollama.reply = _two_groups()
    real_verify = execute._verify

    def verify_then_edit(*args: Any) -> list[str]:
        problems = real_verify(*args)
        repo.write("docs/usage.md", "edited between commits\n")
        return problems

    monkeypatch.setattr(execute, "_verify", verify_then_edit)
    code, out = review(repo, fake_ollama, Script(["y", "y", "yes"]))

    assert code == 4, out
    assert subjects(repo) == ["fix(app): return 2 from main", "initial"]
    assert "docs/usage.md: file contents changed" in out
    assert "[2] NOT RUN" in out


def test_unsafe_state_at_execution_stops(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.reply = _two_groups()
    git_dir = repo.root / ".git"

    code, out = review(repo, fake_ollama, Script(
        ["y", "y", "yes"], before_confirm=lambda: (git_dir / "MERGE_HEAD").write_text(blob(repo, "HEAD"))
    ))

    assert code == 4, out
    assert "merge in progress" in out
    assert subjects(repo) == ["initial"]


# --- secrets ---------------------------------------------------------------------


def test_secret_excluded_files_cannot_be_committed_via_plan(repo: Repo, fake_ollama: FakeOllama) -> None:
    repo.write(".env", f"DB_PASSWORD={FAKE_ENV_VALUE}\n")
    repo.write("src/config.py", f'KEY = "{FAKE_AWS_KEY}"\n')
    repo.write("src/app.py", APP_V2)
    fake_ollama.reply = plan_reply(
        group(["src/app.py", ".env"], "chore", "update config"),
        group(["src/config.py"], "feat", "add config"),
    )

    code, out = review(repo, fake_ollama, Script(["y", "y", "yes"]))

    assert code == 0
    assert "did not produce any valid commit groups" in out
    assert subjects(repo) == ["initial"]
    assert {"?? .env", "?? src/config.py"} <= status_lines(repo)
    assert FAKE_AWS_KEY not in out and FAKE_ENV_VALUE not in out


def _approved(state_repo: Repo, files: list[str], message: str = "sneak") -> tuple[int, list[GroupResult]]:
    state = read_state(state_repo.root)
    collected = collect(state)
    items = [GroupResult(1, Group("feat", "", message, files, "r", "high"), status=APPROVED)]
    run = execute.execute(state, collected, take_snapshot(state), items)
    assert run.stopped
    return len(subjects(state_repo)), items


@pytest.mark.parametrize(
    "files,message,reason",
    [
        ([".env"], "add env", "not a secret-scanned file"),
        (["src/config.py"], "add config", "not a secret-scanned file"),
        (["src/app.py"], f"leak {FAKE_AWS_KEY}", "commit message matches a secret pattern"),
        (["src/app.py", "src/never.py"], "invented", "not a secret-scanned file"),
    ],
)
def test_executor_refuses_unsafe_groups_even_if_approved(
    repo: Repo, files: list[str], message: str, reason: str
) -> None:
    repo.write(".env", f"DB_PASSWORD={FAKE_ENV_VALUE}\n")
    repo.write("src/config.py", f'KEY = "{FAKE_AWS_KEY}"\n')
    repo.write("src/app.py", APP_V2)
    count, items = _approved(repo, files, message)
    assert count == 1
    assert items[0].status == BLOCKED
    assert any(reason in r for r in items[0].blocked)
    assert all(FAKE_AWS_KEY not in r for r in items[0].blocked)


def test_binary_files_are_not_committed_automatically(repo: Repo, fake_ollama: FakeOllama) -> None:
    repo.write("blob.bin", b"\x00\x01binary")
    fake_ollama.reply = plan_reply(group(["blob.bin"], "chore", "add blob"))
    code, out = review(repo, fake_ollama, Script([]))
    assert code == 0
    assert "binary content could not be secret-scanned" in out
    assert subjects(repo) == ["initial"]


# --- failures -------------------------------------------------------------------


def _hook(repo: Repo, body: str) -> None:
    hook = repo.root / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(exist_ok=True)
    hook.write_bytes(("#!/bin/sh\n" + body).encode())
    os.chmod(hook, 0o755)


def test_failed_commit_stops_subsequent_groups(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    repo.write("README.md", "# readme edit\n")
    start = blob(repo, "HEAD")
    _hook(repo, "if git diff --cached --name-only | grep -q '^docs/'; then\n"
                "  echo 'hook says no' >&2\n  exit 1\nfi\nexit 0\n")
    fake_ollama.reply = plan_reply(
        group(["src/app.py"], "fix", "return 2"),
        group(["docs/usage.md"], "docs", "add usage"),
        group(["README.md"], "docs", "edit readme"),
    )

    code, out = review(repo, fake_ollama, Script(["y", "y", "y", "yes"]))

    assert code == 1, out
    assert subjects(repo) == ["fix: return 2", "initial"]
    assert "[1] COMMITTED" in out
    assert "[2] FAILED" in out and "hook says no" in out
    assert "[3] NOT RUN" in out
    assert f"git reset --soft {start}" in out
    # The failed group's new file is untracked again, not left half-staged.
    assert status_lines(repo) == {"?? docs/usage.md", " M README.md"}


def test_hook_that_adds_unapproved_changes_fails_verification(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    repo.write("README.md", "# not approved\n")
    _hook(repo, "git add README.md\nexit 0\n")
    fake_ollama.reply = _two_groups()

    code, out = review(repo, fake_ollama, Script(["y", "y", "yes"]))

    assert code == 1, out
    assert "[1] FAILED" in out
    assert "commit includes unapproved path(s): README.md" in out
    assert "[2] NOT RUN" in out
    assert "git show --stat HEAD" in out
    assert len(subjects(repo)) == 2  # the one unverified commit, not undone


def test_write_wrapper_rejects_dangerous_commands(repo: Repo) -> None:
    root = repo.root
    bad: list[tuple[list[str], list[str]]] = [
        (["add"], ["src/app.py"]),
        (["add", "-A"], ["src/app.py"]),
        (["add", "--intent-to-add"], ["."]),
        (["add", "--intent-to-add"], []),
        (["add", "--intent-to-add"], ["-A"]),
        (["commit", "-m", "x"], ["src/app.py"]),
        (["commit", "--only", "--amend", "-m", "x"], ["src/app.py"]),
        (["commit", "--only", "-a", "-m", "x"], ["src/app.py"]),
        (["rm"], ["src/app.py"]),
        (["push", "--only"], ["origin"]),
        (["reset", "--only"], ["src/app.py"]),
        (["clean", "--only"], ["src/app.py"]),
    ]
    for args, paths in bad:
        with pytest.raises(GitError, match="refus"):
            execute._run_write(root, args, paths)


def test_no_push_amend_reset_clean_or_rewrite(
    repo: Repo, fake_ollama: FakeOllama, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True, capture_output=True)
    repo.git("remote", "add", "origin", str(remote))
    repo.git("push", "-q", "origin", "main")
    remote_before = snapshot(remote)
    start = blob(repo, "HEAD")
    reflog_before = repo.git("reflog", "--format=%H %gs", "main").splitlines()
    _two_changes(repo)
    fake_ollama.reply = _two_groups()

    calls: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(list(cmd))
        assert kwargs.get("shell") is False
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    code, out = review(repo, fake_ollama, Script(["y", "y", "yes"]))
    monkeypatch.setattr(subprocess, "run", real_run)

    assert code == 0, out
    allowed = gitrepo.READ_ONLY_SUBCOMMANDS | {"add", "commit", "rm"}
    seen = []
    for cmd in calls:
        assert cmd[0] == "git"
        sub = next(a for a in cmd[1:] if not a.startswith("-") and "=" not in a)
        seen.append(sub)
        assert sub in allowed, cmd
        if sub == "add":
            assert "--intent-to-add" in cmd and "-A" not in cmd and "." not in cmd
        if sub == "commit":
            assert "--only" in cmd and "--amend" not in cmd and "-a" not in cmd
    assert seen.count("commit") == 2 and "rm" not in seen
    # History only grew: two new single-parent commits on top of the old HEAD.
    assert repo.git("rev-list", "--count", f"{start}..HEAD").strip() == "2"
    assert repo.git("rev-list", "--min-parents=2", "HEAD").strip() == ""
    reflog = repo.git("reflog", "--format=%H %gs", "main").splitlines()
    assert reflog[2:] == reflog_before
    assert all(line.split(" ", 1)[1].startswith("commit: ") for line in reflog[:2])
    # Nothing reached the remote.
    assert snapshot(remote) == remote_before
    assert "Nothing was pushed" in out
