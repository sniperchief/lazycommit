"""Read-only access to a Git repository.

Every read-only Git invocation goes through :func:`run_git`, which:

* never uses a shell (arguments are passed as a list),
* only permits an allowlist of read-only subcommands,
* disables optional locks so ``git status`` does not rewrite the index,
* disables external diff drivers, textconv filters, fsmonitor hooks, and
  pagers, so repository config cannot make us execute arbitrary programs.

The few write operations used to create approved commits live in
``execute.py`` and go through their own, separately restricted wrapper.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Subcommands Lazy-Commit is allowed to run. Nothing here writes to the
# index, working tree, refs, or objects. ``ls-remote`` reads from the network
# but never changes anything; ``config`` is limited to reads (see run_git).
READ_ONLY_SUBCOMMANDS = frozenset({
    "rev-parse", "status", "diff", "ls-files", "diff-tree", "cat-file",
    "log", "rev-list", "merge-base", "ls-remote", "config",
})
_CONFIG_READ_FLAGS = frozenset({"--get", "--get-regexp"})

_SAFE_CONFIG = (
    "-c", "core.fsmonitor=false",
    "-c", "core.quotepath=off",
    "-c", "color.ui=never",
    "-c", "diff.external=",
)

GIT_TIMEOUT_SECONDS = 60


class GitError(Exception):
    """A Git command failed or the directory is not usable."""


class NotARepositoryError(GitError):
    pass


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "GIT_LITERAL_PATHSPECS": "1",
            "LC_ALL": "C",
        }
    )
    # Never let an inherited environment redirect us to a different repo/index.
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_EXTERNAL_DIFF"):
        env.pop(var, None)
    return env


def _run(cwd: Path, args: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
    if not args or args[0] not in READ_ONLY_SUBCOMMANDS:
        raise GitError(f"refusing to run non-allowlisted git command: {args[:1]}")
    if args[0] == "config" and (len(args) < 2 or args[1] not in _CONFIG_READ_FLAGS):
        raise GitError("refusing to run git config without --get/--get-regexp")
    cmd = ["git", "--no-pager", "--no-optional-locks", *_SAFE_CONFIG, *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=_git_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            shell=False,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {args[0]} timed out after {GIT_TIMEOUT_SECONDS}s") from exc
    return proc


def run_git(cwd: Path, *args: str, check: bool = True) -> bytes:
    """Run a read-only git command and return raw stdout."""
    proc = _run(cwd, args)
    if check and proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise GitError(f"git {args[0]} failed: {msg[-1] if msg else 'exit ' + str(proc.returncode)}")
    return proc.stdout


def git_succeeds(cwd: Path, *args: str) -> bool:
    """Run a read-only git command and report whether it exited 0."""
    return _run(cwd, args).returncode == 0


@dataclass
class FileStatus:
    path: str
    index_status: str  # X column of porcelain v2 ('.' = unchanged)
    worktree_status: str  # Y column
    orig_path: str | None = None
    untracked: bool = False
    conflicted: bool = False
    submodule: bool = False
    # File modes (HEAD, index, worktree...) from the porcelain record.
    modes: tuple[str, ...] = ()
    # The raw porcelain record, used to detect any change after planning.
    record: str = ""

    @property
    def staged(self) -> bool:
        return not self.untracked and self.index_status != "."

    @property
    def unstaged(self) -> bool:
        return not self.untracked and self.worktree_status != "."


@dataclass
class RepoState:
    root: Path
    git_dir: Path
    branch: str
    head_oid: str | None
    files: list[FileStatus] = field(default_factory=list)
    in_progress: list[str] = field(default_factory=list)

    @property
    def staged(self) -> list[FileStatus]:
        return [f for f in self.files if f.staged]

    @property
    def unstaged(self) -> list[FileStatus]:
        return [f for f in self.files if f.unstaged]

    @property
    def untracked(self) -> list[FileStatus]:
        return [f for f in self.files if f.untracked]

    @property
    def conflicted(self) -> list[FileStatus]:
        return [f for f in self.files if f.conflicted]

    @property
    def unsafe_reasons(self) -> list[str]:
        reasons = [f"{op} in progress" for op in self.in_progress]
        if self.conflicted:
            reasons.append(f"{len(self.conflicted)} file(s) have unresolved merge conflicts")
        return reasons


def find_repo_root(start: Path) -> Path:
    out = run_git(start, "rev-parse", "--is-inside-work-tree", check=False)
    if out.strip() != b"true":
        raise NotARepositoryError(f"{start} is not inside a Git working tree")
    top = run_git(start, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    return Path(top)


def _git_dir(root: Path) -> Path:
    raw = run_git(root, "rev-parse", "--absolute-git-dir").decode("utf-8").strip()
    return Path(raw)


def _in_progress_operations(git_dir: Path) -> list[str]:
    markers = {
        "MERGE_HEAD": "merge",
        "CHERRY_PICK_HEAD": "cherry-pick",
        "REVERT_HEAD": "revert",
        "rebase-merge": "rebase",
        "rebase-apply": "rebase/am",
        "BISECT_LOG": "bisect",
    }
    found: list[str] = []
    for name, label in markers.items():
        if (git_dir / name).exists() and label not in found:
            found.append(label)
    return found


def parse_porcelain_v2(raw: bytes) -> tuple[str, str | None, list[FileStatus]]:
    """Parse ``git status --porcelain=v2 -z --branch`` output."""
    branch = "(unknown)"
    head_oid: str | None = None
    files: list[FileStatus] = []
    records = raw.decode("utf-8", "replace").split("\0")
    i = 0
    while i < len(records):
        rec = records[i]
        i += 1
        if not rec:
            continue
        if rec.startswith("# branch.head "):
            branch = rec[len("# branch.head "):]
        elif rec.startswith("# branch.oid "):
            oid = rec[len("# branch.oid "):]
            head_oid = None if oid == "(initial)" else oid
        elif rec.startswith("#"):
            continue
        elif rec.startswith("1 "):
            parts = rec.split(" ", 8)
            xy, sub = parts[1], parts[2]
            files.append(
                FileStatus(parts[8], xy[0], xy[1], submodule=sub.startswith("S"),
                           modes=tuple(parts[3:6]), record=rec)
            )
        elif rec.startswith("2 "):
            parts = rec.split(" ", 9)
            xy, sub = parts[1], parts[2]
            orig = records[i] if i < len(records) else None
            i += 1
            files.append(
                FileStatus(parts[9], xy[0], xy[1], orig_path=orig, submodule=sub.startswith("S"),
                           modes=tuple(parts[3:6]), record=f"{rec}\0{orig}")
            )
        elif rec.startswith("u "):
            parts = rec.split(" ", 10)
            xy = parts[1]
            files.append(FileStatus(parts[10], xy[0], xy[1], conflicted=True,
                                    modes=tuple(parts[3:7]), record=rec))
        elif rec.startswith("? "):
            files.append(FileStatus(rec[2:], "?", "?", untracked=True, record=rec))
        # '!' (ignored) entries are never requested, so never included.
    return branch, head_oid, files


def read_state(start: Path) -> RepoState:
    root = find_repo_root(start)
    git_dir = _git_dir(root)
    raw = run_git(
        root,
        "status",
        "--porcelain=v2",
        "-z",
        "--branch",
        "--untracked-files=all",
        "--find-renames",
        "--ignore-submodules=none",
    )
    branch, head_oid, files = parse_porcelain_v2(raw)
    return RepoState(
        root=root,
        git_dir=git_dir,
        branch=branch,
        head_oid=head_oid,
        files=files,
        in_progress=_in_progress_operations(git_dir),
    )


def file_diff(root: Path, status: FileStatus, *, staged: bool) -> bytes:
    """Diff for one file: index vs HEAD (staged) or worktree vs index."""
    args = ["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--find-renames"]
    if staged:
        args.append("--cached")
    args.append("--")
    args.append(status.path)
    if status.orig_path and staged:
        args.append(status.orig_path)
    return run_git(root, *args)
