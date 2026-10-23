"""Create approved commits, one ordinary local commit per approved group.

Staging and committing are deterministic application code; the model only
proposes groups. Safety model:

* When the plan is generated, a :class:`Snapshot` records the branch, HEAD,
  every porcelain status record (which includes index blob ids and modes),
  and a hash of every changed file's working-tree bytes.
* Before every commit the live repository must equal that snapshot minus the
  groups already committed. Any difference stops execution.
* Each commit runs ``git commit --only -- <paths>``: Git builds the commit
  from HEAD plus exactly those paths and leaves every other index entry
  (pre-existing staged changes) untouched. Groups that cannot be isolated
  this way are blocked up front (see :func:`blocked_reasons`).
* After every commit, its parent, the exact set of paths it changed, and the
  rest of the repository are verified against the snapshot.
* The only write commands that can run are ``add --intent-to-add`` (so a new
  file is known to Git), ``commit --only``, and ``rm --cached`` (to undo our
  own intent-to-add if a commit fails). Nothing is pushed, amended, reset,
  cleaned, or rewritten, and a failed commit stops all later groups.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from lazycommit import secrets
from lazycommit.collect import CollectResult
from lazycommit.gitrepo import (
    _SAFE_CONFIG,
    GitError,
    RepoState,
    _git_env,
    read_state,
    run_git,
)
from lazycommit.planner import Group

# Group lifecycle.
PROPOSED = "proposed"
APPROVED = "approved"
REJECTED = "rejected"
BLOCKED = "blocked"  # rejected automatically: cannot be committed safely
COMMITTED = "committed"
FAILED = "failed"
NOT_RUN = "not run"  # approved, but execution stopped before reaching it

WRITE_TIMEOUT_SECONDS = 600  # commit hooks can be slow

# Each allowed write subcommand and the flag it must always carry.
_WRITE_COMMANDS = {"add": "--intent-to-add", "commit": "--only", "rm": "--cached"}
_FORBIDDEN_ARGS = frozenset({
    "-A", "--all", "-a", "-u", "--update", "-f", "--force", "--amend", "-p", "--patch",
    "-i", "--include", "--interactive", "--fixup", "--squash", "-r",
})


@dataclass
class Snapshot:
    branch: str
    head: str | None
    in_progress: tuple[str, ...]
    # path -> (porcelain record, working-tree content hash)
    entries: dict[str, tuple[str, str]] = field(default_factory=dict)


def _worktree_hash(root: Path, path: str) -> str:
    full = root / path
    try:
        st = full.lstat()
        if stat.S_ISLNK(st.st_mode):
            return "symlink:" + hashlib.sha256(os.readlink(full).encode("utf-8", "replace")).hexdigest()
        if stat.S_ISDIR(st.st_mode):
            return "directory"
        if not stat.S_ISREG(st.st_mode):
            return "special file"
        digest = hashlib.sha256()
        with full.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        return f"unreadable ({type(exc).__name__})"


def take_snapshot(state: RepoState) -> Snapshot:
    return Snapshot(
        branch=state.branch,
        head=state.head_oid,
        in_progress=tuple(state.in_progress),
        entries={f.path: (f.record, _worktree_hash(state.root, f.path)) for f in state.files},
    )


def differences(expected: Snapshot, current: Snapshot, *, committed: set[str], head: str | None) -> list[str]:
    """Describe how ``current`` differs from ``expected`` minus ``committed`` paths."""
    problems: list[str] = []
    if current.branch != expected.branch:
        problems.append(f"branch changed from {expected.branch!r} to {current.branch!r}")
    if current.head != head:
        problems.append(f"HEAD moved (expected {_short(head)}, found {_short(current.head)})")
    for op in current.in_progress:
        problems.append(f"{op} in progress")
    want = {p: v for p, v in expected.entries.items() if p not in committed}
    for path in sorted(set(want) | set(current.entries)):
        if path not in current.entries:
            problems.append(f"{path}: change is no longer present")
        elif path not in want:
            problems.append(f"{path}: new or re-appeared change since the plan was generated")
        elif current.entries[path][0] != want[path][0]:
            problems.append(f"{path}: index or file status changed")
        elif current.entries[path][1] != want[path][1]:
            problems.append(f"{path}: file contents changed")
    return problems


def _short(oid: str | None) -> str:
    return oid[:12] if oid else "no commit"


def session_problems(state: RepoState) -> list[str]:
    """Repository conditions under which no commit may be created at all."""
    problems = list(state.unsafe_reasons)
    if state.branch == "(detached)":
        problems.append("HEAD is detached; commits would not be on any branch")
    if (state.git_dir / "index.lock").exists():
        problems.append("index.lock exists; another Git process may be running")
    return problems


def commit_paths(group: Group, state: RepoState) -> list[str]:
    """Every path a group's commit touches: its files plus rename sources."""
    by_path = {f.path: f for f in state.files}
    paths = list(group.files)
    for path in group.files:
        st = by_path.get(path)
        if st and st.orig_path and st.orig_path not in paths:
            paths.append(st.orig_path)
    return paths


def _unsafe_pathspec(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        not path
        or path.startswith(("-", ":", "/"))
        or "\\" in path
        or re.match(r"^[A-Za-z]:", path) is not None
        or any(part in ("", ".", "..", ".git") for part in pure.parts)
        or path in (".", "*")
    )


def blocked_reasons(group: Group, state: RepoState, collected: CollectResult) -> list[str]:
    """Why a group cannot be committed safely. Empty means it can be."""
    reasons: list[str] = []
    by_path = {f.path: f for f in state.files}
    candidates = {c.path: c for c in collected.candidates}
    not_eligible = {s.path for s in collected.excluded + collected.manual}
    rename_sources = {f.orig_path: f.path for f in state.files if f.orig_path}

    header = group.header
    reasons += [f"needs manual review: {flag}" for flag in group.flags]
    if not group.message.strip() or re.search(r"[\x00-\x1f\x7f]", header):
        reasons.append("commit message is empty or contains control characters")
    if secrets.contains_secret(header):
        reasons.append("commit message matches a secret pattern")
    if not group.files:
        reasons.append("group has no files")

    for path in group.files:
        st = by_path.get(path)
        cand = candidates.get(path)
        if st is None or cand is None or path in not_eligible:
            reasons.append(f"{path}: not a secret-scanned file from this plan")
            continue
        names = [path] + ([st.orig_path] if st.orig_path else [])
        if any(secrets.sensitive_filename(n) or secrets.contains_secret(n) for n in names):
            reasons.append(f"{path}: sensitive file; excluded by safety checks")
            continue
        if any(_unsafe_pathspec(n) for n in names):
            reasons.append(f"{path}: path cannot be passed to Git safely")
            continue
        if st.conflicted:
            reasons.append(f"{path}: has unresolved merge conflicts")
        elif st.submodule:
            reasons.append(f"{path}: submodule changes are not committed automatically")
        elif st.staged and st.unstaged:
            reasons.append(
                f"{path}: has both staged and unstaged changes; Lazy-Commit cannot commit "
                "one part without the other, so it will not commit either"
            )
        elif cand.binary:
            reasons.append(f"{path}: binary content could not be secret-scanned")
        elif (
            "T" in (st.index_status, st.worktree_status)
            or "120000" in st.modes
            or (state.root / path).is_symlink()
        ):
            reasons.append(f"{path}: symbolic links and file-type changes are not committed automatically")
        if st.orig_path and st.orig_path in by_path:
            reasons.append(f"{path}: rename source {st.orig_path} also has separate changes")
        if path in rename_sources:
            reasons.append(f"{path}: is also the source of the rename to {rename_sources[path]}")
    return reasons


@dataclass
class GroupResult:
    index: int
    group: Group
    blocked: list[str] = field(default_factory=list)
    status: str = PROPOSED
    commit: str | None = None
    subject: str | None = None
    error: str | None = None


@dataclass
class Execution:
    """Outcome of :func:`execute`. ``stopped`` explains an early stop."""

    start_head: str | None
    stale: list[str] = field(default_factory=list)
    stopped: str | None = None


def _run_write(root: Path, args: list[str], paths: list[str]) -> None:
    """Run one of the few allowed write commands with explicit literal paths."""
    if not args or _WRITE_COMMANDS.get(args[0]) not in args:
        raise GitError(f"refusing to run non-allowlisted git write command: {args[:1]}")
    if not paths or any(_unsafe_pathspec(p) for p in paths):
        raise GitError("refusing to run git write command without safe explicit paths")
    if args[0] != "commit" and len(args) != 2:
        raise GitError(f"refusing unexpected arguments for git {args[0]}")
    options = args[:-2] if args[0] == "commit" else args  # skip the -m value
    if args[0] == "commit" and args[-2] != "-m":
        raise GitError("commit message must be the last argument")
    if _FORBIDDEN_ARGS.intersection(options):
        raise GitError("refusing forbidden git option")
    cmd = ["git", "--no-pager", "--no-optional-locks", *_SAFE_CONFIG, *args, "--", *paths]
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=_git_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=WRITE_TIMEOUT_SECONDS,
            shell=False,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {args[0]} timed out after {WRITE_TIMEOUT_SECONDS}s") from exc
    if proc.returncode != 0:
        lines = [
            ln for ln in (proc.stderr + proc.stdout).decode("utf-8", "replace").splitlines()
            if ln.strip() and not ln.startswith(("warning: in the working copy", "hint:"))
        ]
        detail = secrets.redact(lines[-1].strip()) if lines else f"exit {proc.returncode}"
        raise GitError(f"git {args[0]} failed: {detail}")


def _head(root: Path) -> str | None:
    out = run_git(root, "rev-parse", "--verify", "--quiet", "HEAD", check=False).decode().strip()
    return out or None


def _subject(root: Path, oid: str) -> str:
    raw = run_git(root, "cat-file", "commit", oid).decode("utf-8", "replace")
    message = raw.split("\n\n", 1)[1] if "\n\n" in raw else ""
    return message.strip().splitlines()[0] if message.strip() else ""


def _verify(root: Path, item: GroupResult, paths: list[str], prev: str | None, new: str,
            snapshot: Snapshot, committed: set[str]) -> list[str]:
    problems: list[str] = []
    parents = run_git(root, "rev-parse", f"{new}^@").decode().split()
    if parents != ([prev] if prev else []):
        problems.append("new commit does not sit directly on the previous HEAD")
    changed = {
        p for p in run_git(
            root, "diff-tree", "--root", "--no-commit-id", "-r", "-z", "--name-only", "--no-renames", new
        ).decode("utf-8", "replace").split("\0") if p
    }
    extra, missing = changed - set(paths), set(paths) - changed
    if extra:
        problems.append("commit includes unapproved path(s): " + ", ".join(sorted(extra)))
    if missing:
        problems.append("commit is missing approved path(s): " + ", ".join(sorted(missing)))
    for path in item.group.files:
        if _worktree_hash(root, path) != snapshot.entries[path][1]:
            problems.append(f"{path}: working-tree file changed during the commit (a hook may have edited it)")
    current = take_snapshot(read_state(root))
    problems += differences(snapshot, current, committed=committed, head=new)
    return problems


def execute(state: RepoState, collected: CollectResult, snapshot: Snapshot,
            items: list[GroupResult]) -> Execution:
    """Commit every APPROVED item in order, updating statuses in place.

    Stops at the first stale state, failure, or verification problem; later
    approved items are marked NOT_RUN. Successful commits are never undone.
    """
    root = state.root
    result = Execution(start_head=snapshot.head)
    approved = [it for it in items if it.status == APPROVED]

    def stop(reason: str, after: GroupResult | None = None) -> Execution:
        result.stopped = reason
        for it in approved:
            if it.status == APPROVED and it is not after:
                it.status = NOT_RUN
        return result

    # Defense in depth: the review step should never let these through.
    for it in approved:
        reasons = blocked_reasons(it.group, state, collected)
        if reasons:
            it.blocked, it.status = reasons, BLOCKED
            return stop(f"group {it.index} is not safe to commit; nothing was committed")
    claimed: set[str] = set()
    for it in approved:
        touched = set(commit_paths(it.group, state))
        if touched & claimed:
            return stop("two approved groups share a file; nothing was committed")
        claimed |= touched

    by_path = {f.path: f for f in state.files}
    committed: set[str] = set()
    head = snapshot.head
    for it in approved:
        current_state = read_state(root)
        problems = session_problems(current_state)
        problems += differences(snapshot, take_snapshot(current_state), committed=committed, head=head)
        if problems:
            result.stale = problems
            return stop(f"the repository changed; stopped before group {it.index}")

        paths = commit_paths(it.group, state)
        new_files = [p for p in it.group.files if by_path[p].untracked]
        try:
            if new_files:
                _run_write(root, ["add", "--intent-to-add"], new_files)
            _run_write(root, ["commit", "--only", "--no-status", "-m", it.group.header], paths)
        except GitError as exc:
            it.status, it.error = FAILED, str(exc)
            moved = _head(root)
            if moved != head:
                it.commit = moved
                it.error += f"; HEAD nevertheless moved to {_short(moved)} - inspect it with `git show --stat HEAD`"
            elif new_files:
                try:
                    _run_write(root, ["rm", "--cached"], new_files)
                except GitError:
                    it.error += (
                        "; new file(s) are still marked intent-to-add: "
                        + ", ".join(new_files) + " (undo with `git rm --cached -- <path>`)"
                    )
            return stop(f"group {it.index} failed", after=it)

        new_head = _head(root)
        if new_head is None or new_head == head:
            it.status, it.error = FAILED, "git reported success but HEAD did not move"
            return stop(f"group {it.index} failed", after=it)
        it.commit = new_head
        it.subject = _subject(root, new_head)
        committed |= set(it.group.files)
        problems = _verify(root, it, paths, head, new_head, snapshot, committed)
        if problems:
            it.status = FAILED
            it.error = (
                f"commit {_short(new_head)} was created but does not match what you approved: "
                + "; ".join(problems[:5])
            )
            return stop(f"group {it.index} failed verification", after=it)
        it.status = COMMITTED
        head = new_head
    return result
