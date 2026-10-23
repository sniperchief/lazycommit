"""Verified, read-only data about the current branch for pull requests and changelogs.

Everything here comes from read-only Git commands. Commit messages and diffs
are untrusted data: they are secret-scanned and bounded before any of it can
reach the local model, and the commits that would be pushed are scanned in
full before any remote action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lazycommit import secrets
from lazycommit.collect import (
    _MIN_USEFUL_CHARS,
    Limits,
    Section,
    Skipped,
    _is_binary_diff,
    _locate,
    _truncate,
)
from lazycommit.gitrepo import GitError, RepoState, git_succeeds, read_state, run_git

# Branch, remote, and revision names we are willing to pass to Git.
_SAFE_NAME = re.compile(r"^(?!-)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/@{}^~-]+$")
_GITHUB_URL = re.compile(
    r"^(?:https?://(?:[^@/\s]+@)?github\.com/|ssh://git@github\.com/|git@github\.com:)"
    r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_TRAILER = re.compile(r"^(?:[A-Za-z][\w-]*-by|Co-Authored-By|Change-Id|Refs?):\s", re.I)
_DIFF_FLAGS = ("--no-ext-diff", "--no-textconv", "--no-color")


class BranchError(Exception):
    """PR/changelog preparation cannot proceed.

    ``code`` is the CLI exit code; ``category`` is a fixed label for telemetry.
    """

    def __init__(self, message: str, hint: str = "", code: int = 1, category: str = "branch_error") -> None:
        super().__init__(message)
        self.hint, self.code, self.category = hint, code, category


@dataclass
class Commit:
    oid: str
    parents: list[str]
    subject: str
    body: str

    @property
    def short(self) -> str:
        return self.oid[:7]

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1

    @property
    def explanation(self) -> str:
        """The body without trailers such as Signed-off-by / Co-Authored-By."""
        lines = [ln for ln in self.body.splitlines() if not _TRAILER.match(ln.strip())]
        return "\n".join(lines).strip()


@dataclass
class ChangedFile:
    path: str
    status: str  # A, M, D, R, C, T
    orig_path: str | None = None

    @property
    def change(self) -> str:
        words = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed",
                 "C": "copied", "T": "type changed"}
        text = words.get(self.status, self.status)
        return f"{text} (from {self.orig_path})" if self.orig_path else text


@dataclass
class BranchInfo:
    state: RepoState
    branch: str
    head: str
    remote: str | None
    github: tuple[str, str] | None  # (owner, repo) when the remote is on GitHub
    base: str  # what the user would call it, e.g. "main" or a tag
    base_ref: str  # what we compare against, e.g. refs/remotes/origin/main
    commits: list[Commit]
    files: list[ChangedFile] = field(default_factory=list)
    shortstat: str = ""

    @property
    def root(self) -> Path:
        return self.state.root

    @property
    def repo_slug(self) -> str | None:
        return f"{self.github[0]}/{self.github[1]}" if self.github else None


def parse_github_url(url: str) -> tuple[str, str] | None:
    m = _GITHUB_URL.match(url.strip())
    return (m.group(1), m.group(2)) if m else None


def _config_get(root: Path, key: str) -> str | None:
    out = run_git(root, "config", "--get", key, check=False).decode("utf-8", "replace").strip()
    return out or None


def remotes(root: Path) -> list[str]:
    raw = run_git(root, "config", "--get-regexp", r"^remote\..*\.url$", check=False)
    names: list[str] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        key = line.split(" ", 1)[0]
        if key.startswith("remote.") and key.endswith(".url"):
            names.append(key[len("remote."):-len(".url")])
    return names


def _ref_exists(root: Path, ref: str) -> bool:
    return git_succeeds(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")


def _choose_remote(root: Path, branch: str, wanted: str | None, required: bool) -> str | None:
    available = remotes(root)
    if wanted is not None:
        if wanted not in available:
            raise BranchError(
                f"remote {wanted!r} is not configured",
                "Configured remotes: " + (", ".join(available) or "none"),
                category="no_remote",
            )
        return wanted
    upstream = _config_get(root, f"branch.{branch}.remote")
    if upstream in available:
        return upstream
    if "origin" in available:
        return "origin"
    if len(available) == 1:
        return available[0]
    if len(available) > 1:
        raise BranchError("several remotes are configured and none is the branch's upstream",
                          "Pass --remote NAME (one of: " + ", ".join(available) + ").",
                          category="invalid_argument")
    if required:
        raise BranchError(
            "no Git remote is configured; a pull request needs one",
            "Add one with `git remote add origin https://github.com/OWNER/REPO.git` "
            "and push your base branch, then run this again.",
            category="no_remote",
        )
    return None


def _default_bases(root: Path, remote: str | None) -> list[str]:
    names: list[str] = []
    if remote:
        out = run_git(root, "rev-parse", "--abbrev-ref", f"refs/remotes/{remote}/HEAD", check=False)
        default = out.decode("utf-8", "replace").strip()
        if default.startswith(f"{remote}/"):
            names.append(default[len(remote) + 1:])
    for name in ("main", "master"):
        if name not in names:
            names.append(name)
    return names


def _resolve_base(root: Path, remote: str | None, base: str | None, branch: str) -> tuple[str, str]:
    names = [base] if base else _default_bases(root, remote)
    if branch in names[:1]:
        raise BranchError(
            f"you are on the base branch {branch!r}",
            "Create a feature branch for your work first (`git switch -c my-feature`), "
            "or pass a different --base.",
            category="on_base_branch",
        )
    for name in names:
        refs = ([f"refs/remotes/{remote}/{name}"] if remote else []) + [f"refs/heads/{name}"]
        for ref in refs:
            if ref != f"refs/heads/{branch}" and _ref_exists(root, ref):
                return name, ref
    where = f"locally or on remote {remote!r}" if remote else "locally"
    if base:
        hint = (f"Run `git fetch {remote}` to update remote branches, or " if remote else "") \
            + "pass an existing branch with --base."
        raise BranchError(f"base branch {base!r} was not found {where}", hint, category="missing_base")
    raise BranchError(
        f"could not determine the base branch (tried {', '.join(names)} {where})",
        "Pass it explicitly with --base BRANCH.",
        category="missing_base",
    )


def _commits(root: Path, base_ref: str) -> list[Commit]:
    raw = run_git(
        root, "log", "--no-color", "--no-show-signature", "--reverse",
        "--format=%H%x00%P%x00%s%x00%b%x1e", f"{base_ref}..HEAD",
    ).decode("utf-8", "replace")
    commits: list[Commit] = []
    for rec in raw.split("\x1e"):
        rec = rec.lstrip("\n")
        if not rec:
            continue
        oid, parents, subject, body = (rec.split("\x00", 3) + ["", "", ""])[:4]
        commits.append(Commit(oid, parents.split(), subject.strip(), body.strip()))
    return commits


def _changed_files(root: Path, base_ref: str) -> list[ChangedFile]:
    raw = run_git(
        root, "diff", *_DIFF_FLAGS, "--name-status", "-z", "--find-renames", f"{base_ref}...HEAD"
    ).decode("utf-8", "replace").split("\0")
    files: list[ChangedFile] = []
    i = 0
    while i < len(raw):
        code = raw[i]
        i += 1
        if not code:
            continue
        status = code[0]
        if status in ("R", "C"):
            orig, path = raw[i], raw[i + 1]
            i += 2
            files.append(ChangedFile(path, status, orig))
        else:
            files.append(ChangedFile(raw[i], status))
            i += 1
    return files


def resolve(start: Path, *, base: str | None = None, remote: str | None = None,
            since: str | None = None, for_pr: bool = True) -> BranchInfo:
    """Gather verified branch data. Raises BranchError when it cannot proceed."""
    try:
        state = read_state(start)
    except GitError as exc:
        raise BranchError(str(exc), "Run Lazy-Commit from inside a Git working tree, or pass --repo PATH.",
                          category="not_a_repository") from None
    if state.unsafe_reasons:
        raise BranchError(
            "the repository is in an unsafe state: " + "; ".join(state.unsafe_reasons),
            "Finish or abort the operation (e.g. `git merge --abort`), then run this again.",
            code=3, category="unsafe_repository_state",
        )
    if state.branch == "(detached)":
        raise BranchError("HEAD is detached; it is not on any branch",
                          "Switch to the branch you want to use (`git switch BRANCH`).",
                          category="detached_head")
    if state.head_oid is None:
        raise BranchError(f"branch {state.branch!r} has no commits yet",
                          "Commit your work first, for example with `lazycommit --review`.",
                          category="no_unique_commits")
    for value, flag in ((base, "--base"), (remote, "--remote"), (since, "--since")):
        if value is not None and not _SAFE_NAME.match(value):
            raise BranchError(f"invalid {flag} value", "Use a plain branch, remote, or tag name.", code=2,
                              category="invalid_argument")

    root, branch = state.root, state.branch
    chosen = _choose_remote(root, branch, remote, required=for_pr)
    url = _config_get(root, f"remote.{chosen}.url") if chosen else None
    github = parse_github_url(url) if url else None

    if since:
        if not _ref_exists(root, since):
            raise BranchError(f"--since {since!r} is not a commit, branch, or tag in this repository",
                              category="missing_base")
        base_name, base_ref = since, since
    else:
        base_name, base_ref = _resolve_base(root, chosen, base, branch)

    commits = _commits(root, base_ref)
    if not commits:
        raise BranchError(
            f"branch {branch!r} has no commits that are not already in {base_name!r}",
            "Commit your work on this branch first (for example with `lazycommit --review`), "
            "or choose a different base with --base.",
            category="no_unique_commits",
        )
    info = BranchInfo(state, branch, state.head_oid, chosen, github, base_name, base_ref, commits)
    if for_pr:
        info.files = _changed_files(root, base_ref)
        if not info.files:
            raise BranchError(
                f"the {len(commits)} commit(s) on {branch!r} make no net change compared to {base_name!r}",
                "Their changes cancel out or are already in the base branch, so there is nothing to review.",
                category="no_net_change",
            )
        info.shortstat = run_git(root, "diff", *_DIFF_FLAGS, "--shortstat", f"{base_ref}...HEAD") \
            .decode("utf-8", "replace").strip()
    return info


# --- model input -------------------------------------------------------------


@dataclass
class PRFile:
    file: ChangedFile
    sections: list[Section]
    binary: bool = False

    @property
    def truncated(self) -> bool:
        return any(s.truncated for s in self.sections)


@dataclass
class PRInput:
    files: list[PRFile] = field(default_factory=list)
    excluded: list[Skipped] = field(default_factory=list)  # safety checks
    manual: list[Skipped] = field(default_factory=list)  # budget etc.
    withheld: set[str] = field(default_factory=set)  # oids whose messages matched a secret rule

    @property
    def paths(self) -> set[str]:
        return {f.file.path for f in self.files}


def message_withheld(commit: Commit) -> bool:
    return secrets.contains_secret(commit.subject) or secrets.contains_secret(commit.body)


def collect_pr(info: BranchInfo, limits: Limits | None = None) -> PRInput:
    """Secret-scan and bound the branch diff. Same rules as collect.collect()."""
    limits = limits or Limits()
    result = PRInput(withheld={c.oid for c in info.commits if message_withheld(c)})
    budget = limits.max_total_chars
    for f in info.files:
        names = [f.path] + ([f.orig_path] if f.orig_path else [])
        reason = next((r for r in map(secrets.sensitive_filename, names) if r), None)
        if reason:
            result.excluded.append(Skipped(f.path, f"sensitive file type: {reason}"))
            continue
        if any(secrets.contains_secret(n) for n in names):
            result.excluded.append(Skipped(f.path, "file path matches a secret pattern"))
            continue
        try:
            text = run_git(info.root, "diff", *_DIFF_FLAGS, "--find-renames",
                           f"{info.base_ref}...HEAD", "--", *names).decode("utf-8", "replace")
        except GitError as exc:
            result.manual.append(Skipped(f.path, f"could not read diff: {secrets.redact(str(exc))}"))
            continue
        section = Section("branch changes", text or "(no textual diff)\n")
        findings = [f"{x.rule} ({_locate(section, x.line or 0)})" for x in secrets.scan_text(text)]
        if findings:
            result.excluded.append(Skipped(f.path, "possible secret: " + "; ".join(findings[:3])))
            continue
        if len(result.files) >= limits.max_files:
            result.manual.append(Skipped(f.path, f"file limit reached ({limits.max_files} files)"))
            continue
        share = min(limits.max_file_chars, budget)
        if share < _MIN_USEFUL_CHARS:
            result.manual.append(Skipped(f.path, "model input size budget exhausted"))
            continue
        bounded = _truncate(section, share)
        budget -= len(bounded.text)
        result.files.append(PRFile(f, [bounded], _is_binary_diff(text)))
    return result


# --- outgoing content scan ------------------------------------------------------


@dataclass
class OutgoingScan:
    commits: int = 0
    problems: list[str] = field(default_factory=list)  # block any push / PR creation
    unscanned: list[str] = field(default_factory=list)  # binary content; warn only


def _patch_path(header: str) -> str:
    if header.startswith("diff --cc "):
        return header[len("diff --cc "):]
    rest = header[len("diff --git a/"):] if header.startswith("diff --git a/") else header
    mid = rest.find(" b/")
    return rest[:mid] if mid >= 0 else rest


def scan_outgoing(info: BranchInfo) -> OutgoingScan:
    """Scan every commit that is not yet on the remote: messages, added files, and patches.

    Commits are scanned one by one, so a secret added in one commit and removed
    in a later one is still caught (it would be in the pushed history).
    """
    root = info.root
    if info.remote:
        oids = run_git(root, "rev-list", "HEAD", "--not", f"--remotes={info.remote}").decode().split()
    else:
        oids = [c.oid for c in info.commits]
    scan = OutgoingScan(commits=len(oids))
    for oid in oids:
        short = oid[:7]
        raw = run_git(root, "cat-file", "commit", oid).decode("utf-8", "replace")
        message = raw.split("\n\n", 1)[1] if "\n\n" in raw else ""
        rules = sorted({f.rule for f in secrets.scan_text(message)})
        if rules:
            scan.problems.append(f"commit {short}: message matches {', '.join(rules)}")
        statuses = run_git(root, "diff-tree", "-r", "--root", "--no-commit-id", "--name-status",
                           "-z", "-m", "--no-renames", oid).decode("utf-8", "replace").split("\0")
        for status, path in zip(statuses[::2], statuses[1::2]):
            reason = secrets.sensitive_filename(path)
            if reason and not status.startswith("D"):
                scan.problems.append(f"commit {short}: adds or changes {path} ({reason})")
        patch = run_git(root, "diff-tree", *_DIFF_FLAGS, "--root", "--no-commit-id", "-r", "--cc",
                        oid).decode("utf-8", "replace")
        path = "?"
        reported: set[tuple[str, str]] = set()
        for line in patch.splitlines():
            if line.startswith(("diff --git ", "diff --cc ")):
                path = _patch_path(line)
                continue
            if line.startswith("Binary files ") and line.endswith(" differ"):
                scan.unscanned.append(f"commit {short}: {path} (binary)")
                continue
            for rule, pattern in secrets.CONTENT_RULES:
                if (path, rule) not in reported and pattern.search(line):
                    reported.add((path, rule))
                    scan.problems.append(f"commit {short}: {path} contains a possible {rule}")
    return scan


# --- remote branch state ----------------------------------------------------------

NOT_PUSHED, UP_TO_DATE, FAST_FORWARD, DIVERGED = "not pushed", "up to date", "behind local", "diverged"


def remote_branch_state(info: BranchInfo) -> tuple[str, str | None]:
    """Ask the remote (read-only, network) where the branch is. Returns (state, remote oid)."""
    assert info.remote
    out = run_git(info.root, "ls-remote", "--heads", info.remote, f"refs/heads/{info.branch}")
    remote_oid = None
    for line in out.decode("utf-8", "replace").splitlines():
        oid, _, ref = line.partition("\t")
        if ref == f"refs/heads/{info.branch}":
            remote_oid = oid
    if remote_oid is None:
        return NOT_PUSHED, None
    if remote_oid == info.head:
        return UP_TO_DATE, remote_oid
    if git_succeeds(info.root, "merge-base", "--is-ancestor", remote_oid, info.head):
        return FAST_FORWARD, remote_oid
    return DIVERGED, remote_oid
