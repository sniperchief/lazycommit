"""Turn repository state into sanitized, bounded model input.

Order of operations for every file is fixed:

1. Sensitive filename check (content is never read for these files).
2. Read the full diff / untracked content.
3. Secret scan of the *full* content (before truncation).
4. Only clean files are truncated to the size budget and become candidates.

Any file with a finding is excluded entirely; we do not attempt partial
redaction because reliably proving a redacted diff is safe is not possible.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from lazycommit import gitrepo, secrets
from lazycommit.gitrepo import FileStatus, RepoState

DEFAULT_MAX_FILE_CHARS = 8_000
DEFAULT_MAX_TOTAL_CHARS = 48_000
DEFAULT_MAX_FILES = 150
# Untracked files larger than this cannot be fully secret-scanned cheaply,
# so they are routed to manual review instead.
MAX_SCAN_BYTES = 2_000_000
_MIN_USEFUL_CHARS = 400

_STATUS_WORDS = {
    "M": "modified",
    "T": "type changed",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "U": "conflicted",
    "?": "untracked",
}


@dataclass
class Limits:
    max_file_chars: int = DEFAULT_MAX_FILE_CHARS
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS
    max_files: int = DEFAULT_MAX_FILES


@dataclass
class Section:
    label: str
    text: str
    truncated: bool = False


@dataclass
class Candidate:
    path: str
    status: FileStatus
    sections: list[Section] = field(default_factory=list)
    binary: bool = False

    @property
    def kinds(self) -> list[str]:
        st = self.status
        if st.untracked:
            return ["untracked"]
        out = []
        if st.staged:
            out.append("staged")
        if st.unstaged:
            out.append("unstaged")
        return out

    @property
    def change(self) -> str:
        st = self.status
        if st.untracked:
            return "new untracked file"
        codes = [c for c in (st.index_status, st.worktree_status) if c != "."]
        words = sorted({_STATUS_WORDS.get(c, c) for c in codes})
        text = ", ".join(words)
        if st.orig_path:
            text += f" (from {st.orig_path})"
        return text

    @property
    def truncated(self) -> bool:
        return any(s.truncated for s in self.sections)

    @property
    def size(self) -> int:
        return sum(len(s.text) for s in self.sections)


@dataclass
class Skipped:
    path: str
    reason: str


@dataclass
class CollectResult:
    candidates: list[Candidate] = field(default_factory=list)
    # Excluded by safety checks (sensitive name or secret finding).
    excluded: list[Skipped] = field(default_factory=list)
    # Not sent for other reasons (budget, unreadable, conflicts...).
    manual: list[Skipped] = field(default_factory=list)

    @property
    def candidate_paths(self) -> set[str]:
        return {c.path for c in self.candidates}


def _is_binary_diff(text: str) -> bool:
    return any(
        line.startswith("Binary files ") and line.endswith(" differ")
        for line in text.splitlines()
    )


def _within(root: Path, path: Path) -> bool:
    try:
        real_root = os.path.realpath(root)
        real = os.path.realpath(path)
        return os.path.commonpath([real_root, real]) == real_root
    except ValueError:  # different drives on Windows
        return False


def _untracked_section(root: Path, st: FileStatus) -> tuple[Section | None, bool, str | None]:
    """Return (section, is_binary, skip_reason) for an untracked file."""
    full = root / st.path
    if full.is_symlink():
        return Section("untracked symlink", "(symbolic link; target not read)"), False, None
    if not _within(root, full):
        return None, False, "resolves outside the repository"
    try:
        size = full.stat().st_size
        if size > MAX_SCAN_BYTES:
            return None, False, f"too large to secret-scan ({size} bytes)"
        data = full.read_bytes()
    except OSError as exc:
        return None, False, f"unreadable ({exc.strerror or type(exc).__name__})"
    if b"\0" in data[:8000]:
        return Section("untracked new file", f"(binary file, {len(data)} bytes; content not shown)"), True, None
    text = data.decode("utf-8", "replace")
    body = "".join(f"+{line}\n" for line in text.splitlines())
    return Section("untracked new file", body or "(empty file)\n"), False, None


_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _locate(section: Section, diff_line: int) -> str:
    """Map a line number within a section to a line in the actual file."""
    if section.label.startswith("untracked"):
        return f"line {diff_line}"
    old = new = 0
    in_hunk = False
    for n, line in enumerate(section.text.splitlines(), start=1):
        m = _HUNK.match(line)
        if m:
            old, new, in_hunk = int(m.group(1)), int(m.group(2)), True
            if n == diff_line:
                return "diff header"
            continue
        if not in_hunk:
            if n == diff_line:
                return "diff header"
            continue
        if line.startswith("-"):
            if n == diff_line:
                return f"removed line {old}"
            old += 1
        elif line.startswith("+"):
            if n == diff_line:
                return f"line {new}"
            new += 1
        else:
            if n == diff_line:
                return f"line {new}"
            old += 1
            new += 1
    return "unknown line"


def _truncate(section: Section, limit: int) -> Section:
    if len(section.text) <= limit:
        return section
    cut = section.text[:limit]
    nl = cut.rfind("\n")
    if nl > limit // 2:
        cut = cut[: nl + 1]
    omitted = len(section.text) - len(cut)
    return Section(section.label, cut + f"\n[... truncated, {omitted} more characters not shown ...]\n", True)


def collect(state: RepoState, limits: Limits | None = None) -> CollectResult:
    limits = limits or Limits()
    result = CollectResult()
    budget = limits.max_total_chars

    ordered = state.staged + [f for f in state.unstaged if not f.staged] + state.untracked
    for st in ordered:
        # 0. States we refuse to interpret.
        if st.conflicted:
            result.manual.append(Skipped(st.path, "unresolved merge conflict"))
            continue

        # 1. Sensitive filenames: excluded before any content is read.
        name_reason = secrets.sensitive_filename(st.path) or (
            secrets.sensitive_filename(st.orig_path) if st.orig_path else None
        )
        if name_reason:
            result.excluded.append(Skipped(st.path, f"sensitive file type: {name_reason}"))
            continue
        if secrets.contains_secret(st.path):
            result.excluded.append(Skipped(st.path, "file path matches a secret pattern"))
            continue

        # 2. Gather full content.
        sections: list[Section] = []
        binary = False
        skip_reason: str | None = None
        try:
            if st.untracked:
                sec, binary, skip_reason = _untracked_section(state.root, st)
                if sec:
                    sections.append(sec)
            else:
                if st.staged:
                    text = gitrepo.file_diff(state.root, st, staged=True).decode("utf-8", "replace")
                    binary |= _is_binary_diff(text)
                    sections.append(Section("staged changes (index vs HEAD)", text or "(no textual diff)\n"))
                if st.unstaged:
                    text = gitrepo.file_diff(state.root, st, staged=False).decode("utf-8", "replace")
                    binary |= _is_binary_diff(text)
                    sections.append(Section("unstaged changes (working tree vs index)", text or "(no textual diff)\n"))
        except gitrepo.GitError as exc:
            skip_reason = f"could not read diff: {secrets.redact(str(exc))}"
        if skip_reason:
            result.manual.append(Skipped(st.path, skip_reason))
            continue

        # 3. Secret scan of the full, untruncated content.
        findings: list[str] = []
        for sec in sections:
            where = sec.label.split(" (")[0]
            findings += [
                f"{f.rule} ({_locate(sec, f.line or 0)}, {where})" for f in secrets.scan_text(sec.text)
            ]
        if findings:
            shown = "; ".join(findings[:3]) + (f"; +{len(findings) - 3} more" if len(findings) > 3 else "")
            result.excluded.append(Skipped(st.path, f"possible secret: {shown}"))
            continue

        # 4. Bound the input.
        if len(result.candidates) >= limits.max_files:
            result.manual.append(Skipped(st.path, f"file limit reached ({limits.max_files} files per run)"))
            continue
        per_file = min(limits.max_file_chars, budget)
        if per_file < _MIN_USEFUL_CHARS:
            result.manual.append(Skipped(st.path, "model input size budget exhausted"))
            continue
        share = max(per_file // max(len(sections), 1), _MIN_USEFUL_CHARS // 2)
        bounded = [_truncate(s, share) for s in sections]
        cand = Candidate(st.path, st, bounded, binary)
        budget -= cand.size
        result.candidates.append(cand)

    return result
