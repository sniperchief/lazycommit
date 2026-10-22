"""Changelog sections built deterministically from verified commits.

Conventional Commit types are taken as confirmed categories (the author chose
them). Anything else is listed separately as unclassified, with at most a
wording-based guess clearly labelled as such. Nothing is written without
explicit approval, and the file is re-checked right before writing.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from lazycommit import secrets
from lazycommit.branch import Commit

CATEGORIES: dict[str, str] = {
    "feat": "Features",
    "fix": "Bug fixes",
    "perf": "Performance",
    "refactor": "Refactoring",
    "docs": "Documentation",
    "test": "Tests",
    "build": "Build and CI",
    "ci": "Build and CI",
    "style": "Chores",
    "chore": "Chores",
    "revert": "Reverts",
}
_ORDER = ["Features", "Bug fixes", "Performance", "Refactoring", "Documentation", "Tests",
          "Build and CI", "Chores", "Reverts"]
_HEADER = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^()\r\n]{1,40})\))?(?P<bang>!)?: (?P<desc>\S.*)$")
_GUESSES = [
    (re.compile(r"(?i)^(?:fix|fixes|fixed|bugfix|resolve[sd]?|correct)\b"), "Bug fixes"),
    (re.compile(r"(?i)^(?:add|adds|added|implement|introduce|support|create|allow|enable)\b"), "Features"),
    (re.compile(r"(?i)\b(?:docs?|readme|documentation)\b"), "Documentation"),
    (re.compile(r"(?i)^(?:refactor|clean ?up|rename|move|simplif\w*|extract)\b"), "Refactoring"),
    (re.compile(r"(?i)\btests?\b"), "Tests"),
]


@dataclass
class Entry:
    short: str
    text: str
    scope: str = ""
    category: str = ""  # confirmed category, or "" when unclassified
    guess: str = ""  # wording-based guess for unclassified entries
    breaking: bool = False


@dataclass
class Changelog:
    entries: list[Entry] = field(default_factory=list)
    merges_skipped: int = 0
    withheld: int = 0


def _clean(text: str) -> str:
    return " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", text).split()).replace("<", "&lt;")


def build(commits: list[Commit]) -> Changelog:
    log = Changelog()
    for c in commits:
        if c.is_merge:
            log.merges_skipped += 1
            continue
        if secrets.contains_secret(c.subject) or secrets.contains_secret(c.body):
            log.withheld += 1
            log.entries.append(Entry(c.short, "(commit message withheld: it matched a secret pattern)"))
            continue
        m = _HEADER.match(c.subject)
        breaking = "BREAKING CHANGE:" in c.body or "BREAKING-CHANGE:" in c.body
        if m and m.group("type") in CATEGORIES:
            log.entries.append(Entry(
                c.short, _clean(m.group("desc")), _clean(m.group("scope") or ""),
                CATEGORIES[m.group("type")], breaking=breaking or bool(m.group("bang")),
            ))
        else:
            text = _clean(m.group("desc") if m else c.subject)
            guess = next((label for pattern, label in _GUESSES if pattern.search(text)), "")
            log.entries.append(Entry(c.short, text, guess=guess, breaking=breaking))
    return log


def _bullet(e: Entry) -> str:
    scope = f"**{e.scope}:** " if e.scope else ""
    return f"- {scope}{e.text} ({e.short})"


def render(log: Changelog, title: str, date: str) -> str:
    lines = [f"## {_clean(title)} - {date}", ""]
    breaking = [e for e in log.entries if e.breaking]
    if breaking:
        lines += ["### Breaking changes", *[_bullet(e) for e in breaking], ""]
    for category in _ORDER:
        items = [e for e in log.entries if e.category == category]
        if items:
            lines += [f"### {category}", *[_bullet(e) for e in items], ""]
    unclassified = [e for e in log.entries if not e.category]
    if unclassified:
        lines += ["### Unclassified (please review)",
                  "_These commits do not use a Conventional Commit type; any category shown is a "
                  "guess from their wording._"]
        for e in unclassified:
            hint = f" - likely: {e.guess}" if e.guess else ""
            lines.append(_bullet(e) + hint)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# --- writing ------------------------------------------------------------------


class ChangelogWriteError(Exception):
    pass


def target_path(root: Path, rel: str) -> Path:
    """Resolve the changelog path, refusing anything outside the repository."""
    candidate = Path(rel)
    if candidate.is_absolute() or rel.startswith("-"):
        raise ChangelogWriteError("--file must be a path relative to the repository root")
    full = root / candidate
    real_root = os.path.realpath(root)
    real = os.path.realpath(full)
    if os.path.commonpath([real_root, real]) != real_root or ".git" in Path(real).relative_to(real_root).parts:
        raise ChangelogWriteError("--file must be inside the repository and outside .git")
    if full.is_symlink():
        raise ChangelogWriteError(f"{rel} is a symbolic link; refusing to write through it")
    if full.exists() and not full.is_file():
        raise ChangelogWriteError(f"{rel} exists and is not a regular file")
    return full


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def insert_section(existing: str | None, section: str) -> str:
    """Place the new section above earlier releases, below any title/intro."""
    if not existing or not existing.strip():
        return "# Changelog\n\n" + section
    newline = "\r\n" if "\r\n" in existing else "\n"
    section = section.replace("\n", newline)
    lines = existing.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("## "):
            return "".join(lines[:i]) + section + newline + "".join(lines[i:])
    sep = "" if existing.endswith(("\n", "\r\n")) else newline
    return existing + sep + newline + section


def write_section(path: Path, section: str, expected: str) -> None:
    """Insert ``section`` if the file still matches ``expected``; atomic replace."""
    if fingerprint(path) != expected:
        raise ChangelogWriteError(f"{path.name} changed since the preview was generated; nothing was written")
    existing = None
    if path.exists():
        try:
            existing = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raise ChangelogWriteError(f"{path.name} is not UTF-8 text; nothing was written") from None
    data = insert_section(existing, section).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".lazycommit-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
