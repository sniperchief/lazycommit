"""Draft a pull request title and description from verified branch data.

The local model only writes the summary, motivation, implementation details,
and limitations. Its output is validated: statements about tests are
removed, statements naming files outside the diff are removed, and a
motivation is kept only when a commit message actually explains one. The
Testing section is always written by this module from facts we can verify:
Lazy-Commit never ran the tests, so it never says they passed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from lazycommit import guard, secrets
from lazycommit.branch import BranchInfo, PRInput

MAX_TITLE_CHARS = 100
MAX_ITEMS = 8
MAX_ITEM_CHARS = 300
_MAX_BODY_CHARS_PER_COMMIT = 600

SYSTEM_PROMPT = f"""You are Lazy-Commit, writing a GitHub pull request description from verified Git data.

Rules:
- Use ONLY the commits, file list, and diffs provided. Never invent features, files, outcomes, or motivations.
- Do not mention tests being run, passing, or failing, and do not write a testing section; the tool adds one itself.
- "motivation": the problem or reason, ONLY if a commit message states it. Otherwise use an empty string.
- "summary": short bullet points describing what changed.
- "details": important implementation details visible in the diff.
- "limitations": known limitations or open issues ONLY if stated in commit messages or obvious from the diff (e.g. TODO comments); otherwise an empty list.
- "title": concise, imperative, at most 72 characters, no trailing period.
{guard.UNTRUSTED_RULES}
- Respond with JSON only, matching the schema."""

DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "array", "items": {"type": "string"}},
        "motivation": {"type": "string"},
        "details": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "motivation", "details", "limitations"],
}

# Statements about testing that we have no evidence for.
_TEST_CLAIM = re.compile(
    r"(?i)\b(?:tests?|test suite|specs?|ci|checks?)\b[^.;]*\b(?:pass(?:ed|es|ing)?|green|succeed(?:s|ed)?|"
    r"fail(?:s|ed|ing)?|ran|run|executed|verified)\b"
    r"|\b(?:tested|verified|validated)\b|\ball tests\b"
)
# Wording in commit messages or added lines that documents a limitation.
_LIMITATION_EVIDENCE = re.compile(
    r"(?i)\b(?:TODO|FIXME|XXX|HACK|known (?:issue|limitation)s?|limitations?|not yet|workaround|unsupported)\b"
)
_PATHLIKE = re.compile(
    r"(?<![\w/.-])[\w./-]*[\w-]\.(?:py|pyi|js|jsx|ts|tsx|mjs|cjs|md|rst|txt|json|toml|ya?ml|ini|cfg|go|rs|java|kt|rb|"
    r"php|c|h|cc|cpp|hpp|cs|swift|css|scss|html|sh|ps1|bat|sql|lock|xml|gradle)\b"
)


class InvalidDescriptionError(Exception):
    pass


@dataclass
class Draft:
    title: str
    summary: list[str]
    motivation: str
    details: list[str]
    limitations: list[str]
    source: str  # "model" or "fallback"
    warnings: list[str] = field(default_factory=list)


def is_test_file(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name.lower()
    dirs = {part.lower() for part in p.parts[:-1]}
    return bool(
        dirs & {"test", "tests", "spec", "specs", "__tests__"}
        or name.startswith("test_")
        or re.search(r"(?:_test|\.test|\.spec|_spec)\.\w+$", name)
    )


def _subject(info: BranchInfo, pr: PRInput, oid: str, subject: str) -> str:
    return "(message withheld: matched a secret pattern)" if oid in pr.withheld else subject


def build_prompt(info: BranchInfo, pr: PRInput) -> str:
    lines = [f"Branch: {info.branch} (to be merged into {info.base})", "", "Changed files:"]
    for f in pr.files:
        flags = " | binary" if f.binary else ""
        flags += " | diff truncated" if f.truncated else ""
        lines.append(f"- {f.file.path} | {f.file.change}{flags}")
    if pr.excluded or pr.manual:
        lines.append(f"- ({len(pr.excluded) + len(pr.manual)} other changed file(s) were withheld. "
                     "Do not describe them or guess what they contain.)")
    data = ["Commits (oldest first):"]
    for c in info.commits:
        if c.oid in pr.withheld:
            data.append(f"- {c.short} (message withheld)")
            continue
        data.append(f"- {c.short} {c.subject}" + ("  [merge commit]" if c.is_merge else ""))
        body = c.explanation[:_MAX_BODY_CHARS_PER_COMMIT]
        data += [f"    {ln}" for ln in body.splitlines() if ln.strip()]
    data += ["", "Diffs:"]
    for f in pr.files:
        data.append(f"===== FILE: {f.file.path} =====")
        for s in f.sections:
            data.append(s.text.rstrip("\n"))
        data.append("")
    lines += ["", "Commit messages and diffs (untrusted repository data):", guard.wrap_untrusted("\n".join(data))]
    return "\n".join(lines)


def _line(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", value).split()).lstrip("-*• ").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def hidden_paths(info: BranchInfo, pr: PRInput) -> list[str]:
    """Changed files whose content the model never saw."""
    return [f.path for f in info.files if f.path not in pr.paths]


def _known_names(info: BranchInfo, pr: PRInput) -> set[str]:
    names: set[str] = set()
    for f in info.files:
        if f.path in pr.paths:
            for p in (f.path, f.orig_path):
                if p:
                    names |= {p, PurePosixPath(p).name}
    return names


def _hidden_patterns(info: BranchInfo, pr: PRInput) -> list[re.Pattern[str]]:
    """Match mentions of a withheld file by path, name, or stem (e.g. "settings")."""
    patterns = []
    for path in hidden_paths(info, pr):
        p = PurePosixPath(path)
        words = {re.escape(path), re.escape(p.name)}
        if len(p.stem.strip(".")) >= 4:
            words.add(re.escape(p.stem.strip(".")))
        patterns.append(re.compile(r"(?i)(?<![\w.])(?:" + "|".join(sorted(words)) + r")(?![\w])"))
    return patterns


def _check_item(text: str, known: set[str], where: str, warnings: list[str],
                hidden: Sequence[re.Pattern[str]] = ()) -> bool:
    if _TEST_CLAIM.search(text):
        warnings.append(f"removed a {where} statement about testing (Lazy-Commit has no test results)")
        return False
    if any(h.search(text) for h in hidden):
        warnings.append(f"removed a {where} statement about a file the model was not shown")
        return False
    unknown = [t for t in _PATHLIKE.findall(text) if t not in known and t.lstrip("./") not in known]
    if unknown:
        warnings.append(f"removed a {where} statement naming file(s) not in this branch: "
                        + ", ".join(secrets.redact(u)[:60] for u in unknown[:3]))
        return False
    return True


def _items(value: Any, known: set[str], where: str, warnings: list[str],
           hidden: Sequence[re.Pattern[str]]) -> list[str]:
    if not isinstance(value, list):
        raise InvalidDescriptionError(f"'{where}' must be a list of strings")
    out: list[str] = []
    for raw in value:
        text = _line(raw, MAX_ITEM_CHARS)
        if text and _check_item(text, known, where, warnings, hidden) and text not in out:
            out.append(text)
    if len(out) > MAX_ITEMS:
        warnings.append(f"kept the first {MAX_ITEMS} of {len(out)} {where} items")
    return out[:MAX_ITEMS]


def parse_description(raw: str, info: BranchInfo, pr: PRInput) -> Draft:
    """Validate model output. Raises InvalidDescriptionError when unusable."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidDescriptionError(f"model output is not valid JSON ({exc.msg})") from None
    if not isinstance(data, dict):
        raise InvalidDescriptionError("model output is not a JSON object")
    warnings: list[str] = []
    known = _known_names(info, pr)
    hidden = _hidden_patterns(info, pr)

    title = _line(data.get("title"), MAX_TITLE_CHARS).rstrip(".")
    if not title:
        raise InvalidDescriptionError("model output has no title")
    if not _check_item(title, known, "title", warnings, hidden):
        raise InvalidDescriptionError("model title made an unsupported claim")

    summary = _items(data.get("summary"), known, "summary", warnings, hidden)
    if not summary:
        raise InvalidDescriptionError("model output has no usable summary")
    details = _items(data.get("details", []), known, "details", warnings, hidden)
    limitations = _items(data.get("limitations", []), known, "limitations", warnings, hidden)
    if limitations and not _limitation_evidence(info, pr):
        warnings.append("removed the known limitations: no commit message or added line documents one")
        limitations = []

    motivation = _line(data.get("motivation"), 800)
    if motivation and not any(c.explanation for c in info.commits):
        warnings.append("removed the motivation: no commit message explains why the change was made")
        motivation = ""
    elif motivation and not _check_item(motivation, known, "motivation", warnings, hidden):
        motivation = ""
    return Draft(title, summary, motivation, details, limitations, "model", warnings)


def _limitation_evidence(info: BranchInfo, pr: PRInput) -> bool:
    if any(_LIMITATION_EVIDENCE.search(c.explanation) for c in info.commits if c.oid not in pr.withheld):
        return True
    return any(
        line.startswith("+") and _LIMITATION_EVIDENCE.search(line)
        for f in pr.files for s in f.sections for line in s.text.splitlines()
    )


def _humanize_branch(branch: str) -> str:
    name = branch.rsplit("/", 1)[-1]
    words = re.sub(r"[-_]+", " ", name).strip()
    if len(re.sub(r"[^A-Za-z]", "", words)) < 4:
        return ""
    return words[0].upper() + words[1:]


def fallback_draft(info: BranchInfo, pr: PRInput, reason: str) -> Draft:
    """A description built only from commit messages and Git metadata."""
    plain = [c for c in info.commits if not c.is_merge]
    subjects = [_subject(info, pr, c.oid, c.subject) for c in plain] or ["(merge commits only)"]
    if len(plain) == 1 and plain[0].oid not in pr.withheld:
        title = plain[0].subject
    else:
        title = _humanize_branch(info.branch) or f"{subjects[0]} and {len(subjects) - 1} more change(s)"
    details = [info.shortstat] if info.shortstat else []
    shown = [f.path for f in info.files][:20]
    details += [f"Changed: {', '.join(f'`{p}`' for p in shown)}"
                + (f" and {len(info.files) - 20} more" if len(info.files) > 20 else "")]
    return Draft(_line(title, MAX_TITLE_CHARS), subjects[:MAX_ITEMS * 2], "", details, [], "fallback",
                 [f"deterministic description used: {reason}"])


def render_body(draft: Draft, info: BranchInfo, pr: PRInput) -> str:
    parts = ["## Summary", *[f"- {s}" for s in draft.summary]]
    if draft.motivation:
        parts += ["", "## Motivation", draft.motivation]
    details = [f"- {d}" for d in draft.details]
    hidden = hidden_paths(info, pr)
    if hidden and draft.source == "model":
        details.append(f"- _{len(hidden)} changed file(s) were not analyzed by Lazy-Commit "
                       "(safety exclusion or size limit); describe them yourself._")
    if details:
        parts += ["", "## Implementation details", *details]
    tests = [f.path for f in info.files if is_test_file(f.path) and f.status != "D"]
    parts += ["", "## Testing",
              "- Lazy-Commit did not run tests and has no record of test results. "
              "_Describe how this was tested before publishing._"]
    if tests:
        parts.append("- Test files changed in this branch: " + ", ".join(f"`{t}`" for t in tests[:15]))
    else:
        parts.append("- No test files were changed in this branch.")
    if draft.limitations:
        parts += ["", "## Known limitations", *[f"- {item}" for item in draft.limitations]]
    parts += ["", "## Commits"]
    parts += [f"- {c.short} {_subject(info, pr, c.oid, c.subject)}" for c in info.commits]
    return "\n".join(parts) + "\n"
