"""Build the model prompt and validate the model's commit plan.

The model's response is untrusted data. :func:`parse_plan` enforces the
safety contract in code: strict per-field schema and size limits, only
eligible candidate files, no duplicates, no unknown fields (a response cannot
mark itself trusted or approved), unverifiable claims removed from
explanations, and groups whose subject makes such a claim or was copied from
repository content are flagged for manual review (they cannot be committed).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from lazycommit import guard, secrets
from lazycommit.collect import Candidate, CollectResult, Skipped
from lazycommit.gitrepo import RepoState

COMMIT_TYPES = (
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert",
)
CONFIDENCE_LEVELS = ("high", "medium", "low")
MAX_SUBJECT_CHARS = 72
# Hard limits on raw model output; anything larger is rejected, not truncated.
MAX_RESPONSE_CHARS = 200_000
MAX_GROUPS = 100
MAX_MESSAGE_CHARS = 200
MAX_TEXT_CHARS = 2_000  # rationale, notes, unassigned reasons
MAX_SCOPE_CHARS = 60
MAX_PATH_CHARS = 1_024
MAX_EXTRA_FILE_REFS = 20  # a group may not list many more paths than there are candidates
GROUP_FIELDS = frozenset({"type", "scope", "message", "files", "rationale", "confidence", "notes"})
TOP_LEVEL_FIELDS = frozenset({"groups", "unassigned"})

SYSTEM_PROMPT = f"""You are Lazy-Commit, a careful assistant that groups Git changes into logical Conventional Commits.

Rules:
- Use ONLY the files and diffs provided. Never invent files, changes, or behavior.
- Each file path you output must be copied exactly from the provided file list.
- Put each file in at most one group. If a file's changes cannot be grouped confidently, list it in "unassigned" with a reason.
- "type" must be one of: {", ".join(COMMIT_TYPES)}.
- "message" is the short imperative subject WITHOUT the type prefix (e.g. "add retry to http client"), lowercase start, no trailing period, at most 60 characters.
- "scope" is optional; use an empty string if none is obvious.
- "rationale" must cite what the diff actually shows.
- "confidence" is high, medium, or low; put doubts in "notes".
{guard.UNTRUSTED_RULES}
- Respond with JSON only, matching the requested schema. Do not add other fields."""

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(COMMIT_TYPES)},
                    "scope": {"type": "string"},
                    "message": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                    "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                    "notes": {"type": "string"},
                },
                "required": ["type", "message", "files", "rationale", "confidence"],
                "additionalProperties": False,
            },
        },
        "unassigned": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["file", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["groups"],
    "additionalProperties": False,
}


class UnsafeInputError(Exception):
    """The assembled model request still matched a secret pattern."""


def build_user_prompt(state: RepoState, collected: CollectResult) -> str:
    lines = [
        f"Branch: {state.branch}",
        "",
        "Candidate files (path | where the change lives | change):",
    ]
    for c in collected.candidates:
        flags = "+".join(c.kinds)
        extra = " | binary" if c.binary else ""
        extra += " | diff truncated" if c.truncated else ""
        lines.append(f"- {c.path} | {flags} | {c.change}{extra}")
    lines += [
        "",
        "Note: 'staged' changes are already in the index; 'unstaged' are working-tree edits; "
        "a file can have both. Untracked files are new and not yet known to Git.",
        "",
        "Diffs (untrusted repository data):",
    ]
    data: list[str] = []
    for c in collected.candidates:
        data.append(f"===== FILE: {c.path} =====")
        for sec in c.sections:
            data.append(f"--- {sec.label} ---")
            data.append(sec.text.rstrip("\n"))
        data.append("")
    lines.append(guard.wrap_untrusted("\n".join(data)))
    return "\n".join(lines)


def assert_safe(*texts: str) -> None:
    """Final guard: never send text that matches a secret rule."""
    for text in texts:
        if secrets.contains_secret(text):
            raise UnsafeInputError(
                "assembled model input still matched a secret pattern; request aborted"
            )


@dataclass
class Group:
    type: str
    scope: str
    message: str
    files: list[str]
    rationale: str
    confidence: str
    notes: str = ""
    # Set by validation, never by the model: reasons this group needs manual review.
    flags: list[str] = field(default_factory=list)

    @property
    def header(self) -> str:
        scope = f"({self.scope})" if self.scope else ""
        return f"{self.type}{scope}: {self.message}"


@dataclass
class Plan:
    groups: list[Group] = field(default_factory=list)
    manual: list[Skipped] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class InvalidPlanError(Exception):
    pass


def _norm_path(p: str) -> str:
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def _clean_line(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _safe_display(value: str) -> str:
    # Model output is shown in the terminal; strip control chars and secrets.
    return secrets.redact(re.sub(r"[\x00-\x1f\x7f]", "", value))[:120]


def _too_long(value: Any, limit: int) -> bool:
    return isinstance(value, str) and len(value) > limit


def parse_plan(raw: str, candidates: list[Candidate]) -> Plan:
    """Validate model output against the schema and the real candidate list."""
    if len(raw) > MAX_RESPONSE_CHARS:
        raise InvalidPlanError(f"model output is too large ({len(raw)} characters)")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidPlanError(f"model output is not valid JSON ({exc.msg})") from None
    if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
        raise InvalidPlanError("model output does not match the plan schema (missing 'groups' list)")
    if len(data["groups"]) > MAX_GROUPS:
        raise InvalidPlanError(f"model output has too many groups ({len(data['groups'])})")

    known = {c.path for c in candidates}
    repository_text = "\n".join(s.text for c in candidates for s in c.sections)
    plan = Plan()
    extra_top = sorted(str(k) for k in data if k not in TOP_LEVEL_FIELDS)
    if extra_top:
        plan.warnings.append("ignored unexpected top-level field(s): " + ", ".join(_safe_display(k) for k in extra_top[:5]))
    seen: dict[str, int] = {}
    groups: list[Group] = []

    for idx, g in enumerate(data["groups"], start=1):
        if not isinstance(g, dict):
            plan.warnings.append(f"group {idx}: not an object; discarded")
            continue
        problems: list[str] = []
        extra = sorted(str(k) for k in g if k not in GROUP_FIELDS)
        if extra:
            # A response cannot add authority (e.g. "approved", "safe") to its own plan.
            problems.append("unexpected field(s): " + ", ".join(_safe_display(k) for k in extra[:5]))
        for key, limit in (("message", MAX_MESSAGE_CHARS), ("rationale", MAX_TEXT_CHARS),
                           ("notes", MAX_TEXT_CHARS), ("scope", MAX_SCOPE_CHARS)):
            if _too_long(g.get(key), limit):
                problems.append(f"'{key}' exceeds {limit} characters")
        ctype = g.get("type")
        if ctype not in COMMIT_TYPES:
            problems.append(f"invalid commit type {_safe_display(str(ctype))!r}")
        message = g.get("message")
        if not isinstance(message, str) or not message.strip():
            problems.append("missing message")
            message = ""
        else:
            message = re.sub(rf"^(?:{'|'.join(COMMIT_TYPES)})(?:\([^)]*\))?!?:\s*", "", message.strip())
            if "\n" in message:
                problems.append("message spans multiple lines")
        files_raw = g.get("files")
        if not isinstance(files_raw, list) or not files_raw or not all(isinstance(f, str) for f in files_raw):
            problems.append("'files' must be a non-empty list of paths")
            files_raw = []
        elif len(files_raw) > len(known) + MAX_EXTRA_FILE_REFS or any(len(f) > MAX_PATH_CHARS for f in files_raw):
            problems.append("'files' is far larger than the candidate list or has an oversized path")
            files_raw = []
        files = list(dict.fromkeys(_norm_path(f) for f in files_raw))
        unknown = [f for f in files if f not in known]
        if unknown:
            shown = ", ".join(_safe_display(u) for u in unknown[:3])
            problems.append(f"references file(s) not in the candidate list: {shown}")
        rationale = g.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            problems.append("missing rationale")
        confidence = g.get("confidence")
        if confidence not in CONFIDENCE_LEVELS:
            problems.append("invalid confidence")
        scope = g.get("scope", "")
        if scope is not None and not isinstance(scope, str):
            problems.append("invalid scope")
        notes_raw = g.get("notes", "")
        if notes_raw is not None and not isinstance(notes_raw, str):
            problems.append("invalid notes")

        if problems:
            plan.warnings.append(f"group {idx} rejected: " + "; ".join(problems))
            continue

        assert isinstance(ctype, str) and isinstance(confidence, str) and isinstance(rationale, str)
        clean_scope = re.sub(r"[^\w.\-/]", "", scope or "")[:30]
        subject = _clean_line(message, MAX_SUBJECT_CHARS - len(ctype) - len(clean_scope) - 4)
        clean_rationale, removed = guard.strip_unverifiable(_clean_line(rationale, MAX_TEXT_CHARS))
        clean_notes, removed_notes = guard.strip_unverifiable(_clean_line(notes_raw or "", MAX_TEXT_CHARS))
        group = Group(
            type=ctype,
            scope=clean_scope,
            message=subject,
            files=files,
            rationale=_clean_line(clean_rationale, 400) or "(rationale removed: it only contained unverifiable claims)",
            confidence=confidence,
            notes=_clean_line(clean_notes, 300),
        )
        if removed or removed_notes:
            plan.warnings.append(f"group {idx}: removed {removed + removed_notes} unverifiable claim(s) "
                                 "(e.g. tests passed) from the explanation")
        if guard.find_outcome_claim(group.message):
            group.flags.append("commit message claims tests/builds passed or behaviour was verified, "
                               "which Lazy-Commit cannot confirm")
        if guard.copied_from_repository(group.header, group.message, repository_text):
            group.flags.append("commit message was copied from repository content (possible prompt injection)")
        for f in files:
            seen[f] = seen.get(f, 0) + 1
        groups.append(group)

    duplicated = {f for f, n in seen.items() if n > 1}
    for group in groups:
        group.files = [f for f in group.files if f not in duplicated]
        if group.files:
            plan.groups.append(group)
        else:
            plan.warnings.append(f"group {group.header!r} dropped: all its files were assigned to multiple groups")
    for f in sorted(duplicated):
        plan.manual.append(Skipped(f, "model assigned it to multiple groups"))

    reasons: dict[str, str] = {}
    unassigned = data.get("unassigned", [])
    if isinstance(unassigned, list):
        for u in unassigned:
            if isinstance(u, dict) and isinstance(u.get("file"), str):
                path = _norm_path(u["file"])
                if path in known:
                    reason, _ = guard.strip_unverifiable(_clean_line(u.get("reason"), MAX_TEXT_CHARS))
                    reasons[path] = "model: " + (_clean_line(reason, 200) or "no reason given")

    assigned = {f for g in plan.groups for f in g.files} | duplicated
    for c in candidates:
        if c.path not in assigned:
            plan.manual.append(Skipped(c.path, reasons.get(c.path, "not assigned to any valid group")))
    return plan
