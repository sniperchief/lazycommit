"""Plain-text terminal rendering. Never prints file contents or secret values."""

from __future__ import annotations

import re
from typing import Any, TextIO

from lazycommit import __version__
from lazycommit.collect import CollectResult, Skipped
from lazycommit.gitrepo import RepoState
from lazycommit.planner import Group, Plan
from lazycommit.secrets import redact

RULE = "-" * 72


def _clean(text: str) -> str:
    return redact(re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text))


def skipped(out: TextIO, title: str, items: list[Skipped]) -> None:
    if not items:
        return
    out.write(f"\n{title} ({len(items)}):\n")
    for item in items:
        out.write(f"  - {_clean(item.path)}: {_clean(item.reason)}\n")


def header(out: TextIO, state: RepoState, review: bool = False) -> None:
    if review:
        out.write("Lazy-Commit (review mode - commits are created only after your explicit approval)\n")
    else:
        out.write("Lazy-Commit (preview only - no Git changes will be made)\n")
    out.write(RULE + "\n")
    out.write(f"Repository: {state.root}\n")
    head = state.head_oid[:12] if state.head_oid else "no commits yet"
    out.write(f"Branch:     {state.branch} ({head})\n")
    out.write(
        f"Changes:    {len(state.staged)} staged, {len(state.unstaged)} unstaged, "
        f"{len(state.untracked)} untracked\n"
    )
    both = [f.path for f in state.files if f.staged and f.unstaged]
    if both:
        out.write(f"            {len(both)} file(s) have both staged and unstaged changes\n")


def unsafe_state(out: TextIO, state: RepoState) -> None:
    out.write("\nThe repository is in a state Lazy-Commit will not analyze:\n")
    for reason in state.unsafe_reasons:
        out.write(f"  - {reason}\n")
    for f in state.conflicted:
        out.write(f"      conflicted: {_clean(f.path)}\n")
    out.write(
        "\nNext steps: resolve conflicts and finish or abort the operation "
        "(e.g. `git merge --continue` / `git merge --abort`), then run Lazy-Commit again.\n"
        "Nothing was sent to the model.\n"
    )


def collection(out: TextIO, collected: CollectResult) -> None:
    out.write(f"\nFiles prepared for local analysis: {len(collected.candidates)}\n")
    truncated = [c.path for c in collected.candidates if c.truncated]
    if truncated:
        out.write(f"  ({len(truncated)} with diffs truncated to fit the input budget)\n")
    skipped(out, "Excluded by safety checks (not sent to the model)", collected.excluded)
    if collected.excluded:
        out.write("  Secret detection is best-effort; review these files yourself.\n")


def group(out: TextIO, label: str, g: Group, collected: CollectResult) -> None:
    out.write(f"\n{label} {_clean(g.header)}\n")
    out.write(f"    confidence: {g.confidence}\n")
    for f in g.files:
        cand = next((c for c in collected.candidates if c.path == f), None)
        tag = f"  [{'+'.join(cand.kinds)}]" if cand else ""
        if cand and cand.status.orig_path:
            tag += f" (renamed from {_clean(cand.status.orig_path)})"
        out.write(f"    - {_clean(f)}{tag}\n")
    out.write(f"    why: {_clean(g.rationale)}\n")
    if g.notes:
        out.write(f"    notes: {_clean(g.notes)}\n")


def plan_notes(out: TextIO, result: Plan, collected: CollectResult) -> None:
    """Files the plan does not cover, and discarded model output."""
    skipped(out, "Needs manual review", result.manual + collected.manual)
    if collected.excluded:
        out.write(
            f"\nThe {len(collected.excluded)} file(s) excluded by safety checks above "
            "were not analyzed and also need manual review.\n"
        )
    if result.warnings:
        out.write("\nModel output problems (discarded):\n")
        for w in result.warnings:
            out.write(f"  - {_clean(w)}\n")


def plan(out: TextIO, result: Plan, collected: CollectResult, model: str) -> None:
    out.write("\n" + RULE + "\n")
    out.write(f"PROPOSED COMMIT PLAN (model: {model}) - proposal only, nothing was staged or committed\n")
    out.write(RULE + "\n")
    if not result.groups:
        out.write("\nThe model did not produce any valid commit groups.\n")
    for i, g in enumerate(result.groups, start=1):
        group(out, f"[{i}]", g, collected)
    plan_notes(out, result, collected)
    out.write(
        "\nThis is a proposal. Lazy-Commit did not stage, commit, or modify anything.\n"
        "To review it and approve individual commits, run again with --review.\n"
    )


def plan_dict(state: RepoState, result: Plan, collected: CollectResult, model: str,
              blocked: list[list[str]]) -> dict[str, Any]:
    """Machine-readable preview for scripts. It never implies approval."""
    kinds = {c.path: c.kinds for c in collected.candidates}
    return {
        "lazycommit_version": __version__,
        "status": "proposed",
        "committed": False,
        "repository": str(state.root),
        "branch": state.branch,
        "head": state.head_oid,
        "model": model,
        "groups": [
            {
                "index": i,
                "header": _clean(g.header),
                "type": g.type,
                "scope": g.scope,
                "message": _clean(g.message),
                "files": [{"path": _clean(f), "changes": kinds.get(f, [])} for f in g.files],
                "rationale": _clean(g.rationale),
                "confidence": g.confidence,
                "notes": _clean(g.notes),
                "status": "proposed",
                "commit_safe": not reasons,
                "blocked_reasons": [_clean(r) for r in reasons],
            }
            for i, (g, reasons) in enumerate(zip(result.groups, blocked), start=1)
        ],
        "needs_manual_review": [
            {"path": _clean(s.path), "reason": _clean(s.reason)} for s in result.manual + collected.manual
        ],
        "excluded": [{"path": _clean(s.path), "reason": _clean(s.reason)} for s in collected.excluded],
        "warnings": [_clean(w) for w in result.warnings],
    }


def error(out: TextIO, message: str, hint: str = "") -> None:
    out.write(f"\nError: {_clean(message)}\n")
    if hint:
        out.write(f"Next step: {_clean(hint)}\n")
