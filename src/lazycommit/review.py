"""Interactive review: approve or reject each proposed group, confirm, execute.

Nothing is committed unless the user answers "y" for a group *and* types
"yes" at the final confirmation. End of input, unrecognized answers, and
"quit" all count as "no".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TextIO

from lazycommit import render, telemetry
from lazycommit.collect import CollectResult
from lazycommit.execute import (
    APPROVED,
    BLOCKED,
    COMMITTED,
    FAILED,
    NOT_RUN,
    PROPOSED,
    REJECTED,
    Execution,
    GroupResult,
    Snapshot,
    blocked_reasons,
    execute,
    session_problems,
)
from lazycommit.gitrepo import RepoState
from lazycommit.planner import Plan
from lazycommit.render import RULE, _clean

# Returns the user's answer; raises EOFError when input ends.
Prompt = Callable[[str], str]

MAX_INVALID_ANSWERS = 3

# Outcomes returned to the CLI.
OK, FAILED_OUTCOME, STALE, INTERRUPTED = "ok", "failed", "stale", "interrupted"


def _ask(prompt: Prompt, question: str, choices: dict[str, str]) -> str | None:
    """Return the canonical choice, or None on end of input / repeated junk."""
    for _ in range(MAX_INVALID_ANSWERS):
        try:
            answer = prompt(question).strip().lower()
        except EOFError:
            return None
        if answer in choices:
            return choices[answer]
    return None


def _status_label(it: GroupResult) -> str:
    return it.status.upper()


def _line(it: GroupResult) -> str:
    commit = f"{it.commit[:12]}  " if it.commit else ""
    subject = it.subject if it.status == COMMITTED and it.subject else it.group.header
    return f"  [{it.index}] {_status_label(it):<10} {commit}{_clean(subject)}"


def review_groups(out: TextIO, prompt: Prompt, items: list[GroupResult], collected: CollectResult) -> bool:
    """Ask about each group. Returns False if the user quit the review."""
    total = len(items)
    for it in items:
        render.group(out, f"[{it.index}/{total}] {_status_label(it)}", it.group, collected)
        if it.blocked:
            it.status = BLOCKED
            out.write("    Cannot be committed safely:\n")
            for reason in it.blocked:
                out.write(f"      - {_clean(reason)}\n")
            out.write("    -> BLOCKED (rejected automatically; commit these files yourself if needed)\n")
            continue
        answer = _ask(
            prompt,
            "    Approve this commit? [y]es / [n]o / [q]uit review: ",
            {"y": "y", "yes": "y", "n": "n", "no": "n", "q": "q", "quit": "q"},
        )
        if answer in (None, "q"):
            for rest in items:
                if rest.status in (PROPOSED, APPROVED):
                    rest.status = REJECTED
            out.write("\nReview ended. Nothing was committed.\n")
            return False
        it.status = APPROVED if answer == "y" else REJECTED
        out.write(f"    -> {_status_label(it)}\n")
    return True


def _report(out: TextIO, items: list[GroupResult], run: Execution, branch: str) -> None:
    out.write("\n" + RULE + "\nRESULT\n" + RULE + "\n")
    for it in items:
        out.write(_line(it) + "\n")
        if it.status == FAILED and it.error:
            out.write(f"      error: {_clean(it.error)}\n")
    done = [it for it in items if it.status == COMMITTED]
    failed = [it for it in items if it.status == FAILED]
    not_run = [it for it in items if it.status == NOT_RUN]
    out.write(f"\n{len(done)} commit(s) created on {branch!r}. Nothing was pushed.\n")

    if run.stale:
        out.write("\nThe repository no longer matches the state the plan was generated from:\n")
        for problem in run.stale[:20]:
            out.write(f"  - {_clean(problem)}\n")
        if len(run.stale) > 20:
            out.write(f"  - ... and {len(run.stale) - 20} more\n")
    if run.stopped:
        out.write(f"\nStopped: {_clean(run.stopped)}.\n")
        if not_run:
            out.write("Not attempted: " + ", ".join(f"[{it.index}]" for it in not_run) + "\n")

    if failed or run.stale or not_run:
        out.write("\nRecovery:\n")
        if done:
            out.write("  - The commits listed as COMMITTED were kept; Lazy-Commit never undoes them.\n")
            if run.start_head:
                out.write(
                    "    To undo them yourself while keeping their changes staged, you could run:\n"
                    f"      git reset --soft {run.start_head}\n"
                )
        if any(it.status == FAILED and it.commit for it in failed):
            out.write("  - Inspect the unverified commit with `git show --stat HEAD` before continuing.\n")
        if failed:
            out.write("  - Fix the reported problem (for example a failing commit hook).\n")
        out.write("  - Remaining changes are still in your working tree and index as before.\n"
                  "  - Run `lazycommit --review` again to plan them from the current state.\n")
    else:
        out.write("Your other staged, unstaged, and untracked changes were left as they were.\n")


def _record_outcome(items: list[GroupResult], run: Execution) -> None:
    failed = [it for it in items if it.status == FAILED]
    if any(it.commit for it in failed):
        telemetry.failure("commit.execute", "verification_failed")
    elif failed:
        telemetry.failure("commit.execute", "commit_failed")
    elif run.stale:
        telemetry.failure("commit.execute", "repository_changed", "warning")


def run_review(out: TextIO, prompt: Prompt, state: RepoState, collected: CollectResult,
               plan: Plan, snapshot: Snapshot, model: str) -> str:
    items = [
        GroupResult(i, g, blocked=blocked_reasons(g, state, collected))
        for i, g in enumerate(plan.groups, start=1)
    ]
    blocked = sum(1 for it in items if it.blocked)
    if blocked:
        telemetry.count("blocked_groups", blocked)
        telemetry.failure("plan.validate", "blocked_groups", "warning")
    out.write("\n" + RULE + "\n")
    out.write(f"REVIEW PROPOSED COMMITS (model: {model}) - nothing has been staged or committed\n")
    out.write(RULE + "\n")
    if not items:
        out.write("\nThe model did not produce any valid commit groups.\n")
        render.plan_notes(out, plan, collected)
        out.write("\nNothing was committed.\n")
        return OK

    problems = session_problems(state)
    if problems:
        out.write("\nCommits cannot be created in this repository state:\n")
        for problem in problems:
            out.write(f"  - {_clean(problem)}\n")
        out.write("Nothing was committed. Run without --review to just preview the plan.\n")
        return FAILED_OUTCOME

    if not review_groups(out, prompt, items, collected):
        return OK
    render.plan_notes(out, plan, collected)

    out.write("\n" + RULE + "\nREVIEW SUMMARY\n" + RULE + "\n")
    for it in items:
        out.write(_line(it) + "\n")
    approved = [it for it in items if it.status == APPROVED]
    if not approved:
        out.write("\nNo groups were approved. Nothing was committed.\n")
        return OK

    out.write(
        f"\n{len(approved)} commit(s) will be created on branch {state.branch!r}, in the order shown,\n"
        "each containing only the files listed in its group. Other staged, unstaged, and\n"
        "untracked changes are left as they are. Nothing is pushed.\n"
    )
    try:
        answer = prompt("Type 'yes' to create these commits, or anything else to cancel: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() != "yes":
        for it in approved:
            it.status = REJECTED
        out.write("\nCancelled. Nothing was committed.\n")
        return OK

    out.write("\nRe-checking the repository and creating commits...\n")
    run = Execution(start_head=snapshot.head)
    try:
        with telemetry.span("commit.execute"):
            run = execute(state, collected, snapshot, items)
            _record_outcome(items, run)
    except KeyboardInterrupt:
        telemetry.failure("commit.execute", "interrupted", "warning")
        run.stopped = "interrupted"
        for it in items:
            if it.status == APPROVED:
                it.status = NOT_RUN
        _report(out, items, run, state.branch)
        out.write("Interrupted: check `git log` and `git status` to see exactly what was committed.\n")
        return INTERRUPTED
    _report(out, items, run, state.branch)
    if any(it.status == FAILED for it in items):
        return FAILED_OUTCOME
    return STALE if run.stale else OK
