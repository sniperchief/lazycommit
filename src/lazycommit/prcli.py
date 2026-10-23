"""``lazycommit pr`` and ``lazycommit changelog``.

``pr`` previews a pull request title and description (read-only, no network
except the local model). ``pr --publish`` is the only path that pushes or
creates a pull request, and each of those steps needs its own typed
confirmation. ``changelog`` previews a section; ``changelog --write`` adds it
to the file only after confirmation, and never stages or commits it.

Exit codes follow the main CLI: 0 success/cancelled, 1 error, 2 usage,
3 unsafe repository state, 4 something changed after the preview, 130
interrupted.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO

from lazycommit import __version__, branch, changelog, prwriter, render, telemetry
from lazycommit.branch import BranchError, BranchInfo, PRInput
from lazycommit.collect import Limits
from lazycommit.github import INSTALL_HINT, ExistingPR, GitHub, GitHubError, find_gh, push_branch
from lazycommit.gitrepo import GitError, read_state
from lazycommit.ollama import DEFAULT_HOST, DEFAULT_TIMEOUT_SECONDS, ModelError, OllamaClient
from lazycommit.planner import UnsafeInputError, assert_safe
from lazycommit.render import RULE, _clean
from lazycommit.review import Prompt, _ask
from lazycommit.secrets import scan_text

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_STALE, EXIT_INTERRUPTED = 0, 1, 2, 4, 130
COMMANDS = ("pr", "changelog")


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", type=Path, default=Path.cwd(), help="path inside the repository (default: cwd)")
    p.add_argument("--base", help="branch the work will be merged into (default: the remote's default "
                                  "branch, else main/master)")
    p.add_argument("--remote", help="remote to use (default: the branch's upstream, else origin)")


def build_pr_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lazycommit pr",
        description="Draft a pull request for the current branch from its verified commits and diff. "
                    "By default this is a preview: nothing is pushed and no pull request is created.",
    )
    _common(p)
    p.add_argument("--model", default=os.environ.get("LAZYCOMMIT_MODEL"),
                   help="Ollama model name (default: $LAZYCOMMIT_MODEL)")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
                   help=f"local Ollama URL (default: $OLLAMA_HOST or {DEFAULT_HOST})")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                   help="seconds to wait for the model (default: %(default)s)")
    p.add_argument("--max-file-chars", type=_positive_int, default=Limits.max_file_chars)
    p.add_argument("--max-total-chars", type=_positive_int, default=Limits.max_total_chars)
    p.add_argument("--max-files", type=_positive_int, default=Limits.max_files)
    p.add_argument("--no-model", action="store_true",
                   help="skip the model and build the description from commit messages only")
    p.add_argument("--draft", action="store_true", help="with --publish: create the pull request as a draft")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--show-input", action="store_true",
                      help="print the exact sanitized model input and exit")
    mode.add_argument("--json", action="store_true",
                      help="print the preview as JSON on stdout (progress on stderr); never publishes")
    mode.add_argument("--publish", action="store_true",
                      help="review, edit, and (after confirmation) push and create the pull request "
                           "with the GitHub CLI; requires a terminal")
    return p


def build_changelog_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lazycommit changelog",
        description="Preview a changelog section from the commits on this branch (or since a tag). "
                    "Nothing is written unless you pass --write and confirm.",
    )
    _common(p)
    p.add_argument("--since", help="use commits after this tag/branch/commit instead of the base branch")
    p.add_argument("--title", default="Unreleased", help="section heading (default: %(default)s)")
    p.add_argument("--file", default="CHANGELOG.md", help="file to update with --write (default: %(default)s)")
    p.add_argument("--write", action="store_true",
                   help="after preview and confirmation, insert the section into --file (requires a terminal)")
    return p


def _needs_terminal(ui: TextIO, flag: str) -> int:
    render.error(ui, f"{flag} needs an interactive terminal to ask for your approval",
                 "Run it in a terminal. Without it, the command is a preview and changes nothing.")
    return EXIT_USAGE


def _terminal_prompt(ui: TextIO) -> Prompt:
    def prompt(question: str) -> str:
        ui.write(question)
        ui.flush()
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line

    return prompt


def _confirm(prompt: Prompt, question: str, word: str) -> bool:
    try:
        return prompt(question).strip().lower() == word
    except EOFError:
        return False


def _branch_error(ui: TextIO, exc: BranchError, operation: str) -> int:
    render.error(ui, str(exc), exc.hint)
    telemetry.failure(operation, exc.category, "warning" if exc.code != 1 else "error")
    return exc.code


def _pr_header(ui: TextIO, info: BranchInfo, publish: bool) -> None:
    if publish:
        ui.write("Lazy-Commit pull request (publish mode - nothing is pushed or created without your confirmation)\n")
    else:
        ui.write("Lazy-Commit pull request (preview only - nothing will be pushed or published)\n")
    ui.write(RULE + "\n")
    ui.write(f"Repository: {info.root}\n")
    ui.write(f"Branch:     {info.branch} ({info.head[:12]}) -> base {info.base} (compared with {info.base_ref})\n")
    where = f"GitHub: {info.repo_slug}" if info.github else "not a GitHub URL; publishing is unavailable"
    ui.write(f"Remote:     {info.remote} ({where})\n")
    merges = sum(c.is_merge for c in info.commits)
    ui.write(f"Commits:    {len(info.commits)} on this branch not in {info.base}"
             + (f" ({merges} merge commit(s))" if merges else "") + "\n")
    ui.write(f"Changes:    {info.shortstat or f'{len(info.files)} file(s) changed'}\n")
    if info.state.files:
        ui.write(f"Note:       {len(info.state.files)} uncommitted change(s) are not part of this pull request.\n")


def _input_report(ui: TextIO, pr: PRInput) -> None:
    render.skipped(ui, "Excluded from the model by safety checks", pr.excluded)
    render.skipped(ui, "Not sent to the model", pr.manual)
    if pr.withheld:
        ui.write(f"\n{len(pr.withheld)} commit message(s) matched a secret pattern and were withheld.\n")


def _outgoing_report(ui: TextIO, scan: branch.OutgoingScan, blocking: bool) -> None:
    if scan.problems:
        ui.write(f"\nPossible secrets in the {scan.commits} commit(s) that are not on the remote yet:\n")
        for p in scan.problems[:20]:
            ui.write(f"  - {_clean(p)}\n")
        if len(scan.problems) > 20:
            ui.write(f"  - ... and {len(scan.problems) - 20} more\n")
        ui.write("  Pushing would publish them in the branch history"
                 + (". Publishing is blocked until they are removed.\n" if blocking
                    else "; `--publish` will refuse until they are removed.\n"))
    for u in scan.unscanned[:10]:
        ui.write(f"  warning: could not secret-scan {_clean(u)}\n")


def _draft(args: argparse.Namespace, ui: TextIO, info: BranchInfo, pr: PRInput) -> tuple[prwriter.Draft | None, str]:
    """Ask the local model. Returns (draft, "") or (None, error message)."""
    if args.no_model:
        return prwriter.fallback_draft(info, pr, "--no-model was given"), ""
    if not pr.files:
        return None, "no file content could be sent to the model safely"
    user_prompt = prwriter.build_prompt(info, pr)
    try:
        assert_safe(prwriter.SYSTEM_PROMPT, user_prompt)
        with telemetry.span("model.request"):
            client = OllamaClient(args.ollama_host, timeout=args.timeout)
            if not args.model:
                raise ModelError("no model selected", "Pass --model NAME or set LAZYCOMMIT_MODEL.",
                                 "model_not_selected")
            client.ensure_model(args.model)
            ui.write(f"\nAsking local model {args.model!r} to draft the description...\n")
            ui.flush()
            raw = client.chat_json(args.model, prwriter.SYSTEM_PROMPT, user_prompt, prwriter.DESCRIPTION_SCHEMA)
        return prwriter.parse_description(raw, info, pr), ""
    except ModelError as exc:
        telemetry.failure("model.request", exc.category)
        return None, str(exc) + (f" ({exc.hint})" if exc.hint else "")
    except UnsafeInputError as exc:
        telemetry.failure("model.request", "unsafe_model_input")
        return None, str(exc)
    except prwriter.InvalidDescriptionError as exc:
        telemetry.failure("model.request", "invalid_model_output")
        return None, f"rejected model output: {exc}"


def _show(ui: TextIO, title: str, body: str, draft: prwriter.Draft, model: str | None) -> None:
    ui.write("\n" + RULE + "\nPULL REQUEST TITLE\n" + RULE + "\n" + _clean(title) + "\n")
    ui.write(RULE + "\nPULL REQUEST DESCRIPTION\n" + RULE + "\n")
    ui.write(body.rstrip("\n") + "\n" + RULE + "\n")
    if draft.source == "model":
        ui.write(f"Drafted by local model {model!r}; checked against the branch's commits and files.\n")
    for w in draft.warnings:
        ui.write(f"  note: {_clean(w)}\n")


def run(command: str, argv: list[str], out: TextIO, prompt: Prompt | None, err: TextIO | None) -> int:
    try:
        if command == "pr":
            args = build_pr_parser().parse_args(argv)
            with telemetry.span("pr.publish" if args.publish else "pr.prepare"):
                return _pr(args, out, prompt, err)
        args = build_changelog_parser().parse_args(argv)
        with telemetry.span("changelog"):
            return _changelog(args, out, prompt)
    except KeyboardInterrupt:
        out.write("\nInterrupted. Check `git status` and GitHub to see what, if anything, was completed.\n")
        return EXIT_INTERRUPTED


# --- pr -------------------------------------------------------------------------


def _pr(args: argparse.Namespace, out: TextIO, prompt: Prompt | None, err: TextIO | None) -> int:
    ui = (err or sys.stderr) if args.json else out
    if args.publish and prompt is None:
        if not sys.stdin.isatty():
            return _needs_terminal(ui, "--publish")
        prompt = _terminal_prompt(ui)
    try:
        info = branch.resolve(args.repo.resolve(), base=args.base, remote=args.remote)
    except BranchError as exc:
        if args.json:
            out.write(json.dumps({"lazycommit_version": __version__, "status": "error",
                                  "message": _clean(str(exc)), "published": False}) + "\n")
        return _branch_error(ui, exc, "pr.publish" if args.publish else "pr.prepare")

    _pr_header(ui, info, args.publish)
    pr = branch.collect_pr(info, Limits(args.max_file_chars, args.max_total_chars, args.max_files))
    _input_report(ui, pr)

    if args.show_input:
        ui.write("\n----- BEGIN SANITIZED MODEL INPUT (not sent) -----\n")
        ui.write(prwriter.build_prompt(info, pr) + "\n")
        ui.write("----- END SANITIZED MODEL INPUT -----\n")
        return EXIT_OK
    if args.publish:
        assert prompt is not None
        return _publish(args, ui, prompt, info, pr)

    scan = branch.scan_outgoing(info)
    _outgoing_report(ui, scan, blocking=False)
    draft, error = _draft(args, ui, info, pr)
    if draft is None:
        render.error(ui, error)
        telemetry.failure("pr.prepare", "fallback_description", "warning")
        ui.write("Showing a description built only from commit messages and Git data instead.\n")
        draft = prwriter.fallback_draft(info, pr, error)
    body = prwriter.render_body(draft, info, pr)
    _show(ui, draft.title, body, draft, args.model)
    if args.json:
        out.write(json.dumps(_pr_json(info, pr, draft, body, scan, args.model), indent=2) + "\n")
    ui.write("\nThis is a preview. Nothing was pushed and no pull request was created.\n"
             "To review, edit, and publish it: lazycommit pr --publish\n")
    return EXIT_OK


def _pr_json(info: BranchInfo, pr: PRInput, draft: prwriter.Draft, body: str,
             scan: branch.OutgoingScan, model: str | None) -> dict[str, Any]:
    return {
        "lazycommit_version": __version__,
        "status": "preview",
        "published": False,
        "pushed": False,
        "branch": info.branch,
        "base": info.base,
        "repository": info.repo_slug,
        "title": _clean(draft.title),
        "body": body,
        "source": draft.source,
        "model": model if draft.source == "model" else None,
        "commits": [{"oid": c.oid, "subject": "(withheld)" if c.oid in pr.withheld else _clean(c.subject)}
                    for c in info.commits],
        "files": [f.path for f in info.files],
        "excluded": [{"path": _clean(s.path), "reason": _clean(s.reason)} for s in pr.excluded],
        "outgoing_secret_problems": [_clean(p) for p in scan.problems],
        "warnings": [_clean(w) for w in draft.warnings],
    }


def _edit(ui: TextIO, prompt: Prompt, title: str, body: str, draft: prwriter.Draft,
          model: str | None) -> tuple[str, str] | None:
    """Let the user edit the title/body. Returns None if they quit."""
    while True:
        choice = _ask(prompt, "\n[c]ontinue, edit [t]itle, edit [b]ody, or [q]uit: ",
                      {"c": "c", "continue": "c", "t": "t", "title": "t", "b": "b", "body": "b",
                       "q": "q", "quit": "q"})
        if choice in (None, "q"):
            return None
        if choice == "c":
            return title, body
        if choice == "t":
            try:
                new = " ".join(prompt("New title (empty keeps the current one): ").split())
            except EOFError:
                return None
            title = new[:prwriter.MAX_TITLE_CHARS] or title
        else:
            fd, tmp = tempfile.mkstemp(prefix="lazycommit-pr-", suffix=".md")
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(body)
                try:
                    answer = prompt(f"Edit the description in {tmp}, save it, then press Enter "
                                    "(type q to discard your edits): ")
                except EOFError:
                    return None
                if answer.strip().lower() != "q":
                    edited = Path(tmp).read_text(encoding="utf-8", errors="replace")
                    body = edited if edited.strip() else body
            finally:
                os.unlink(tmp)
        _show(ui, title, body, draft, model)


def _unchanged(info: BranchInfo) -> bool:
    try:
        now = read_state(info.root)
    except GitError:
        return False
    return now.branch == info.branch and now.head_oid == info.head


def _push(ui: TextIO, prompt: Prompt, info: BranchInfo, state: str, purpose: str) -> bool | None:
    """Ask for and perform the push. True pushed, False declined, None failed/stale."""
    assert info.remote
    if state == branch.NOT_PUSHED:
        ui.write(f"\nBranch {info.branch!r} is not on {info.remote} yet. {purpose}\n")
    else:
        ui.write(f"\n{info.remote} has an older version of {info.branch!r}. {purpose}\n")
    ui.write(f"Lazy-Commit will run (never with --force):\n"
             f"  git push --set-upstream {info.remote} refs/heads/{info.branch}:refs/heads/{info.branch}\n")
    if not _confirm(prompt, "Type 'push' to push it now, or anything else to stop: ", "push"):
        return False
    if not _unchanged(info):
        render.error(ui, "the branch changed since the preview was generated; nothing was pushed",
                     "Run `lazycommit pr --publish` again.")
        telemetry.failure("pr.push", "repository_changed", "warning")
        return None
    try:
        with telemetry.span("pr.push"):
            push_branch(info.root, info.remote, info.branch)
            after, oid = branch.remote_branch_state(info)
    except (GitHubError, GitError) as exc:
        render.error(ui, str(exc), getattr(exc, "hint", ""))
        telemetry.failure("pr.push", getattr(exc, "category", "git_error"))
        return None
    if after != branch.UP_TO_DATE:
        render.error(ui, f"git push finished but {info.remote} reports {info.branch!r} at "
                         f"{(oid or 'nothing')[:12]}, not {info.head[:12]}")
        telemetry.failure("pr.push", "push_failed")
        return None
    ui.write(f"Pushed {info.branch} ({info.head[:12]}) to {info.remote}.\n")
    return True


def _existing(ui: TextIO, prompt: Prompt, info: BranchInfo, prs: list[ExistingPR], state: str) -> int:
    ui.write(f"\nAn open pull request already exists for {info.branch!r}:\n")
    for pr in prs:
        ui.write(f"  #{pr.number} {_clean(pr.title)}{' (draft)' if pr.draft else ''} -> {pr.base}\n  {pr.url}\n")
    ui.write("\nLazy-Commit will not create a second pull request or change this one. Your options:\n"
             f"  - Open it: {prs[0].url}\n"
             f"  - Change its title or description on GitHub or with `gh pr edit {prs[0].number}`.\n")
    if state in (branch.NOT_PUSHED, branch.FAST_FORWARD):
        ui.write("  - Push your new local commits so the pull request includes them.\n")
        pushed = _push(ui, prompt, info, state, f"Pushing adds your new commits to #{prs[0].number}.")
        if pushed:
            ui.write(f"Pull request #{prs[0].number} now includes your latest commits: {prs[0].url}\n")
        elif pushed is False:
            ui.write("Not pushed. Nothing was changed.\n")
        return EXIT_OK if pushed is not None else EXIT_ERROR
    ui.write("  - Its branch is already up to date with your local commits.\n")
    return EXIT_OK


def _publish(args: argparse.Namespace, ui: TextIO, prompt: Prompt, info: BranchInfo, pr: PRInput) -> int:
    stop = "Nothing was pushed and no pull request was created."
    if info.github is None or info.remote is None:
        render.error(ui, f"remote {info.remote!r} does not point to a github.com repository",
                     "Publishing supports GitHub only; use --remote to pick a GitHub remote.")
        telemetry.failure("pr.publish", "not_github")
        return EXIT_ERROR
    gh_path = find_gh()
    if gh_path is None:
        render.error(ui, "the GitHub CLI (gh) is not installed or not on PATH", INSTALL_HINT)
        telemetry.failure("pr.publish", "gh_missing")
        ui.write(stop + "\n")
        return EXIT_ERROR
    gh = GitHub(gh_path, info.github[0], info.github[1], info.root)
    try:
        gh.check_auth()
        ui.write(f"GitHub:     gh is logged in; target repository {gh.slug}\n")
        scan = branch.scan_outgoing(info)
        _outgoing_report(ui, scan, blocking=True)
        if scan.problems:
            render.error(ui, "possible secrets would be pushed",
                         "Remove them from the branch history yourself (Lazy-Commit never rewrites history), "
                         "then run this again.")
            telemetry.failure("pr.publish", "outgoing_secrets")
            ui.write(stop + "\n")
            return EXIT_ERROR
        state, _ = branch.remote_branch_state(info)
        ui.write(f"Remote branch: {state}\n")
        if state == branch.DIVERGED:
            render.error(ui, f"{info.remote}/{info.branch} has commits that your local branch does not have",
                         "Lazy-Commit never force-pushes. Integrate them first (e.g. `git pull --rebase`), "
                         "then run this again.")
            telemetry.failure("pr.publish", "remote_diverged")
            ui.write(stop + "\n")
            return EXIT_ERROR
        existing = gh.open_prs(info.branch)
    except (GitHubError, GitError) as exc:
        render.error(ui, str(exc), getattr(exc, "hint", ""))
        telemetry.failure("pr.publish", getattr(exc, "category", "git_error"))
        ui.write(stop + "\n")
        return EXIT_ERROR
    if existing:
        return _existing(ui, prompt, info, existing, state)

    draft, error = _draft(args, ui, info, pr)
    if draft is None:
        render.error(ui, error)
        telemetry.failure("pr.publish", "fallback_description", "warning")
        if not _confirm(prompt, "Use a description built only from commit messages and Git data instead? "
                                "Type 'y' to use it: ", "y"):
            ui.write(stop + "\n")
            return EXIT_ERROR
        draft = prwriter.fallback_draft(info, pr, error)
    body = prwriter.render_body(draft, info, pr)
    _show(ui, draft.title, body, draft, args.model)
    edited = _edit(ui, prompt, draft.title, body, draft, args.model)
    if edited is None:
        ui.write("\nStopped. " + stop + "\n")
        return EXIT_OK
    title, body = edited

    rules = sorted({f.rule for f in scan_text(title + "\n" + body)})
    if rules:
        render.error(ui, "the title or description matches a secret pattern: " + ", ".join(rules),
                     "Edit it out and run this again.")
        telemetry.failure("pr.publish", "outgoing_secrets")
        ui.write(stop + "\n")
        return EXIT_ERROR

    pushed_now = False
    if state != branch.UP_TO_DATE:
        pushed = _push(ui, prompt, info, state, "A pull request needs the branch on GitHub.")
        if pushed is None:
            ui.write("No pull request was created.\n")
            return EXIT_STALE if not _unchanged(info) else EXIT_ERROR
        if not pushed:
            ui.write("Not pushed. No pull request was created.\n")
            return EXIT_OK
        pushed_now = True

    kind = "draft pull request" if args.draft else "pull request"
    ui.write(f"\nReady to create a {kind} on {gh.slug}: {info.branch} -> {info.base}\n"
             f"  title: {_clean(title)}\n")
    if not _confirm(prompt, "Type 'create' to create it, or anything else to stop: ", "create"):
        ui.write("Not created." + (" The branch was pushed; nothing else changed.\n" if pushed_now
                                   else " Nothing was changed.\n"))
        return EXIT_OK
    pushed_note = f" (the branch was pushed to {info.remote})" if pushed_now else ""
    if not _unchanged(info):
        render.error(ui, f"the branch changed since the preview; no pull request was created{pushed_note}",
                     "Run `lazycommit pr --publish` again.")
        telemetry.failure("pr.create", "repository_changed", "warning")
        return EXIT_STALE
    try:
        again = gh.open_prs(info.branch)
        if again:
            return _existing(ui, prompt, info, again, branch.UP_TO_DATE)
        with telemetry.span("pr.create"):
            url = gh.create_pr(info.base, info.branch, title, body, args.draft)
    except GitHubError as exc:
        render.error(ui, f"{exc}{pushed_note}", exc.hint)
        telemetry.failure("pr.create", exc.category)
        return EXIT_ERROR
    ui.write(f"\nCreated {kind}: {url}\n")
    return EXIT_OK


# --- changelog --------------------------------------------------------------------


def _changelog(args: argparse.Namespace, out: TextIO, prompt: Prompt | None) -> int:
    if args.write and prompt is None:
        if not sys.stdin.isatty():
            return _needs_terminal(out, "--write")
        prompt = _terminal_prompt(out)
    try:
        info = branch.resolve(args.repo.resolve(), base=args.base, remote=args.remote,
                              since=args.since, for_pr=False)
    except BranchError as exc:
        return _branch_error(out, exc, "changelog")

    log = changelog.build(info.commits)
    section = changelog.render(log, args.title, datetime.date.today().isoformat())
    out.write("Lazy-Commit changelog (" + ("write mode - the file changes only after your confirmation"
                                           if args.write else "preview only - no files will be changed") + ")\n")
    out.write(RULE + "\n")
    out.write(f"Range:      {info.base}..{info.branch} ({len(info.commits)} commit(s)")
    if log.merges_skipped:
        out.write(f", {log.merges_skipped} merge commit(s) skipped")
    out.write(")\n")
    confirmed = sum(1 for e in log.entries if e.category)
    out.write(f"Entries:    {confirmed} confirmed by Conventional Commit type, "
              f"{len(log.entries) - confirmed} unclassified\n")
    if log.withheld:
        out.write(f"            {log.withheld} commit message(s) withheld: they matched a secret pattern\n")
    out.write("\n----- BEGIN CHANGELOG SECTION -----\n" + section + "----- END CHANGELOG SECTION -----\n")
    out.write("Built only from commit messages. It lists changes, not test results.\n")
    if not args.write:
        out.write(f"\nPreview only. To add it to {args.file}: lazycommit changelog --write\n")
        return EXIT_OK

    assert prompt is not None
    try:
        path = changelog.target_path(info.root, args.file)
        expected = changelog.fingerprint(path)
    except (changelog.ChangelogWriteError, OSError) as exc:
        render.error(out, str(exc))
        telemetry.failure("changelog.write", "invalid_path")
        return EXIT_ERROR
    action = "add this section to" if path.exists() else "create"
    if not _confirm(prompt, f"Type 'write' to {action} {args.file}, or anything else to cancel: ", "write"):
        out.write("Cancelled. Nothing was written.\n")
        return EXIT_OK
    try:
        with telemetry.span("changelog.write"):
            changelog.write_section(path, section, expected)
    except changelog.ChangelogWriteError as exc:
        render.error(out, str(exc), "Run the command again to preview the current state.")
        conflict = "changed since" in str(exc)
        telemetry.failure("changelog.write", "changelog_conflict" if conflict else "changelog_write_failed")
        return EXIT_STALE if conflict else EXIT_ERROR
    except OSError as exc:
        render.error(out, f"could not write {args.file}: {exc.strerror or type(exc).__name__}")
        telemetry.failure("changelog.write", "changelog_write_failed")
        return EXIT_ERROR
    out.write(f"Updated {args.file}. It was not staged or committed; review it with `git diff`.\n")
    return EXIT_OK
