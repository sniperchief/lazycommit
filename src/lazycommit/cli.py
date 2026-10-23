"""Command-line entry point.

The default run is a read-only preview. ``--json`` is the same preview in a
machine-readable form for scripts; it never commits. ``--review`` is the only
mode that can create commits, and only after interactive approval.
``lazycommit pr`` and ``lazycommit changelog`` live in ``prcli.py``.

Exit codes: 0 success (proposal shown, nothing to do, or approved commits
created), 1 error or failed commit, 2 usage error, 3 repository in an unsafe
state (conflicts / operation in progress), 4 repository changed after the
plan was generated, 130 interrupted.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from lazycommit import __version__, prcli, render, review, telemetry
from lazycommit.collect import CollectResult, Limits, collect
from lazycommit.execute import Snapshot, blocked_reasons, take_snapshot
from lazycommit.gitrepo import GitError, RepoState, read_state
from lazycommit.ollama import DEFAULT_HOST, DEFAULT_TIMEOUT_SECONDS, ModelError, OllamaClient
from lazycommit.planner import (
    PLAN_SCHEMA,
    SYSTEM_PROMPT,
    InvalidPlanError,
    Plan,
    UnsafeInputError,
    assert_safe,
    build_user_prompt,
    parse_plan,
)

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_UNSAFE, EXIT_STALE, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 130

_REVIEW_EXIT = {
    review.OK: EXIT_OK,
    review.FAILED_OUTCOME: EXIT_ERROR,
    review.STALE: EXIT_STALE,
    review.INTERRUPTED: EXIT_INTERRUPTED,
}


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lazycommit",
        description="Propose logical Conventional Commits for the current Git changes "
        "using a local Ollama model. By default this is a read-only preview; use "
        "--review to approve individual commits interactively.",
        epilog="Other commands: `lazycommit pr --help` (pull request preview/publish) and "
        "`lazycommit changelog --help` (changelog preview/write).",
    )
    p.add_argument("--repo", type=Path, default=Path.cwd(), help="path inside the repository (default: cwd)")
    p.add_argument("--model", default=os.environ.get("LAZYCOMMIT_MODEL"),
                   help="Ollama model name (default: $LAZYCOMMIT_MODEL)")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
                   help=f"local Ollama URL (default: $OLLAMA_HOST or {DEFAULT_HOST})")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                   help="seconds to wait for the model (default: %(default)s)")
    p.add_argument("--max-file-chars", type=_positive_int, default=Limits.max_file_chars,
                   help="max diff characters sent per file (default: %(default)s)")
    p.add_argument("--max-total-chars", type=_positive_int, default=Limits.max_total_chars,
                   help="max diff characters sent in total (default: %(default)s)")
    p.add_argument("--max-files", type=_positive_int, default=Limits.max_files,
                   help="max files sent to the model (default: %(default)s)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--show-input", action="store_true",
                      help="print the exact sanitized model input and exit without calling the model")
    mode.add_argument("--json", action="store_true",
                      help="print the proposal as JSON on stdout (progress goes to stderr); never commits")
    mode.add_argument("--review", action="store_true",
                      help="review each proposed commit interactively and create the approved ones "
                      "(requires a terminal)")
    p.add_argument("--version", action="version", version=f"lazycommit {__version__}")
    return p


class _Stop(Exception):
    """Planning ended early; the message has already been shown."""

    def __init__(self, code: int, status: str, message: str) -> None:
        super().__init__(message)
        self.code, self.status, self.message = code, status, message


@dataclass
class _Planned:
    state: RepoState
    collected: CollectResult
    plan: Plan
    model: str
    snapshot: Snapshot | None


def _plan(args: argparse.Namespace, ui: TextIO) -> _Planned:
    try:
        state = read_state(args.repo.resolve())
    except GitError as exc:
        render.error(ui, str(exc), "Run Lazy-Commit from inside a Git working tree, or pass --repo PATH.")
        telemetry.failure("plan", "not_a_repository")
        raise _Stop(EXIT_ERROR, "error", str(exc)) from None

    render.header(ui, state, review=args.review)

    if state.unsafe_reasons:
        render.unsafe_state(ui, state)
        telemetry.failure("plan", "unsafe_repository_state", "warning")
        raise _Stop(EXIT_UNSAFE, "unsafe", "; ".join(state.unsafe_reasons))

    if not state.files:
        ui.write("\nWorking tree is clean - nothing to propose.\n")
        raise _Stop(EXIT_OK, "clean", "working tree is clean")

    # Review mode records the exact state the plan is based on, so execution
    # can refuse to run if anything changes afterwards.
    snapshot = take_snapshot(state) if args.review else None

    limits = Limits(args.max_file_chars, args.max_total_chars, args.max_files)
    collected = collect(state, limits)
    render.collection(ui, collected)

    if not collected.candidates:
        render.skipped(ui, "Needs manual review", collected.manual)
        ui.write("\nNo changes are eligible for AI analysis. Nothing was sent to the model.\n")
        raise _Stop(EXIT_OK, "nothing_eligible", "no changes are eligible for AI analysis")

    if snapshot is not None:
        try:
            unchanged = take_snapshot(read_state(state.root)) == snapshot
        except GitError:
            unchanged = False
        if not unchanged:
            render.error(ui, "the repository changed while it was being analyzed",
                         "Wait for other tools to finish, then run Lazy-Commit again.")
            telemetry.failure("plan", "repository_changed", "warning")
            raise _Stop(EXIT_STALE, "changed", "repository changed during analysis")

    user_prompt = build_user_prompt(state, collected)
    try:
        assert_safe(SYSTEM_PROMPT, user_prompt)
    except UnsafeInputError as exc:
        render.error(ui, str(exc), "Please report this; review the excluded files manually.")
        telemetry.failure("plan", "unsafe_model_input")
        raise _Stop(EXIT_ERROR, "error", str(exc)) from None

    if args.show_input:
        ui.write("\n----- BEGIN SANITIZED MODEL INPUT (not sent) -----\n")
        ui.write(user_prompt + "\n")
        ui.write("----- END SANITIZED MODEL INPUT -----\n")
        raise _Stop(EXIT_OK, "input_shown", "model input shown")

    telemetry.count("files", len(collected.candidates))
    telemetry.count("excluded_files", len(collected.excluded))
    try:
        with telemetry.span("model.request"):
            client = OllamaClient(args.ollama_host, timeout=args.timeout)
            if not args.model:
                installed = client.installed_models()
                raise ModelError(
                    "no model selected",
                    "Pass --model NAME or set LAZYCOMMIT_MODEL. Installed models: "
                    + (", ".join(installed) or "none (run `ollama pull qwen2.5-coder:7b`, for example)"),
                    "model_not_selected",
                )
            client.ensure_model(args.model)
            ui.write(f"\nAsking local model {args.model!r} for a proposal...\n")
            ui.flush()
            raw = client.chat_json(args.model, SYSTEM_PROMPT, user_prompt, PLAN_SCHEMA)
        with telemetry.span("plan.validate"):
            plan = parse_plan(raw, collected.candidates)
            telemetry.count("groups", len(plan.groups))
            if plan.warnings:
                telemetry.count("rejected_groups", len(plan.warnings))
                telemetry.failure("plan.validate", "rejected_groups", "warning")
    except ModelError as exc:
        telemetry.failure("model.request", exc.category)
        render.error(ui, str(exc), exc.hint)
        ui.write("No proposal was produced. Nothing was changed.\n")
        raise _Stop(EXIT_ERROR, "error", str(exc)) from None
    except InvalidPlanError as exc:
        telemetry.failure("plan.validate", "invalid_model_output")
        render.error(ui, f"rejected model output: {exc}",
                     "Re-run, or try a different/larger model with --model.")
        ui.write("No proposal was produced. Nothing was changed.\n")
        raise _Stop(EXIT_ERROR, "error", f"rejected model output: {exc}") from None

    return _Planned(state, collected, plan, args.model, snapshot)


def _terminal_prompt(ui: TextIO) -> review.Prompt:
    def prompt(question: str) -> str:
        ui.write(question)
        ui.flush()
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line

    return prompt


def _write_json(out: TextIO, payload: dict[str, Any]) -> None:
    out.write(json.dumps(payload, indent=2) + "\n")


def run(
    argv: list[str] | None = None,
    out: TextIO | None = None,
    prompt: review.Prompt | None = None,
    err: TextIO | None = None,
) -> int:
    """Run the CLI. ``prompt`` answers review questions (tests inject one)."""
    out = out or sys.stdout
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in prcli.COMMANDS:
        return prcli.run(argv[0], argv[1:], out, prompt, err)
    args = build_parser().parse_args(argv)
    with telemetry.span("plan"):
        return _run_plan(args, out, prompt, err)


def _run_plan(args: argparse.Namespace, out: TextIO, prompt: review.Prompt | None, err: TextIO | None) -> int:
    ui = (err or sys.stderr) if args.json else out

    if args.review and prompt is None:
        if not sys.stdin.isatty():
            render.error(
                ui,
                "--review needs an interactive terminal to ask for your approval",
                "Run it in a terminal. For scripts, use the default preview or --json; they never commit.",
            )
            return EXIT_USAGE
        prompt = _terminal_prompt(ui)

    try:
        planned = _plan(args, ui)
    except _Stop as stop:
        if args.json:
            _write_json(out, {"lazycommit_version": __version__, "status": stop.status,
                              "message": render._clean(stop.message), "committed": False})
        return stop.code

    state, collected, plan = planned.state, planned.collected, planned.plan
    if args.json:
        blocked = [blocked_reasons(g, state, collected) for g in plan.groups]
        _record_blocked(sum(1 for b in blocked if b))
        render.plan(ui, plan, collected, planned.model)
        _write_json(out, render.plan_dict(state, plan, collected, planned.model, blocked))
        return EXIT_OK
    if args.review:
        assert prompt is not None and planned.snapshot is not None
        with telemetry.span("commit.review"):
            outcome = review.run_review(out, prompt, state, collected, plan, planned.snapshot, planned.model)
        return _REVIEW_EXIT[outcome]
    render.plan(out, plan, collected, planned.model)
    return EXIT_OK


def _record_blocked(blocked: int) -> None:
    if blocked:
        telemetry.count("blocked_groups", blocked)
        telemetry.failure("plan.validate", "blocked_groups", "warning")


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")
    telemetry.init()
    try:
        return run(argv)
    except KeyboardInterrupt:
        sys.stdout.write("\nInterrupted. Nothing was committed.\n")
        return EXIT_INTERRUPTED
    finally:
        telemetry.shutdown()
