"""Run the benchmark against a local Ollama model and write a JSON report.

The report holds case names, metrics, failure categories, and the model's
short commit messages for the synthetic fixtures. It never holds diffs, file
contents, prompts, secret values, or local paths; it is checked for those
before it is written.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import stat
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from lazycommit import __version__, branch, prwriter, secrets
from lazycommit.benchmark import BENCHMARK_NAME, BENCHMARK_VERSION, scoring
from lazycommit.benchmark.cases import CASES, FAKE_SECRETS, Fixture, PlanCase, PRCase
from lazycommit.collect import Limits, collect
from lazycommit.execute import blocked_reasons
from lazycommit.gitrepo import GitError, read_state
from lazycommit.ollama import DEFAULT_HOST, DEFAULT_TIMEOUT_SECONDS, ModelError, OllamaClient
from lazycommit.planner import PLAN_SCHEMA, SYSTEM_PROMPT, InvalidPlanError, assert_safe, build_user_prompt, parse_plan

ModelCall = Callable[[str, str, dict[str, Any]], str]


@contextmanager
def _isolated_git() -> Iterator[None]:
    """Make Lazy-Commit's own Git reads ignore the user's config during the run."""
    saved = {k: os.environ.get(k) for k in ("GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL")}
    os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    os.environ["GIT_CONFIG_GLOBAL"] = os.devnull
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _rmtree(path: Path) -> None:
    def onerror(func: Callable[..., Any], target: str, _exc: Any) -> None:
        os.chmod(target, stat.S_IWRITE)  # Git marks object files read-only on Windows
        func(target)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=onerror)
    else:
        shutil.rmtree(path, onerror=onerror)


def _call(model_call: ModelCall | None, system: str, user: str, schema: dict[str, Any],
          result: dict[str, Any]) -> str | None:
    """Invoke the model, recording latency and a safe failure category."""
    if model_call is None:
        result["outcome"] = "skipped"
        return None
    start = time.monotonic()
    try:
        raw = model_call(system, user, schema)
    except ModelError as exc:
        result["outcome"] = "model_error"
        result["error_category"] = exc.category
        return None
    finally:
        result["latency_s"] = round(time.monotonic() - start, 2)
    return raw


def _safe_text(text: str, limit: int = 160) -> str:
    return secrets.redact(" ".join(text.split()))[:limit]


def run_plan_case(case: PlanCase, workdir: Path, model_call: ModelCall | None, limits: Limits) -> dict[str, Any]:
    fixture = Fixture(workdir / "repo")
    case.setup(fixture)
    state = read_state(fixture.root)
    collected = collect(state, limits)
    candidates = {c.path for c in collected.candidates}
    user_prompt = build_user_prompt(state, collected)
    assert_safe(SYSTEM_PROMPT, user_prompt)

    result: dict[str, Any] = {"name": case.name, "kind": "plan", "description": case.description}
    pre_safety = scoring.plan_safety([], [], candidates, set(case.excluded), set(case.manual),
                                     user_prompt, FAKE_SECRETS)
    result["safety"] = asdict(pre_safety)
    result["excluded_detected"] = sorted(s.path for s in collected.excluded)

    raw = _call(model_call, SYSTEM_PROMPT, user_prompt, PLAN_SCHEMA, result)
    if raw is None:
        return result
    try:
        plan = parse_plan(raw, collected.candidates)
    except InvalidPlanError:
        result["outcome"] = "invalid_output"
        return result

    blocked = [blocked_reasons(g, state, collected) for g in plan.groups]
    safety = scoring.plan_safety(plan.groups, blocked, candidates, set(case.excluded), set(case.manual),
                                 user_prompt, FAKE_SECRETS, case.forbidden)
    handled = ({m.path for m in plan.manual} | {s.path for s in collected.manual + collected.excluded}
               | {f for g, reasons in zip(plan.groups, blocked) if reasons for f in g.files})
    scorable = candidates - set(case.manual)
    expected = [g & scorable for g in case.groups]
    pair = scoring.pairwise(expected, [g.files for g in plan.groups], scorable)
    grounded, issues = scoring.groundedness(plan.groups, user_prompt, candidates, case.forbidden)
    messages_ok = [not scoring.message_problems(g) for g in plan.groups]

    result.update({
        "outcome": "ok",
        "safety": asdict(safety),
        "metrics": {
            "grouping_precision": pair.precision,
            "grouping_recall": pair.recall,
            "grouping_f1": pair.f1,
            "type_accuracy": scoring.type_accuracy(expected, case.types, plan.groups),
            "message_quality": round(sum(messages_ok) / len(messages_ok), 4) if messages_ok else 1.0,
            "groundedness": grounded,
            "manual_review_recall": scoring.manual_recall(set(case.manual), handled),
            "validator_rejections": len(plan.warnings),
        },
        "groundedness_issues": [_safe_text(i, 80) for i in issues[:10]],
        "message_problems": sorted({p for g in plan.groups for p in scoring.message_problems(g)}),
        "predicted": [{"header": _safe_text(g.header, 100), "files": g.files, "confidence": g.confidence,
                       "rationale": _safe_text(g.rationale), "blocked": bool(reasons),
                       "blocked_reasons": [_safe_text(r, 120) for r in reasons]}
                      for g, reasons in zip(plan.groups, blocked)],
    })
    return result


_PR_WARNING_KINDS = {
    "statement about testing": "test_claims",
    "naming file(s) not in this branch": "unknown_files",
    "file the model was not shown": "hidden_file_mentions",
    "removed the motivation": "unsupported_motivation",
    "removed the known limitations": "unsupported_limitations",
}


def run_pr_case(case: PRCase, workdir: Path, model_call: ModelCall | None, limits: Limits) -> dict[str, Any]:
    fixture = Fixture(workdir / "repo")
    case.setup(fixture)
    info = branch.resolve(fixture.root)
    pr = branch.collect_pr(info, limits)
    user_prompt = prwriter.build_prompt(info, pr)
    assert_safe(prwriter.SYSTEM_PROMPT, user_prompt)

    result: dict[str, Any] = {"name": case.name, "kind": "pr", "description": case.description}
    hidden_seen = sum(1 for h in case.hidden if f"FILE: {h} " in user_prompt)
    result["safety"] = {"secrets_in_model_input": sum(1 for v in FAKE_SECRETS if v in user_prompt),
                        "hidden_files_in_model_input": hidden_seen}
    raw = _call(model_call, prwriter.SYSTEM_PROMPT, user_prompt, prwriter.DESCRIPTION_SCHEMA, result)
    if raw is None:
        return result
    try:
        draft = prwriter.parse_description(raw, info, pr)
    except prwriter.InvalidDescriptionError:
        result["outcome"] = "invalid_output"
        return result

    raw_counts = dict.fromkeys(_PR_WARNING_KINDS.values(), 0)
    for w in draft.warnings:
        for needle, key in _PR_WARNING_KINDS.items():
            if needle in w:
                raw_counts[key] += 1
    body = prwriter.render_body(draft, info, pr)
    testing = body.split("## Testing", 1)[1].split("\n## ", 1)[0] if "## Testing" in body else ""
    outside_testing = body.replace(testing, "")
    known = {f.path for f in info.files if f.path in pr.paths}
    final_issues = scoring.unsupported_claims(f"{draft.title}\n{outside_testing}", user_prompt + body, known,
                                              case.forbidden)
    final_issues += [f"mentions withheld file {h}" for h in case.hidden if h in outside_testing.replace(
        "were not analyzed", "")]
    result["safety"]["secrets_in_description"] = sum(1 for v in FAKE_SECRETS if v in body or v in draft.title)
    result.update({
        "outcome": "ok",
        "metrics": {
            "raw_unsupported_statements": sum(raw_counts.values()),
            **{f"raw_{k}": v for k, v in raw_counts.items()},
            "final_description_clean": 1.0 if not final_issues else 0.0,
            "title_ok": 1.0 if 0 < len(draft.title) <= 72 else 0.0,
            "motivation_present": bool(draft.motivation),
            "motivation_expected": case.expect_motivation,
        },
        "final_issues": [_safe_text(i, 80) for i in final_issues[:10]],
        "title": _safe_text(draft.title, 100),
    })
    return result


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [c for c in cases if c.get("outcome") != "skipped"]
    ok = [c for c in calls if c.get("outcome") == "ok"]
    latencies = [c["latency_s"] for c in calls if "latency_s" in c]
    plan_ok = [c["metrics"] for c in ok if c["kind"] == "plan"]
    pr_ok = [c["metrics"] for c in ok if c["kind"] == "pr"]
    violations = sum(v for c in cases for k, v in c.get("safety", {}).items() if isinstance(v, int))

    def avg(rows: list[dict[str, Any]], key: str) -> float | None:
        return scoring.mean(float(r[key]) for r in rows if key in r)

    return {
        "cases": len(cases),
        "reliability": {
            "model_calls": len(calls),
            "valid_output_rate": round(len(ok) / len(calls), 4) if calls else None,
            "invalid_output_rate": round(sum(c["outcome"] == "invalid_output" for c in calls) / len(calls), 4)
            if calls else None,
            "timeout_rate": round(sum(c.get("error_category") == "model_timeout" for c in calls) / len(calls), 4)
            if calls else None,
            "error_rate": round(sum(c["outcome"] == "model_error" for c in calls) / len(calls), 4) if calls else None,
            "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
        },
        "plan": {key: avg(plan_ok, key) for key in (
            "grouping_precision", "grouping_recall", "grouping_f1", "type_accuracy", "message_quality",
            "groundedness", "manual_review_recall")},
        "pr": {key: avg(pr_ok, key) for key in ("raw_unsupported_statements", "final_description_clean", "title_ok")},
        "safety_violations": violations,
    }


def run_benchmark(model_call: ModelCall | None, model: str | None, names: Sequence[str] | None = None,
                  repeat: int = 1, limits: Limits | None = None, out: TextIO | None = None,
                  settings: dict[str, Any] | None = None) -> dict[str, Any]:
    out = out or sys.stdout
    limits = limits or Limits()
    selected = [c for c in CASES if not names or c.name in names]
    results: list[dict[str, Any]] = []
    workroot = Path(tempfile.mkdtemp(prefix="lazycommit-bench-"))
    try:
        with _isolated_git():
            for attempt in range(1, repeat + 1):
                for case in selected:
                    out.write(f"[{attempt}/{repeat}] {case.name} ... ")
                    out.flush()
                    workdir = workroot / f"{case.name}-{attempt}"
                    workdir.mkdir()
                    try:
                        if isinstance(case, PlanCase):
                            row = run_plan_case(case, workdir, model_call, limits)
                        else:
                            row = run_pr_case(case, workdir, model_call, limits)
                    except (GitError, branch.BranchError, OSError) as exc:
                        row = {"name": case.name, "kind": case.kind, "outcome": "fixture_error",
                               "error_category": getattr(exc, "category", type(exc).__name__)}
                    row["attempt"] = attempt
                    results.append(row)
                    out.write(row["outcome"] + (f" ({row.get('latency_s')}s)" if "latency_s" in row else "") + "\n")
                    _rmtree(workdir)
    finally:
        _rmtree(workroot)
    return {
        "benchmark": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "lazycommit_version": __version__,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "settings": {"temperature": 0, "repeat": repeat, "max_file_chars": limits.max_file_chars,
                     "max_total_chars": limits.max_total_chars, "max_files": limits.max_files,
                     **(settings or {})},
        "summary": _summary(results),
        "cases": results,
        "failures": [{"case": r["name"], "attempt": r["attempt"], "outcome": r["outcome"],
                      "category": r.get("error_category")}
                     for r in results if r.get("outcome") not in ("ok", "skipped")],
    }


class UnsafeReportError(Exception):
    pass


def check_report(report: dict[str, Any], forbidden: Sequence[str]) -> str:
    """Serialize the report, refusing if it contains secrets, paths, or diff text."""
    text = json.dumps(report, indent=2, ensure_ascii=False)
    for value in [*FAKE_SECRETS, *forbidden]:
        if value and (value in text or json.dumps(value)[1:-1] in text):
            raise UnsafeReportError("report contains a secret value or local path; not written")
    if secrets.contains_secret(text) or "diff --git" in text or "@@ -" in text:
        raise UnsafeReportError("report contains secret-like or diff content; not written")
    return text


def compare(report: dict[str, Any], baseline: dict[str, Any], out: TextIO) -> None:
    out.write(f"\nComparison with baseline ({baseline.get('model')}, {baseline.get('timestamp')}):\n")
    if baseline.get("benchmark_version") != report.get("benchmark_version"):
        out.write("  warning: different benchmark versions; numbers are not comparable.\n")
    for section in ("reliability", "plan", "pr"):
        now, then = report["summary"].get(section, {}), baseline.get("summary", {}).get(section, {})
        for key, value in now.items():
            old = then.get(key)
            if isinstance(value, (int, float)) and isinstance(old, (int, float)):
                out.write(f"  {section}.{key:<28} {old:>8} -> {value:<8} ({value - old:+.4f})\n")
    out.write("  Small differences can be run-to-run noise; repeat runs (--repeat) before claiming a change.\n")


def _print_summary(report: dict[str, Any], out: TextIO) -> None:
    s = report["summary"]
    out.write(f"\nModel: {report['model']}   cases: {s['cases']}   safety violations: {s['safety_violations']}\n")
    for section in ("reliability", "plan", "pr"):
        out.write(f"  {section}:\n")
        for key, value in s[section].items():
            out.write(f"    {key:<28} {value}\n")
    for f in report["failures"]:
        out.write(f"  failure: {f['case']} (attempt {f['attempt']}): {f['outcome']} {f['category'] or ''}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lazycommit-bench",
                                description="Evaluate the local model on deterministic fixture repositories. "
                                            "Never touches the repository you run it from.")
    p.add_argument("--model", default=os.environ.get("LAZYCOMMIT_MODEL"),
                   help="Ollama model name (default: $LAZYCOMMIT_MODEL)")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST))
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    p.add_argument("--cases", help="comma-separated case names (default: all)")
    p.add_argument("--repeat", type=int, default=1, help="run every case N times to measure variance")
    p.add_argument("--output", type=Path, help="report path (default: benchmark-results/<model>-<time>.json)")
    p.add_argument("--compare", type=Path, help="baseline report to compare against")
    p.add_argument("--no-model", action="store_true",
                   help="only run the deterministic safety checks (no Ollama needed)")
    p.add_argument("--list", action="store_true", help="list the cases and exit")
    return p


def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    if args.list:
        for c in CASES:
            out.write(f"{c.name:<26} {c.kind:<5} {c.description}\n")
        return 0
    names = [n.strip() for n in args.cases.split(",")] if args.cases else None
    unknown = set(names or []) - {c.name for c in CASES}
    if unknown or args.repeat < 1:
        out.write(f"error: unknown case(s): {', '.join(sorted(unknown))}\n" if unknown else "error: --repeat must be >= 1\n")
        return 2

    model_call: ModelCall | None = None
    if not args.no_model:
        try:
            client = OllamaClient(args.ollama_host, timeout=args.timeout)
            if not args.model:
                raise ModelError("no model selected", "Pass --model NAME or set LAZYCOMMIT_MODEL.")
            client.ensure_model(args.model)
        except ModelError as exc:
            out.write(f"error: {exc}\n" + (f"next step: {exc.hint}\n" if exc.hint else ""))
            out.write("Nothing was run. Use --no-model to run only the deterministic safety checks.\n")
            return 1
        model = args.model

        def model_call(system: str, user: str, schema: dict[str, Any]) -> str:
            return client.chat_json(model, system, user, schema)

    out.write(f"{BENCHMARK_NAME} v{BENCHMARK_VERSION} - model: {args.model if model_call else '(none: --no-model)'}\n")
    report = run_benchmark(model_call, args.model if model_call else None, names, args.repeat,
                           settings={"timeout_s": args.timeout}, out=out)
    _print_summary(report, out)
    if args.compare:
        try:
            compare(report, json.loads(args.compare.read_text(encoding="utf-8")), out)
        except (OSError, ValueError) as exc:
            out.write(f"warning: could not read baseline: {type(exc).__name__}\n")

    stamp = report["timestamp"].replace(":", "").replace("+0000", "Z")
    label = "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in (report["model"] or "no-model"))
    path = args.output or Path("benchmark-results") / f"{label}-{stamp}.json"
    try:
        text = check_report(report, [str(Path(tempfile.gettempdir())), str(Path.home())])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    except (UnsafeReportError, OSError) as exc:
        out.write(f"error: {exc}\n")
        return 1
    out.write(f"\nReport written to {path}\n")
    return 0 if report["summary"]["safety_violations"] == 0 else 3
