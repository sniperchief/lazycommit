"""Optional, privacy-preserving Sentry telemetry.

Off unless ``LAZYCOMMIT_SENTRY_DSN`` is set, and ``sentry_sdk`` is not even
imported otherwise. When on, Lazy-Commit only ever reports:

* spans named after a fixed set of workflow operations, with durations and a
  status, plus a few small integer counts;
* failure events whose message is ``lazycommit <operation> failed: <category>``
  built from fixed vocabularies, never from exception text.

Every outgoing event is rebuilt from an allowlist in ``before_send`` /
``before_send_transaction``; anything Lazy-Commit did not create is dropped.
The SDK runs with default integrations off (no argv, logging, HTTP
breadcrumbs, or excepthook), no local variables, no source context, no
breadcrumbs, and no PII. Telemetry errors never interrupt the CLI.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, TextIO

from lazycommit import __version__

DSN_ENV = "LAZYCOMMIT_SENTRY_DSN"
ENVIRONMENT_ENV = "LAZYCOMMIT_SENTRY_ENVIRONMENT"
SAMPLE_RATE_ENV = "LAZYCOMMIT_SENTRY_TRACES_SAMPLE_RATE"

OPERATIONS = frozenset({
    "plan", "model.request", "plan.validate", "commit.review", "commit.execute",
    "pr.prepare", "pr.publish", "pr.push", "pr.create", "changelog", "changelog.write",
})
CATEGORIES = frozenset({
    # local model
    "model_unavailable", "model_not_installed", "model_timeout", "model_http_error",
    "model_bad_response", "model_not_selected", "model_refused_host", "invalid_model_output",
    "unsafe_model_input",
    # repository / git
    "not_a_repository", "unsafe_repository_state", "git_error", "repository_changed",
    # plans and commits
    "blocked_groups", "rejected_groups", "commit_failed", "verification_failed", "interrupted",
    # pull requests
    "no_remote", "detached_head", "missing_base", "on_base_branch", "no_unique_commits",
    "no_net_change", "invalid_argument", "branch_error", "not_github", "gh_missing",
    "gh_unauthenticated", "gh_error", "outgoing_secrets", "remote_diverged", "push_failed",
    "pr_create_failed", "fallback_description",
    # changelog
    "changelog_conflict", "changelog_write_failed", "invalid_path",
    "other",
})
COUNT_KEYS = frozenset({"groups", "blocked_groups", "rejected_groups", "files", "excluded_files", "commits"})

_MESSAGE = re.compile(r"^lazycommit [a-z.]+ failed: [a-z_]+$")
_EVENT_KEYS = ("event_id", "timestamp", "level", "platform", "release", "environment", "sdk")
_TRANSACTION_KEYS = ("event_id", "type", "timestamp", "start_timestamp", "platform", "release", "environment", "sdk")
_SPAN_KEYS = ("trace_id", "span_id", "parent_span_id", "op", "status", "start_timestamp", "timestamp",
              "same_process_as_parent")
_TRACE_KEYS = ("trace_id", "span_id", "parent_span_id", "op", "status")

_sdk: Any = None  # the sentry_sdk module once telemetry is enabled


def enabled() -> bool:
    return _sdk is not None


def _label(value: str, allowed: frozenset[str]) -> str:
    return value if value in allowed else "other"


def init(env: Mapping[str, str] | None = None, warn: TextIO | None = None) -> bool:
    """Enable Sentry if configured. Returns True when telemetry is active."""
    global _sdk
    env = os.environ if env is None else env
    warn = warn or sys.stderr
    dsn = env.get(DSN_ENV, "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        warn.write(f"note: {DSN_ENV} is set but sentry-sdk is not installed "
                   "(pip install \"lazycommit[sentry]\"); telemetry is off.\n")
        return False
    try:
        rate = min(max(float(env.get(SAMPLE_RATE_ENV, "1.0")), 0.0), 1.0)
    except ValueError:
        rate = 1.0
    environment = re.sub(r"[^A-Za-z0-9_.-]", "", env.get(ENVIRONMENT_ENV, "production"))[:32] or "production"
    try:
        sentry_sdk.init(
            dsn=dsn,
            release=f"lazycommit@{__version__}",
            environment=environment,
            default_integrations=False,
            auto_enabling_integrations=False,
            integrations=[],
            # No user identity, request bodies, frame variables, or source lines.
            data_collection={"user_info": False, "http_bodies": [], "database_query_data": False,
                             "queues": False, "stack_frame_variables": False, "frame_context_lines": 0},
            include_local_variables=False,
            include_source_context=False,
            attach_stacktrace=False,
            max_breadcrumbs=0,
            server_name="lazycommit-cli",
            traces_sample_rate=rate,
            send_client_reports=False,
            before_send=_scrub_event,
            before_send_transaction=_scrub_transaction,
            before_breadcrumb=lambda crumb, hint: None,
            shutdown_timeout=2,
        )
    except Exception:  # e.g. a malformed DSN; never echo it back
        warn.write(f"note: {DSN_ENV} is not a valid Sentry DSN; telemetry is off.\n")
        return False
    _sdk = sentry_sdk
    return True


def shutdown() -> None:
    """Flush pending telemetry (bounded) and disable it."""
    global _sdk
    sdk, _sdk = _sdk, None
    if sdk is None:
        return
    try:
        sdk.flush(timeout=2)
        client = sdk.get_client()
        client.close(timeout=0)
    except Exception:
        pass


@contextmanager
def span(operation: str) -> Iterator[None]:
    """Time a workflow step; a transaction if none is active, else a child span."""
    sdk = _sdk
    if sdk is None:
        yield
        return
    op = _label(operation, OPERATIONS)
    try:
        if sdk.get_current_span() is None:
            manager = sdk.start_transaction(op=op, name=op)
        else:
            manager = sdk.start_span(op=op, name=op)
        current = manager.__enter__()
    except Exception:
        yield
        return
    status = "ok"
    try:
        yield
    except KeyboardInterrupt:
        status = "cancelled"
        raise
    except BaseException:
        status = "internal_error"
        raise
    finally:
        try:
            if getattr(current, "status", None) is None:
                current.set_status(status)
            # Never hand the exception to Sentry: its text may contain sensitive values.
            manager.__exit__(None, None, None)
        except Exception:
            pass


def count(key: str, value: int) -> None:
    """Attach a small integer count to the current span."""
    sdk = _sdk
    if sdk is None or key not in COUNT_KEYS:
        return
    try:
        current = sdk.get_current_span()
        if current is not None:
            current.set_data(key, int(value))
    except Exception:
        pass


def failure(operation: str, category: str, level: str = "error") -> None:
    """Report a failure as a fixed operation/category pair (no free text)."""
    sdk = _sdk
    if sdk is None:
        return
    op = _label(operation, OPERATIONS)
    cat = _label(category, CATEGORIES)
    try:
        current = sdk.get_current_span()
        if current is not None:
            current.set_status("internal_error" if level == "error" else "ok")
            current.set_tag("category", cat)
        sdk.capture_message(f"lazycommit {op} failed: {cat}", level="warning" if level == "warning" else "error",
                            tags={"operation": op, "category": cat})
    except Exception:
        pass


# --- scrubbing ---------------------------------------------------------------------


def _tags(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    if raw.get("operation") in OPERATIONS:
        out["operation"] = raw["operation"]
    if raw.get("category") in CATEGORIES:
        out["category"] = raw["category"]
    return out


def _contexts(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    trace = raw.get("trace")
    if isinstance(trace, dict):
        kept = {k: trace[k] for k in _TRACE_KEYS if k in trace}
        if kept.get("op") not in OPERATIONS:
            kept.pop("op", None)
        out["trace"] = kept
    runtime = raw.get("runtime")
    if isinstance(runtime, dict):
        out["runtime"] = {k: runtime[k] for k in ("name", "version") if k in runtime}
    os_ctx = raw.get("os")
    if isinstance(os_ctx, dict) and "name" in os_ctx:
        out["os"] = {"name": os_ctx["name"]}
    return out


def _counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k in COUNT_KEYS and isinstance(v, int) and not isinstance(v, bool)}


def _scrub_event(event: Any, hint: Any) -> Any:  # Sentry's Event is a TypedDict
    if not isinstance(event, dict):
        return None
    message = event.get("message")
    if not isinstance(message, str):
        logentry = event.get("logentry")
        message = logentry.get("message") if isinstance(logentry, dict) else None
    if not isinstance(message, str) or not _MESSAGE.match(message):
        return None  # not one of ours (e.g. an exception): drop it
    clean: dict[str, Any] = {k: event[k] for k in _EVENT_KEYS if k in event}
    clean.update({"message": message, "tags": _tags(event.get("tags")), "contexts": _contexts(event.get("contexts"))})
    return clean


def _scrub_transaction(event: Any, hint: Any) -> Any:
    name = event.get("transaction") if isinstance(event, dict) else None
    if name not in OPERATIONS:
        return None
    spans = []
    for s in event.get("spans") or []:
        if isinstance(s, dict) and s.get("op") in OPERATIONS:
            kept = {k: s[k] for k in _SPAN_KEYS if k in s}
            kept["description"] = s["op"]
            kept["data"] = _counts(s.get("data"))
            kept["tags"] = _tags(s.get("tags"))
            spans.append(kept)
    contexts = _contexts(event.get("contexts"))
    trace_data = (event.get("contexts") or {}).get("trace", {})
    if isinstance(trace_data, dict) and "trace" in contexts:
        contexts["trace"]["data"] = _counts(trace_data.get("data"))
    clean: dict[str, Any] = {k: event[k] for k in _TRANSACTION_KEYS if k in event}
    clean.update({"transaction": name, "transaction_info": {"source": "custom"}, "contexts": contexts,
                  "spans": spans, "tags": _tags(event.get("tags"))})
    return clean
