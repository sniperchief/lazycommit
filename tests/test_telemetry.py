"""Sentry telemetry: off by default, privacy-preserving, and never disruptive.

The real sentry-sdk and its real HTTP transport are used, pointed at a
capture server on 127.0.0.1, so these tests check the exact bytes that would
leave the machine. Nothing is sent to sentry.io.
"""

from __future__ import annotations

import gzip
import io
import json
import socket
import subprocess
import sys
import threading
import time
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from conftest import FAKE_AWS_KEY, FakeOllama, Repo, Script, group, plan_reply, run_cli, run_cmd

from lazycommit import cli, telemetry


@dataclass
class Capture:
    dsn: str
    bodies: list[bytes] = field(default_factory=list)

    @property
    def raw(self) -> str:
        return "\n".join(b.decode("utf-8", "replace") for b in self.bodies)

    def items(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for body in self.bodies:
            lines = [ln for ln in body.split(b"\n") if ln.strip()]
            for header, payload in zip(lines[1::2], lines[2::2]):
                item = json.loads(payload)
                item["_type"] = json.loads(header).get("type")
                out.append(item)
        return out

    def messages(self) -> list[str]:
        return [i["message"] for i in self.items() if i["_type"] == "event"]

    def transactions(self) -> list[dict[str, Any]]:
        return [i for i in self.items() if i["_type"] == "transaction"]


@pytest.fixture
def sentry() -> Iterator[Capture]:
    cap = Capture(dsn="")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            encoding = self.headers.get("Content-Encoding", "")
            if encoding == "gzip":
                body = gzip.decompress(body)
            elif encoding == "deflate":
                body = zlib.decompress(body)
            cap.bodies.append(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cap.dsn = f"http://publickey@127.0.0.1:{server.server_address[1]}/7"
    try:
        yield cap
    finally:
        telemetry.shutdown()
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _telemetry_off() -> Iterator[None]:
    yield
    telemetry.shutdown()


def _enable(cap: Capture) -> None:
    assert telemetry.init({telemetry.DSN_ENV: cap.dsn}, warn=io.StringIO())


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


# --- off by default ---------------------------------------------------------------


def test_disabled_by_default_and_sdk_not_imported() -> None:
    code = ("import sys; from lazycommit import telemetry; "
            "print(telemetry.init({}), telemetry.enabled(), 'sentry_sdk' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False", "False"]


def test_cli_sends_nothing_without_dsn(repo: Repo, sentry: Capture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(telemetry.DSN_ENV, raising=False)
    repo.write("src/app.py", "x = 1\n")
    assert telemetry.init() is False
    run_cli(repo, "--show-input")
    telemetry.shutdown()
    assert sentry.bodies == []


def test_invalid_dsn_is_ignored_without_echoing_it() -> None:
    warn = io.StringIO()
    assert telemetry.init({telemetry.DSN_ENV: "not-a-dsn-secretvalue"}, warn=warn) is False
    assert "not a valid Sentry DSN" in warn.getvalue() and "secretvalue" not in warn.getvalue()


def test_missing_sdk_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)  # import now raises ImportError
    warn = io.StringIO()
    assert telemetry.init({telemetry.DSN_ENV: "http://k@127.0.0.1:9/1"}, warn=warn) is False
    assert "sentry-sdk is not installed" in warn.getvalue()


# --- what is sent ---------------------------------------------------------------------


def _sensitive_values(repo: Repo) -> list[str]:
    root = str(repo.root)
    return [FAKE_AWS_KEY, root, root.replace("\\", "\\\\"), repo.root.as_posix(), str(Path.home()),
            "src/config.py", "src/app.py", "feature/secret-branch-xyz", "FILE:", "diff --git",
            "MODEL-OUTPUT-MARKER", "Lazy-Commit (preview", "PROMPT"]


def test_model_failure_reported_as_safe_category(repo: Repo, sentry: Capture) -> None:
    repo.git("switch", "-q", "-c", "feature/secret-branch-xyz")
    repo.write("src/app.py", "def main():\n    return 2\n")
    repo.write("src/config.py", f"KEY = '{FAKE_AWS_KEY}'\n")
    _enable(sentry)

    code, _ = run_cli(repo, host=f"http://127.0.0.1:{_free_port()}")
    telemetry.shutdown()

    assert code == 1
    assert sentry.messages() == ["lazycommit model.request failed: model_unavailable"]
    txn = sentry.transactions()[0]
    assert txn["transaction"] == "plan"
    assert {s["op"] for s in txn["spans"]} == {"model.request"}
    assert txn["contexts"]["trace"]["data"] == {"files": 1, "excluded_files": 1}
    for value in _sensitive_values(repo):
        assert value not in sentry.raw, value


def test_malformed_model_output_is_a_category_not_content(repo: Repo, sentry: Capture,
                                                          fake_ollama: FakeOllama) -> None:
    repo.write("src/app.py", "def main():\n    return 2\n")
    fake_ollama.reply = '{"groups": "MODEL-OUTPUT-MARKER PROMPT'
    _enable(sentry)
    code, _ = run_cli(repo, host=fake_ollama.url)
    telemetry.shutdown()
    assert code == 1
    assert sentry.messages() == ["lazycommit plan.validate failed: invalid_model_output"]
    for value in _sensitive_values(repo):
        assert value not in sentry.raw, value


def test_blocked_and_rejected_groups_are_counted(repo: Repo, sentry: Capture, fake_ollama: FakeOllama) -> None:
    repo.write("src/app.py", "def main():\n    return 2\n")
    repo.git("add", "src/app.py")
    repo.write("src/app.py", "def main():\n    return 3\n")  # half-staged: blocked
    fake_ollama.reply = plan_reply(group(["src/app.py"], "fix", "return 3"),
                                   group(["src/ghost.py"], "feat", "invented"))
    _enable(sentry)
    run_cli(repo, "--json", host=fake_ollama.url)
    telemetry.shutdown()
    assert set(sentry.messages()) == {"lazycommit plan.validate failed: rejected_groups",
                                      "lazycommit plan.validate failed: blocked_groups"}
    assert "ghost" not in sentry.raw and "return 3" not in sentry.raw


def test_commit_execution_failure(repo: Repo, sentry: Capture, fake_ollama: FakeOllama) -> None:
    repo.write("src/app.py", "def main():\n    return 2\n")
    hook = repo.root / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'hook says no' >&2\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    fake_ollama.reply = plan_reply(group(["src/app.py"], "fix", "return two from main"))
    _enable(sentry)
    out = io.StringIO()
    code = cli.run(["--repo", str(repo.root), "--ollama-host", fake_ollama.url, "--model", "fake-model",
                    "--review"], out=out, prompt=Script(["y", "yes"]))
    telemetry.shutdown()
    assert code == 1, out.getvalue()
    assert "lazycommit commit.execute failed: commit_failed" in sentry.messages()
    ops = {s["op"] for t in sentry.transactions() for s in t["spans"]}
    assert {"commit.review", "commit.execute"} <= ops
    for text in ("return two from main", "hook says no", "yes"):
        assert text not in sentry.raw


def test_pr_and_changelog_failures(repo: Repo, sentry: Capture) -> None:
    repo.git("switch", "-q", "-c", "feature/secret-branch-xyz")
    repo.write("a.py", "a\n")
    repo.commit_all("feat: add a")
    _enable(sentry)
    run_cmd(repo, "pr", "--no-model")

    path = repo.root / "CHANGELOG.md"
    path.write_text("# Changelog\n", encoding="utf-8")
    run_cmd(repo, "changelog", "--write",
            prompt=Script(["write"], before={"Type 'write'": lambda _: path.write_text("edited\n")}))
    telemetry.shutdown()
    assert sentry.messages() == ["lazycommit pr.prepare failed: no_remote",
                                 "lazycommit changelog.write failed: changelog_conflict"]
    assert {t["transaction"] for t in sentry.transactions()} == {"pr.prepare", "changelog"}
    for value in _sensitive_values(repo) + ["CHANGELOG.md", "feat: add a"]:
        assert value not in sentry.raw, value


# --- never disruptive ------------------------------------------------------------------


def test_unreachable_sentry_does_not_break_or_stall(repo: Repo, fake_ollama: FakeOllama) -> None:
    repo.write("src/app.py", "def main():\n    return 2\n")
    fake_ollama.reply = plan_reply(group(["src/app.py"], "fix", "return 2"))
    assert telemetry.init({telemetry.DSN_ENV: f"http://k@127.0.0.1:{_free_port()}/1"}, warn=io.StringIO())
    start = time.monotonic()
    code, out = run_cli(repo, host=fake_ollama.url)
    telemetry.shutdown()
    assert code == 0 and "PROPOSED COMMIT PLAN" in out
    assert time.monotonic() - start < 15


class _Exploding:
    def __getattr__(self, name: str) -> Any:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("telemetry backend exploded")
        return boom


def test_telemetry_exceptions_never_escape(repo: Repo, fake_ollama: FakeOllama,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    repo.write("src/app.py", "def main():\n    return 2\n")
    fake_ollama.reply = "not json"
    monkeypatch.setattr(telemetry, "_sdk", _Exploding())
    with telemetry.span("plan"):
        telemetry.count("groups", 1)
        telemetry.failure("plan", "other")
    code, out = run_cli(repo, host=fake_ollama.url)
    assert code == 1 and "rejected model output" in out


# --- scrubbers (defense in depth) ---------------------------------------------------------


def test_event_scrubber_drops_foreign_events_and_fields() -> None:
    assert telemetry._scrub_event({"message": f"leak {FAKE_AWS_KEY}"}, {}) is None
    assert telemetry._scrub_event({"exception": {"values": [{"value": "C:/Users/x"}]}}, {}) is None
    event = {
        "message": "lazycommit pr.publish failed: push_failed", "level": "error", "event_id": "e",
        "tags": {"operation": "pr.publish", "category": "push_failed", "branch": "feature/x"},
        "user": {"ip_address": "1.2.3.4"}, "request": {"url": "u"}, "extra": {"sys.argv": ["--repo", "C:/x"]},
        "breadcrumbs": {"values": [{"message": "m"}]}, "server_name": "my-laptop", "modules": {"a": "1"},
        "contexts": {"os": {"name": "Windows", "kernel_version": "k"}, "device": {"model": "m"}},
    }
    clean = telemetry._scrub_event(event, {})
    assert clean == {"event_id": "e", "level": "error", "message": "lazycommit pr.publish failed: push_failed",
                     "tags": {"operation": "pr.publish", "category": "push_failed"},
                     "contexts": {"os": {"name": "Windows"}}}


def test_transaction_scrubber_keeps_only_known_spans_and_counts() -> None:
    event = {
        "transaction": "pr.publish", "contexts": {"trace": {"trace_id": "t", "op": "pr.publish",
                                                           "data": {"files": 2, "thread.name": "MainThread"}}},
        "spans": [
            {"op": "pr.push", "description": "git push origin feature/x", "data": {"url": "x", "groups": 1}},
            {"op": "http.client", "description": "GET https://api.github.com"},
        ],
    }
    clean = telemetry._scrub_transaction(event, {})
    assert clean["spans"] == [{"op": "pr.push", "description": "pr.push", "data": {"groups": 1}, "tags": {}}]
    assert clean["contexts"]["trace"]["data"] == {"files": 2}
    assert telemetry._scrub_transaction({"transaction": "custom user text"}, {}) is None


def test_unknown_labels_become_other(sentry: Capture) -> None:
    _enable(sentry)
    telemetry.failure("rm -rf /home/user", f"oops {FAKE_AWS_KEY}")
    telemetry.shutdown()
    assert sentry.messages() == ["lazycommit other failed: other"]
