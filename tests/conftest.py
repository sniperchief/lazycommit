"""Shared fixtures: throwaway Git repos and a fake local Ollama server.

All repositories live under pytest's tmp_path and all credentials are fake.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from lazycommit import cli

# Fake credentials. They match detector patterns but are not real secrets.
FAKE_AWS_KEY = "AKIA" + "FAKEFAKEFAKE1234"
FAKE_GH_TOKEN = "ghp_" + "Fake0123456789Fake0123456789FakeAB"
FAKE_KEY_BODY = "MIIFAKEKEYMATERIALdoNotUseThisIsATestFixture0123456789"
FAKE_PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n" + FAKE_KEY_BODY + "\n-----END RSA PRIVATE KEY-----\n"
)
FAKE_ENV_VALUE = "super-fake-db-password-value-42"


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root

    def git(self, *args: str) -> str:
        env = dict(os.environ)
        env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"})
        proc = subprocess.run(
            ["git", *args], cwd=self.root, env=env, capture_output=True, check=False
        )
        if proc.returncode != 0 and args[0] != "merge":
            raise AssertionError(f"git {args} failed: {proc.stderr.decode()}")
        return proc.stdout.decode("utf-8", "replace")

    def write(self, rel: str, content: str | bytes) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_bytes(content.encode("utf-8"))
        else:
            path.write_bytes(content)
        return path

    def commit_all(self, message: str = "commit") -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    root = tmp_path / "repo"
    root.mkdir()
    r = Repo(root)
    r.git("init", "-q", "-b", "main")
    r.git("config", "user.name", "Test User")
    r.git("config", "user.email", "test@example.invalid")
    r.git("config", "commit.gpgsign", "false")
    r.git("config", "core.autocrlf", "false")
    r.write("README.md", "# demo\n")
    r.write("src/app.py", "def main():\n    return 1\n")
    r.commit_all("initial")
    return r


@dataclass
class FakeOllama:
    url: str
    models: list[str] = field(default_factory=lambda: ["fake-model:latest"])
    reply: str | Callable[[dict[str, Any]], str] = '{"groups": []}'
    delay: float = 0.0
    requests: list[tuple[str, str, bytes]] = field(default_factory=list)

    @property
    def chat_bodies(self) -> list[str]:
        return [body.decode("utf-8") for method, path, body in self.requests if path == "/api/chat"]

    def all_traffic(self) -> str:
        return "\n".join(body.decode("utf-8", "replace") for _, _, body in self.requests)


@pytest.fixture
def fake_ollama() -> Iterator[FakeOllama]:
    state = FakeOllama(url="")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, payload: Any) -> None:
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            state.requests.append(("GET", self.path, b""))
            if self.path == "/api/tags":
                self._send({"models": [{"name": m} for m in state.models]})
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            state.requests.append(("POST", self.path, body))
            if self.path != "/api/chat":
                self.send_error(404)
                return
            if state.delay:
                time.sleep(state.delay)
            reply = state.reply(json.loads(body)) if callable(state.reply) else state.reply
            try:
                self._send({"model": "fake", "message": {"role": "assistant", "content": reply}, "done": True})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def run_cli(repo: Repo, *args: str, host: str | None = None, model: str | None = "fake-model") -> tuple[int, str]:
    argv = ["--repo", str(repo.root)]
    if host:
        argv += ["--ollama-host", host]
    if model:
        argv += ["--model", model]
    out = io.StringIO()
    code = cli.run([*argv, *args], out=out)
    return code, out.getvalue()


def plan_reply(*groups: dict[str, Any], unassigned: list[dict[str, str]] | None = None) -> str:
    return json.dumps({"groups": list(groups), "unassigned": unassigned or []})


def group(files: list[str], type_: str = "feat", message: str = "add thing", **extra: Any) -> dict[str, Any]:
    g: dict[str, Any] = {
        "type": type_,
        "scope": "",
        "message": message,
        "files": files,
        "rationale": "diff adds code",
        "confidence": "medium",
    }
    g.update(extra)
    return g


def snapshot(root: Path) -> dict[str, str]:
    """Hash every file under the repo, including .git, to detect any write."""
    result: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            result[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


class Script:
    """Answers interactive questions in order, like a user typing them.

    ``before`` maps a substring of a question to a callable run just before
    that question is answered (e.g. to change the repository mid-flow).
    """

    def __init__(self, answers: list[str], before: dict[str, Callable[[str], object]] | None = None) -> None:
        self.answers = list(answers)
        self.before = before or {}
        self.questions: list[str] = []

    def __call__(self, question: str) -> str:
        self.questions.append(question)
        for key, action in self.before.items():
            if key in question:
                action(question)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def run_cmd(repo: Repo, *argv: str, prompt: Callable[[str], str] | None = None) -> tuple[int, str]:
    """Run `lazycommit <argv...>` (e.g. "pr", "--no-model") against ``repo``."""
    out = io.StringIO()
    code = cli.run([argv[0], "--repo", str(repo.root), *argv[1:]], out=out, prompt=prompt, err=out)
    return code, out.getvalue()


@pytest.fixture
def remote_repo(repo: Repo, tmp_path: Path) -> tuple[Repo, Path]:
    """``repo`` with a real local bare repository as ``origin`` and main pushed to it."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True, capture_output=True)
    repo.git("remote", "add", "origin", bare.as_posix())
    repo.git("push", "-q", "-u", "origin", "main")
    repo.git("remote", "set-head", "origin", "main")
    return repo, bare
