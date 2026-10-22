"""Local model integration: availability, validation, and failure handling (mocked)."""

from __future__ import annotations

import json
import socket

import pytest
from conftest import FakeOllama, Repo, group, plan_reply, run_cli

from lazycommit.collect import collect
from lazycommit.gitrepo import read_state
from lazycommit.ollama import ModelError, normalize_host
from lazycommit.planner import InvalidPlanError, parse_plan


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


def _two_changes(repo: Repo) -> None:
    repo.write("src/app.py", "def main():\n    return 2\n")
    repo.write("docs/usage.md", "usage\n")


def test_happy_path_proposal(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.reply = plan_reply(
        group(["src/app.py"], "fix", "return 2 from main", scope="app"),
        group(["docs/usage.md"], "docs", "add usage notes", confidence="high"),
    )
    code, out = run_cli(repo, host=fake_ollama.url)
    assert code == 0, out
    assert "PROPOSED COMMIT PLAN" in out and "proposal only" in out
    assert "[1] fix(app): return 2 from main" in out
    assert "[2] docs: add usage notes" in out
    assert "src/app.py  [unstaged]" in out
    assert "docs/usage.md  [untracked]" in out
    assert "did not stage, commit, or modify anything" in out

    body = json.loads(fake_ollama.chat_bodies[0])
    assert body["model"] == "fake-model"
    assert body["stream"] is False
    assert body["format"]["required"] == ["groups"]


def test_ollama_unavailable(repo: Repo) -> None:
    _two_changes(repo)
    code, out = run_cli(repo, host=f"http://127.0.0.1:{_free_port()}")
    assert code == 1
    assert "cannot reach Ollama" in out
    assert "ollama serve" in out
    assert "No proposal was produced" in out


def test_model_not_installed(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.models = ["llama3.2:latest"]
    code, out = run_cli(repo, host=fake_ollama.url, model="qwen2.5-coder:7b")
    assert code == 1
    assert "not installed" in out and "ollama pull qwen2.5-coder:7b" in out
    assert fake_ollama.chat_bodies == []


def test_no_model_configured_lists_installed(repo: Repo, fake_ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAZYCOMMIT_MODEL", raising=False)
    _two_changes(repo)
    code, out = run_cli(repo, host=fake_ollama.url, model=None)
    assert code == 1
    assert "no model selected" in out and "fake-model:latest" in out
    assert fake_ollama.chat_bodies == []


def test_model_tag_shorthand_accepted(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.reply = plan_reply(group(["src/app.py", "docs/usage.md"], "chore", "misc"))
    code, _ = run_cli(repo, host=fake_ollama.url, model="fake-model")
    assert code == 0


def test_malformed_json(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.reply = '{"groups": [ {"type": "feat", '
    code, out = run_cli(repo, host=fake_ollama.url)
    assert code == 1
    assert "rejected model output: model output is not valid JSON" in out
    assert "PROPOSED COMMIT PLAN" not in out


def test_wrong_shape_json(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.reply = '["feat: everything"]'
    code, out = run_cli(repo, host=fake_ollama.url)
    assert code == 1
    assert "does not match the plan schema" in out


def test_unknown_file_rejects_group(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.reply = plan_reply(
        group(["src/app.py", "src/invented.py"], "feat", "add invented module"),
        group(["docs/usage.md"], "docs", "add usage"),
    )
    code, out = run_cli(repo, host=fake_ollama.url)
    assert code == 0
    assert "add invented module" not in out.split("Needs manual review")[0]
    assert "group 1 rejected: references file(s) not in the candidate list: src/invented.py" in out
    assert "[1] docs: add usage" in out
    assert "src/app.py: not assigned to any valid group" in out


def test_excluded_file_cannot_be_smuggled_back(repo: Repo, fake_ollama: FakeOllama) -> None:
    repo.write(".env", "X=1\n")
    repo.write("src/app.py", "def main():\n    return 2\n")
    fake_ollama.reply = plan_reply(group(["src/app.py", ".env"], "chore", "update config"))
    code, out = run_cli(repo, host=fake_ollama.url)
    assert "group 1 rejected" in out and "PROPOSED" in out


def test_large_diff_is_bounded(repo: Repo, fake_ollama: FakeOllama) -> None:
    for i in range(6):
        repo.write(f"src/gen_{i}.py", "".join(f"value_{n} = {n}\n" for n in range(5000)))
    fake_ollama.reply = plan_reply()
    code, out = run_cli(
        repo, "--max-file-chars", "2000", "--max-total-chars", "5000", host=fake_ollama.url
    )
    assert code == 0
    request = json.loads(fake_ollama.chat_bodies[0])
    user = request["messages"][1]["content"]
    assert len(user) < 5000 + 3000  # diff budget plus headers/listing
    assert "truncated" in user
    assert "model input size budget exhausted" in out
    assert "truncated to fit the input budget" in out


def test_model_timeout(repo: Repo, fake_ollama: FakeOllama) -> None:
    _two_changes(repo)
    fake_ollama.delay = 3.0
    code, out = run_cli(repo, "--timeout", "0.5", host=fake_ollama.url)
    assert code == 1
    assert "did not respond within" in out
    assert "No proposal was produced" in out


def test_non_loopback_host_refused() -> None:
    for host in ("http://api.example.com:11434", "10.0.0.5:11434", "https://ollama.com"):
        with pytest.raises(ModelError, match="non-local"):
            normalize_host(host)
    assert normalize_host("localhost") == "http://localhost:11434"
    assert normalize_host("127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert normalize_host("http://[::1]:11434") == "http://[::1]:11434"


def test_parse_plan_validation_rules(repo: Repo) -> None:
    _two_changes(repo)
    cands = collect(read_state(repo.root)).candidates
    raw = plan_reply(
        group(["./src/app.py"], "feat", "feat(core): add thing"),  # prefix stripped, ./ normalized
        group(["docs/usage.md"], "wip", "bad type"),
        group(["docs/usage.md"], "docs", "dup a"),
        group(["docs/usage.md"], "docs", "dup b"),
    )
    plan = parse_plan(raw, cands)
    assert [g.header for g in plan.groups] == ["feat: add thing"]
    assert any("invalid commit type" in w for w in plan.warnings)
    assert {m.path: m.reason for m in plan.manual}["docs/usage.md"] == "model assigned it to multiple groups"

    with pytest.raises(InvalidPlanError):
        parse_plan("not json at all", cands)


def test_unassigned_reason_from_model(repo: Repo) -> None:
    _two_changes(repo)
    cands = collect(read_state(repo.root)).candidates
    raw = plan_reply(
        group(["src/app.py"], "fix", "x"),
        unassigned=[{"file": "docs/usage.md", "reason": "unclear intent"}],
    )
    plan = parse_plan(raw, cands)
    assert [(m.path, m.reason) for m in plan.manual] == [("docs/usage.md", "model: unclear intent")]
