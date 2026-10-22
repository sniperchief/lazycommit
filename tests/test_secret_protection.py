"""Secret protection: flagged content must never appear in the model request."""

from __future__ import annotations

import pytest
from conftest import (
    FAKE_AWS_KEY,
    FAKE_ENV_VALUE,
    FAKE_GH_TOKEN,
    FAKE_KEY_BODY,
    FAKE_PRIVATE_KEY,
    FakeOllama,
    Repo,
    group,
    plan_reply,
    run_cli,
)

from lazycommit import secrets
from lazycommit.planner import UnsafeInputError, assert_safe


@pytest.mark.parametrize(
    "line,rule",
    [
        (f'AWS_KEY = "{FAKE_AWS_KEY}"', "AWS access key ID"),
        (f"token: {FAKE_GH_TOKEN}", "GitHub token"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", "private key block"),
        ("-----BEGIN PRIVATE KEY-----", "private key block"),
        ('api_key = "abcd1234efgh5678"', "hard-coded secret assignment"),
        ("DATABASE_URL=postgres://admin:hunter2pass@db.local/app", "credentials in URL"),
        ("key = sk-ant-" + "a" * 30, "Anthropic API key"),
        ("Authorization: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop", "JSON Web Token"),
    ],
)
def test_content_rules(line: str, rule: str) -> None:
    rules = {f.rule for f in secrets.scan_text(line)}
    assert rule in rules


def test_benign_code_is_not_flagged() -> None:
    text = 'password = os.environ["DB_PASSWORD"]\ntoken = get_token()\nkey = "short"\n'
    assert secrets.scan_text(text) == []


@pytest.mark.parametrize(
    "path,flagged",
    [
        (".env", True), ("config/.env.production", True), ("prod.env", True),
        ("certs/server.pem", True), ("id_rsa", True), ("id_rsa.pub", False),
        ("aws/credentials", True), ("gcp-credentials.json", True), (".npmrc", True),
        ("src/env.py", False), ("README.md", False), ("keyboard.py", False),
    ],
)
def test_sensitive_filenames(path: str, flagged: bool) -> None:
    assert (secrets.sensitive_filename(path) is not None) is flagged


def test_findings_never_contain_values() -> None:
    findings = secrets.scan_text(f"x = '{FAKE_AWS_KEY}'")
    assert all(FAKE_AWS_KEY not in f.describe() for f in findings)


def _setup_mixed_repo(repo: Repo) -> None:
    # Safe change that should reach the model.
    repo.write("src/app.py", "def main():\n    return 42\n")
    # Fake API key inside an otherwise normal source file (staged).
    repo.write("src/config.py", f'AWS_ACCESS_KEY_ID = "{FAKE_AWS_KEY}"\nREGION = "us-east-1"\n')
    repo.git("add", "src/config.py")
    # Fake token added as an unstaged edit to a tracked file.
    repo.write("README.md", f"# demo\nuse token {FAKE_GH_TOKEN}\n")
    # Private key material in an untracked, innocently-named file.
    repo.write("notes/deploy.txt", "deploy steps\n" + FAKE_PRIVATE_KEY)
    # .env file: excluded by name, content never read.
    repo.write(".env", f"DB_PASSWORD={FAKE_ENV_VALUE}\n")
    repo.write("config/.env.local", f"OTHER={FAKE_ENV_VALUE}\n")


def test_secrets_are_absent_from_actual_model_request(repo: Repo, fake_ollama: FakeOllama) -> None:
    _setup_mixed_repo(repo)
    fake_ollama.reply = plan_reply(group(["src/app.py"], "fix", "return the answer"))

    code, out = run_cli(repo, host=fake_ollama.url)

    assert code == 0, out
    assert len(fake_ollama.chat_bodies) == 1
    traffic = fake_ollama.all_traffic()
    for secret in (FAKE_AWS_KEY, FAKE_GH_TOKEN, FAKE_KEY_BODY, "PRIVATE KEY", FAKE_ENV_VALUE):
        assert secret not in traffic, f"{secret!r} leaked into model request"
    # Excluded files are not even named in the request.
    for path in ("src/config.py", "README.md", "notes/deploy.txt", ".env", "config/.env.local"):
        assert f"FILE: {path} " not in traffic
    # The safe file did reach the model.
    assert "FILE: src/app.py" in traffic and "return 42" in traffic


def test_exclusions_are_reported_without_values(repo: Repo, fake_ollama: FakeOllama) -> None:
    _setup_mixed_repo(repo)
    fake_ollama.reply = plan_reply(group(["src/app.py"], "fix", "return the answer"))
    code, out = run_cli(repo, host=fake_ollama.url)
    assert "Excluded by safety checks" in out
    assert "src/config.py: possible secret: AWS access key ID (line 1, staged changes)" in out
    assert "README.md: possible secret: GitHub token (line 2, unstaged changes)" in out
    assert "notes/deploy.txt: possible secret: private key block (line 2, untracked new file)" in out
    assert ".env: sensitive file type: environment file" in out
    for secret in (FAKE_AWS_KEY, FAKE_GH_TOKEN, FAKE_KEY_BODY, FAKE_ENV_VALUE):
        assert secret not in out


def test_show_input_never_contains_secrets(repo: Repo) -> None:
    _setup_mixed_repo(repo)
    code, out = run_cli(repo, "--show-input", host="http://127.0.0.1:9")
    assert code == 0
    assert "BEGIN SANITIZED MODEL INPUT" in out
    for secret in (FAKE_AWS_KEY, FAKE_GH_TOKEN, FAKE_KEY_BODY, FAKE_ENV_VALUE):
        assert secret not in out


def test_secret_beyond_truncation_point_still_excludes_file(repo: Repo, fake_ollama: FakeOllama) -> None:
    filler = "".join(f"line_{i} = {i}\n" for i in range(3000))
    repo.write("src/big.py", filler + f'TOKEN = "{FAKE_GH_TOKEN}"\n')
    repo.write("src/app.py", "def main():\n    return 7\n")
    fake_ollama.reply = plan_reply(group(["src/app.py"], "fix", "x"))
    code, out = run_cli(repo, "--max-file-chars", "1000", host=fake_ollama.url)
    assert code == 0
    assert FAKE_GH_TOKEN not in fake_ollama.all_traffic()
    assert "FILE: src/big.py" not in fake_ollama.all_traffic()
    assert "src/big.py: possible secret" in out


def test_only_sensitive_changes_sends_nothing(repo: Repo, fake_ollama: FakeOllama) -> None:
    repo.write(".env", f"DB_PASSWORD={FAKE_ENV_VALUE}\n")
    code, out = run_cli(repo, host=fake_ollama.url)
    assert code == 0
    assert "Nothing was sent to the model" in out
    assert fake_ollama.requests == []


def test_final_guard_blocks_secret_in_assembled_prompt() -> None:
    with pytest.raises(UnsafeInputError):
        assert_safe("harmless", f"leak {FAKE_AWS_KEY}")
