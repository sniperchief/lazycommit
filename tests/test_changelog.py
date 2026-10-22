"""Phase 3 changelog: deterministic sections, preview by default, approved writes only."""

from __future__ import annotations

from pathlib import Path

from conftest import FAKE_AWS_KEY, Repo, Script, run_cmd, snapshot

from lazycommit import changelog


def _history(repo: Repo) -> None:
    repo.git("switch", "-q", "-c", "release-prep")
    for name, message in [
        ("a.py", "feat(api): add pagination"),
        ("b.py", "fix: handle empty response"),
        ("c.md", "docs: explain pagination"),
        ("d.py", "Fix crash when offline"),
        ("e.py", "misc tweaks"),
        ("f.py", "feat!: drop python 3.8 support"),
    ]:
        repo.write(name, name + "\n")
        repo.commit_all(message)


def test_preview_groups_commits_and_changes_nothing(repo: Repo) -> None:
    _history(repo)
    repo.git("status")
    before = snapshot(repo.root)

    code, out = run_cmd(repo, "changelog")

    assert code == 0, out
    assert "preview only - no files will be changed" in out
    assert "### Features\n- **api:** add pagination (" in out
    assert "### Bug fixes\n- handle empty response (" in out
    assert "### Documentation\n- explain pagination (" in out
    assert "### Breaking changes\n- drop python 3.8 support (" in out
    assert "### Unclassified (please review)" in out
    assert "- Fix crash when offline (" in out and "- likely: Bug fixes" in out
    assert "- misc tweaks (" in out
    assert "Entries:    4 confirmed by Conventional Commit type, 2 unclassified" in out
    assert "passed" not in out.lower()
    assert not (repo.root / "CHANGELOG.md").exists()
    assert snapshot(repo.root) == before


def test_since_tag_and_no_remote_needed(repo: Repo) -> None:
    repo.git("tag", "v1.0.0")
    repo.write("x.py", "x\n")
    repo.commit_all("feat: add x")
    code, out = run_cmd(repo, "changelog", "--since", "v1.0.0", "--title", "v1.1.0")
    assert code == 0, out
    assert "## v1.1.0 - " in out and "- add x (" in out


def test_secret_in_commit_message_is_withheld(repo: Repo) -> None:
    repo.git("switch", "-q", "-c", "x")
    repo.git("commit", "-q", "--allow-empty", "-m", f"chore: set key {FAKE_AWS_KEY}")
    code, out = run_cmd(repo, "changelog")
    assert FAKE_AWS_KEY not in out
    assert "(commit message withheld: it matched a secret pattern)" in out


def test_write_requires_explicit_confirmation(repo: Repo) -> None:
    _history(repo)
    for answers in (["yes"], ["y"], []):
        code, out = run_cmd(repo, "changelog", "--write", prompt=Script(answers))
        assert code == 0 and "Cancelled. Nothing was written." in out
    assert not (repo.root / "CHANGELOG.md").exists()


def test_write_inserts_above_previous_release(repo: Repo) -> None:
    _history(repo)
    existing = "# Changelog\n\nAll notable changes.\n\n## 1.0.0 - 2026-01-01\n\n- first\n"
    (repo.root / "CHANGELOG.md").write_bytes(existing.encode())

    code, out = run_cmd(repo, "changelog", "--write", prompt=Script(["write"]))

    assert code == 0, out
    text = (repo.root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.startswith("# Changelog\n\nAll notable changes.\n\n## Unreleased - ")
    assert text.index("## Unreleased") < text.index("## 1.0.0")
    assert text.endswith("## 1.0.0 - 2026-01-01\n\n- first\n")
    assert "not staged or committed" in out
    assert "CHANGELOG.md" in repo.git("status", "--porcelain")  # left for the user to review
    assert repo.git("diff", "--cached", "--name-only") == ""


def test_write_creates_file_and_keeps_crlf(repo: Repo) -> None:
    _history(repo)
    code, _ = run_cmd(repo, "changelog", "--write", prompt=Script(["write"]))
    assert code == 0
    assert (repo.root / "CHANGELOG.md").read_text(encoding="utf-8").startswith("# Changelog\n\n## Unreleased")
    assert changelog.insert_section("# C\r\n\r\n## old\r\n", "## new\n- x\n") == "# C\r\n\r\n## new\r\n- x\r\n\r\n## old\r\n"


def test_file_changed_after_preview_is_not_overwritten(repo: Repo) -> None:
    _history(repo)
    path = repo.root / "CHANGELOG.md"
    path.write_text("# Changelog\n", encoding="utf-8")

    def someone_edits(_: str) -> None:
        path.write_text("# Changelog\n\nedited meanwhile\n", encoding="utf-8")

    code, out = run_cmd(repo, "changelog", "--write", prompt=Script(["write"], before={"Type 'write'": someone_edits}))
    assert code == 4
    assert "changed since the preview" in out
    assert path.read_text(encoding="utf-8") == "# Changelog\n\nedited meanwhile\n"


def test_write_refuses_paths_outside_repository(repo: Repo, tmp_path: Path) -> None:
    _history(repo)
    for bad in ("../outside.md", ".git/CHANGELOG.md", str(tmp_path / "abs.md")):
        code, out = run_cmd(repo, "changelog", "--write", "--file", bad, prompt=Script(["write"]))
        assert code == 1, bad
    assert not (tmp_path / "outside.md").exists()
