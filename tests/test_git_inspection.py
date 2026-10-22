"""Repository inspection: status parsing and change classification."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import Repo, run_cli

from lazycommit import gitrepo
from lazycommit.collect import collect


def test_not_a_repository(tmp_path: Path) -> None:
    with pytest.raises(gitrepo.NotARepositoryError):
        gitrepo.read_state(tmp_path)


def test_cli_not_a_repository(tmp_path: Path) -> None:
    from lazycommit import cli
    import io

    out = io.StringIO()
    assert cli.run(["--repo", str(tmp_path), "--model", "x"], out=out) == 1
    assert "not inside a Git working tree" in out.getvalue()


def test_clean_repository(repo: Repo) -> None:
    state = gitrepo.read_state(repo.root)
    assert state.branch == "main"
    assert state.files == []
    code, out = run_cli(repo, host="http://127.0.0.1:9")  # never contacted
    assert code == 0
    assert "clean - nothing to propose" in out
    assert "0 staged, 0 unstaged, 0 untracked" in out


def test_staged_changes(repo: Repo) -> None:
    repo.write("src/app.py", "def main():\n    return 2\n")
    repo.git("add", "src/app.py")
    state = gitrepo.read_state(repo.root)
    assert [f.path for f in state.staged] == ["src/app.py"]
    assert state.unstaged == [] and state.untracked == []
    c = collect(state).candidates[0]
    assert c.kinds == ["staged"]
    assert "+    return 2" in c.sections[0].text


def test_unstaged_changes(repo: Repo) -> None:
    repo.write("README.md", "# demo\nmore\n")
    state = gitrepo.read_state(repo.root)
    assert [f.path for f in state.unstaged] == ["README.md"]
    assert state.staged == []
    c = collect(state).candidates[0]
    assert c.kinds == ["unstaged"]
    assert "+more" in c.sections[0].text


def test_untracked_files(repo: Repo) -> None:
    repo.write("docs/guide.md", "hello\n")
    repo.write(".gitignore", "build/\n")
    repo.write("build/out.txt", "ignored artifact\n")
    state = gitrepo.read_state(repo.root)
    paths = sorted(f.path for f in state.untracked)
    # Ignored files are never listed; untracked dirs are expanded.
    assert paths == [".gitignore", "docs/guide.md"]
    c = next(c for c in collect(state).candidates if c.path == "docs/guide.md")
    assert c.kinds == ["untracked"]
    assert "+hello" in c.sections[0].text


def test_mixed_staged_and_unstaged_same_file(repo: Repo) -> None:
    repo.write("src/app.py", "def main():\n    return 2\n")
    repo.git("add", "src/app.py")
    repo.write("src/app.py", "def main():\n    return 3\n")
    state = gitrepo.read_state(repo.root)
    f = state.files[0]
    assert f.staged and f.unstaged
    c = collect(state).candidates[0]
    assert c.kinds == ["staged", "unstaged"]
    staged, unstaged = c.sections
    assert "staged" in staged.label and "+    return 2" in staged.text
    assert "unstaged" in unstaged.label and "+    return 3" in unstaged.text
    assert "-    return 2" in unstaged.text


def test_binary_and_deleted_files(repo: Repo) -> None:
    repo.write("img.bin", b"\x00\x01\x02binary")
    repo.commit_all("add binary")
    repo.write("img.bin", b"\x00\x09\x08changed")
    (repo.root / "README.md").unlink()
    repo.write("new.bin", b"\x00\xffnew")
    state = gitrepo.read_state(repo.root)
    cands = {c.path: c for c in collect(state).candidates}
    assert cands["img.bin"].binary
    assert cands["new.bin"].binary
    assert "content not shown" in cands["new.bin"].sections[0].text
    assert cands["README.md"].change == "deleted"
    assert "-# demo" in cands["README.md"].sections[0].text


def test_rename_is_reported(repo: Repo) -> None:
    repo.git("mv", "src/app.py", "src/main.py")
    state = gitrepo.read_state(repo.root)
    f = state.files[0]
    assert f.path == "src/main.py" and f.orig_path == "src/app.py"
    c = collect(state).candidates[0]
    assert "renamed (from src/app.py)" in c.change


def test_initial_repo_without_commits(tmp_path: Path) -> None:
    r = Repo(tmp_path)
    r.git("init", "-q", "-b", "trunk")
    r.write("a.txt", "a\n")
    r.git("add", "a.txt")
    state = gitrepo.read_state(tmp_path)
    assert state.branch == "trunk" and state.head_oid is None
    assert "+a" in collect(state).candidates[0].sections[0].text


def test_merge_conflict_state_is_refused(repo: Repo, fake_ollama: object) -> None:
    repo.git("checkout", "-q", "-b", "other")
    repo.write("README.md", "# other\n")
    repo.commit_all("other")
    repo.git("checkout", "-q", "main")
    repo.write("README.md", "# main\n")
    repo.commit_all("main")
    repo.git("merge", "other")  # conflicts
    state = gitrepo.read_state(repo.root)
    assert "merge in progress" in state.unsafe_reasons
    assert [f.path for f in state.conflicted] == ["README.md"]

    from conftest import FakeOllama

    assert isinstance(fake_ollama, FakeOllama)
    code, out = run_cli(repo, host=fake_ollama.url)
    assert code == 3
    assert "unresolved merge conflicts" in out
    assert "Nothing was sent to the model" in out
    assert fake_ollama.requests == []


def test_git_wrapper_refuses_write_commands(repo: Repo) -> None:
    for cmd in ("add", "commit", "reset", "checkout", "stash", "push", "update-index"):
        with pytest.raises(gitrepo.GitError, match="non-allowlisted"):
            gitrepo.run_git(repo.root, cmd, ".")


def test_porcelain_parser_handles_spaces_and_renames() -> None:
    raw = (
        b"# branch.oid abc\0# branch.head feature/x\0"
        b"1 .M N... 100644 100644 100644 h1 h2 dir/with space.txt\0"
        b"2 R. N... 100644 100644 100644 h1 h2 R100 new name.py\0old name.py\0"
        b"? untracked file\0"
    )
    branch, oid, files = gitrepo.parse_porcelain_v2(raw)
    assert branch == "feature/x" and oid == "abc"
    assert [f.path for f in files] == ["dir/with space.txt", "new name.py", "untracked file"]
    assert files[1].orig_path == "old name.py"
    assert files[0].unstaged and not files[0].staged
    assert files[1].staged
    assert files[2].untracked
