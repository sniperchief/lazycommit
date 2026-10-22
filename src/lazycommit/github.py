"""GitHub access through the GitHub CLI (``gh``), and the one allowed ``git push``.

* Authentication is entirely ``gh``'s: Lazy-Commit never asks for, reads,
  stores, or prints a token, and discards ``gh auth status`` output because
  it can include a masked token.
* Only three ``gh`` operations exist here: ``auth status``, ``pr list``, and
  ``pr create``. Nothing edits, closes, or merges pull requests.
* :func:`push_branch` builds the only push Lazy-Commit can run: the current
  branch to the same-named remote branch, with no force, no tags, and no
  submodules. A non-fast-forward push is rejected by Git itself.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lazycommit import secrets
from lazycommit.branch import _SAFE_NAME
from lazycommit.gitrepo import _SAFE_CONFIG, _git_env

GH_TIMEOUT_SECONDS = 120
PUSH_TIMEOUT_SECONDS = 300
_PR_URL = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/pull/\d+")

INSTALL_HINT = (
    "Install the GitHub CLI from https://cli.github.com/ (Windows: `winget install --id GitHub.cli -e`), "
    "open a new terminal, then run `gh auth login`. Previews and local commits work without it."
)


class GitHubError(Exception):
    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


def find_gh() -> str | None:
    """Locate the gh executable ($LAZYCOMMIT_GH overrides PATH)."""
    override = os.environ.get("LAZYCOMMIT_GH")
    path = override if override else shutil.which("gh")
    if not path or not Path(path).is_file():
        return None
    # Batch wrappers are interpreted by cmd.exe, which re-parses arguments.
    if Path(path).suffix.lower() in (".bat", ".cmd"):
        return None
    return path


def _gh_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1", "GH_SPINNER_DISABLED": "1",
                "NO_COLOR": "1", "GH_PAGER": "cat", "PAGER": "cat"})
    return env


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return secrets.redact(lines[-1])[:300] if lines else ""


@dataclass
class ExistingPR:
    number: int
    url: str
    title: str
    base: str
    draft: bool


@dataclass
class GitHub:
    gh: str
    owner: str
    repo: str
    cwd: Path

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    def _run(self, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        # Bytes in, so Windows text mode cannot turn the body's newlines into CRLF.
        try:
            proc = subprocess.run(
                [self.gh, *args], cwd=self.cwd, env=_gh_env(), input=(stdin or "").encode("utf-8"),
                capture_output=True, timeout=GH_TIMEOUT_SECONDS, shell=False, check=False,
            )
        except subprocess.TimeoutExpired:
            raise GitHubError(f"gh {args[0]} timed out after {GH_TIMEOUT_SECONDS}s",
                              "Check your network connection and try again.") from None
        except OSError as exc:
            raise GitHubError(f"could not run gh ({type(exc).__name__})", INSTALL_HINT) from None
        return subprocess.CompletedProcess(proc.args, proc.returncode,
                                           proc.stdout.decode("utf-8", "replace"),
                                           proc.stderr.decode("utf-8", "replace"))

    def check_auth(self) -> None:
        proc = self._run(["auth", "status", "--hostname", "github.com"])
        if proc.returncode != 0:  # output deliberately discarded: it can show a masked token
            raise GitHubError("the GitHub CLI is not logged in to github.com",
                              "Run `gh auth login` (choose GitHub.com and HTTPS), then try again.")

    def open_prs(self, branch: str) -> list[ExistingPR]:
        proc = self._run(["pr", "list", f"--repo={self.slug}", f"--head={branch}", "--state=open",
                          "--json=number,url,title,baseRefName,isDraft", "--limit=20"])
        if proc.returncode != 0:
            raise GitHubError(f"could not list pull requests on {self.slug}: {_last_line(proc.stderr)}",
                              "Check that the repository exists and your account can access it.")
        try:
            rows = json.loads(proc.stdout or "[]")
        except ValueError:
            raise GitHubError("gh returned unreadable pull request data") from None
        return [
            ExistingPR(int(r["number"]), str(r["url"]), str(r.get("title", "")),
                       str(r.get("baseRefName", "")), bool(r.get("isDraft")))
            for r in rows if isinstance(r, dict) and "number" in r and "url" in r
        ]

    def create_pr(self, base: str, branch: str, title: str, body: str, draft: bool) -> str:
        args = ["pr", "create", f"--repo={self.slug}", f"--base={base}", f"--head={branch}",
                f"--title={title}", "--body-file=-"]
        if draft:
            args.append("--draft")
        proc = self._run(args, stdin=body)
        if proc.returncode != 0:
            raise GitHubError(f"GitHub did not create the pull request: {_last_line(proc.stderr) or 'unknown error'}")
        match = _PR_URL.search(proc.stdout)
        if not match:
            raise GitHubError("gh reported success but returned no pull request URL",
                              f"Check https://github.com/{self.slug}/pulls to see whether it was created.")
        return match.group(0)


def push_command(remote: str, branch: str) -> list[str]:
    """The exact push Lazy-Commit runs. Validated; never forces."""
    if not (_SAFE_NAME.match(remote) and _SAFE_NAME.match(branch)) or any(c in branch for c in "+:^~{}"):
        raise GitHubError("refusing to push: unsafe remote or branch name")
    refspec = f"refs/heads/{branch}:refs/heads/{branch}"
    return ["git", "--no-pager", *_SAFE_CONFIG, "-c", "push.followTags=false",
            "push", "--porcelain", "--no-recurse-submodules", "--set-upstream", remote, refspec]


def push_branch(root: Path, remote: str, branch: str) -> None:
    cmd = push_command(remote, branch)
    try:
        proc = subprocess.run(cmd, cwd=root, env=_git_env(), stdin=subprocess.DEVNULL, capture_output=True,
                              timeout=PUSH_TIMEOUT_SECONDS, shell=False, check=False)
    except subprocess.TimeoutExpired:
        raise GitHubError(f"git push timed out after {PUSH_TIMEOUT_SECONDS}s",
                          "Check your network and whether the push completed with `git status`.") from None
    except FileNotFoundError:
        raise GitHubError("git executable not found on PATH") from None
    if proc.returncode != 0:
        text = (proc.stderr + proc.stdout).decode("utf-8", "replace")
        lines = [ln for ln in text.splitlines()
                 if ln.strip() and not ln.startswith(("hint:", "To ", "Done"))]
        detail = secrets.redact(lines[-1].strip())[:300] if lines else f"exit {proc.returncode}"
        raise GitHubError(f"git push failed: {detail}",
                          "Nothing was force-pushed. Fix the problem (e.g. pull new remote commits) and retry.")
