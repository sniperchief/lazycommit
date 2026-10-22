# Lazy-Commit

A local-first CLI that reads your messy Git working tree and proposes logical
[Conventional Commits](https://www.conventionalcommits.org/) using an
open-weight model running in [Ollama](https://ollama.com) on your machine.

By default Lazy-Commit is **preview-only**: it never stages, commits, or
modifies anything. With `--review` you can approve individual proposed commits,
and Lazy-Commit then creates ordinary local commits for exactly those groups.
`lazycommit pr` drafts a pull request from your branch's commits, and only
`lazycommit pr --publish` pushes or opens it on GitHub, each step after its own
typed confirmation. Nothing is ever force-pushed, amended, or rewritten, and
repository content is never sent to an external AI service.

## Install

Requires Python 3.10+, Git, and Ollama.

```sh
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"

# Install Ollama from https://ollama.com/download, then pull any model:
ollama pull qwen2.5-coder:7b
```

## Run

From inside any Git repository:

```sh
lazycommit --model qwen2.5-coder:7b
# or: export LAZYCOMMIT_MODEL=qwen2.5-coder:7b; lazycommit
```

| Option | Purpose |
| --- | --- |
| `--model NAME` / `LAZYCOMMIT_MODEL` | Ollama model to use (required; no hardcoded default) |
| `--ollama-host URL` / `OLLAMA_HOST` | Local Ollama URL (loopback only), default `http://127.0.0.1:11434` |
| `--repo PATH` | Analyze a repo other than the current directory |
| `--timeout SECONDS` | Model timeout (default 180) |
| `--max-file-chars`, `--max-total-chars`, `--max-files` | Bound what is sent to the model |
| `--show-input` | Print the exact sanitized model input and exit without calling the model |
| `--json` | Print the proposal as JSON on stdout (progress on stderr). Never commits; for scripts |
| `--review` | Review each proposed commit interactively and create the approved ones (needs a terminal) |

Exit codes: `0` success (proposal shown, nothing to do, or approved commits
created), `1` error or failed commit, `2` usage error (including `--review`
without a terminal), `3` repository has conflicts or an operation in progress,
`4` repository changed after the plan was generated, `130` interrupted.

### Reviewing and committing

```sh
lazycommit --model qwen2.5-coder:7b             # 1. preview only (default)
lazycommit --model qwen2.5-coder:7b --json      #    same, machine-readable; never commits
lazycommit --model qwen2.5-coder:7b --review    # 2. approve groups, then commit them
```

In review mode each proposed group shows its commit message, files (and
whether each change is staged, unstaged, or untracked), rationale, confidence,
and notes. Answer `y`, `n`, or `q` for each group, then type `yes` at the final
confirmation. Anything else, including end of input, commits nothing.
`--review` refuses to run when stdin is not a terminal, so piping `yes` into it
cannot approve anything.

Groups that cannot be committed safely are shown as **BLOCKED** and are never
offered for approval:

- a file with both staged and unstaged changes (the two cannot be separated
  safely, so neither is committed),
- binary files and symlinks (their content was not secret-scanned),
- submodules, file-type changes, and renames whose source has other changes,
- anything excluded by the secret checks or not in the scanned candidate list,
- commit messages containing control characters or secret-like values.

Statuses: `PROPOSED` -> `APPROVED` / `REJECTED` / `BLOCKED`, then
`COMMITTED` (with its hash), `FAILED`, or `NOT RUN` (execution stopped earlier).

### Pull requests

Requires the [GitHub CLI](https://cli.github.com/) for publishing only
(`winget install --id GitHub.cli -e`, then `gh auth login`). Run these on a
feature branch whose commits are not yet in the base branch.

```sh
lazycommit pr                         # preview title + description (no push, no GitHub)
lazycommit pr --no-model              # preview built only from commit messages
lazycommit pr --json                  # preview as JSON; never publishes
lazycommit pr --show-input            # exact sanitized text the model would see
lazycommit pr --publish               # review, edit, push (asks), create PR (asks)
lazycommit pr --publish --draft --base develop
```

The description has Summary, Motivation (only when a commit message explains
one), Implementation details, Testing, Known limitations (only when a commit or
added line documents one), and Commits. Model output is checked: statements
about tests, about files outside the diff, and about files the model was not
shown are removed. The Testing section is always written by Lazy-Commit itself
and says it did not run tests. If the model is unavailable or its output is
unusable, a description built only from commit messages and Git data is shown
(and offered in `--publish`).

`--publish` flow: check `gh` login -> scan every commit not yet on the remote
(messages, added files, and per-commit patches, so a secret added and later
removed is still caught) -> check the remote branch (never publishes over
diverged history) -> stop if an open PR already exists (and offer to push new
commits to it) -> show the full title/description -> `[c]ontinue / edit
[t]itle / edit [b]ody / [q]uit` -> type `push` -> type `create` -> report the
PR URL GitHub returned.

### Changelog

```sh
lazycommit changelog                          # preview section from this branch's commits
lazycommit changelog --since v1.2.0 --title v1.3.0
lazycommit changelog --write                  # insert into CHANGELOG.md after typing `write`
```

Conventional Commit types become confirmed categories; other commits are listed
under "Unclassified (please review)" with a clearly labelled guess. The file is
re-checked before writing and is never staged or committed.

### Demo

```sh
python scripts/make_demo_repo.py            # prints a temp repo path with fake secrets
lazycommit --repo <that path> --model qwen2.5-coder:7b
```

## How it works

1. **Inspect** (`gitrepo.py`): read-only Git calls only (`status`, `diff`,
   `rev-parse`, `ls-files`), with `--no-optional-locks`, `--no-ext-diff`,
   `--no-textconv`, fsmonitor disabled and no shell. Staged, unstaged, and
   untracked changes are kept separate; ignored files are never listed.
   Merge/rebase/cherry-pick in progress or unresolved conflicts stop the run
   before anything is sent to the model.
2. **Protect** (`secrets.py`, `collect.py`): sensitive filenames (`.env*`,
   keys, keystores, credential files...) are excluded without being read.
   Every other file's *full* diff is scanned for credential patterns before
   truncation; any hit excludes the whole file. The assembled prompt is
   scanned once more and the request is aborted if anything matches.
   Reports show rule names and line numbers, never values.
3. **Ask** (`ollama.py`, `planner.py`): the sanitized, size-bounded diff is
   sent to the local Ollama `/api/chat` with a JSON schema. Non-loopback hosts
   and proxies are refused; there is no cloud fallback.
4. **Validate** (`planner.py`): output must parse as JSON and match the
   schema. Groups that reference files not in the candidate list (including
   excluded files), have invalid types, or lack a rationale are rejected.
   Files in several groups or in no group go to "needs manual review".
5. **Preview** (`render.py`): a clearly labeled proposal.
6. **Review and commit** (`review.py`, `execute.py`, `--review` only):
   - When the plan is generated, a snapshot records the branch, HEAD, every
     `git status` record (including index blob ids) and a SHA-256 of every
     changed file's working-tree bytes.
   - Before *each* commit the repository must match that snapshot exactly
     (minus groups already committed). Any difference, a merge/rebase/
     cherry-pick in progress, conflicts, a detached HEAD, or an `index.lock`
     stops execution and asks you to re-plan.
   - Each group is committed with `git commit --only -m <message> -- <paths>`,
     which builds the commit from HEAD plus exactly those paths and leaves
     every other index entry (your pre-existing staged work) untouched. New
     files are first marked with `git add --intent-to-add -- <path>`; if the
     commit fails, that marker is removed again with `git rm --cached`.
     These three are the only write commands Lazy-Commit can run; all use
     explicit literal paths, argument lists, and no shell.
   - After each commit, Lazy-Commit verifies that the new commit's only parent
     is the previous HEAD, that it changed exactly the approved paths, and
     that everything else is unchanged. Any failure or mismatch stops all
     later groups. Successful commits are never undone automatically; the
     report shows how to undo them yourself (`git reset --soft <old HEAD>`).

7. **Pull requests** (`branch.py`, `prwriter.py`, `github.py`, `prcli.py`):
   branch, remote, base, and commits come from read-only Git (`log`,
   `rev-list`, `ls-remote`, `config --get` ...). The only push is
   `git push --set-upstream <remote> refs/heads/<b>:refs/heads/<b>` with no
   force, tags, or submodules. `gh` is used only for `auth status`, `pr list`,
   and `pr create`; its auth output is discarded and tokens are never handled.

## Tests

```sh
pytest          # temp repos; Phase 1/2 use a local fake Ollama server, Phase 3 uses real git + a real bare remote
mypy            # strict type checks
set LAZYCOMMIT_TEST_MODEL=qwen2.5-coder:3b
pytest -m ollama            # real local model
set LAZYCOMMIT_GITHUB_SANDBOX=sniperchief/lazycommit-sandbox
pytest -m github            # REAL pushes and PRs in a throwaway repo, closed afterwards
```

## Limitations

- Secret detection is pattern-based and best-effort; it will miss secrets in
  unusual formats.
- Grouping is per file: a file whose hunks belong to different commits is
  proposed as a single unit, and files with both staged and unstaged changes
  are never committed by Lazy-Commit.
- Your repository's commit hooks (pre-commit, commit-msg, ...) run as with any
  `git commit`. A hook that edits or stages extra files is detected after the
  commit and stops execution, but the commit it produced is not undone.
- Commit content goes through Git's normal clean filters (e.g. line-ending
  conversion, LFS), as with any `git commit`.
- Ollama must run on loopback; a model on another LAN machine is refused.
- Pull requests are created in the remote's own repository (no fork-to-upstream
  PRs yet), on github.com only.
- Binary files in outgoing commits cannot be secret-scanned; they are listed as
  warnings, not blocked.
- Detecting invented content is heuristic: statements about tests, unknown
  files, and withheld files are removed, but a model can still misdescribe
  code it was shown. Always read the description before publishing.
