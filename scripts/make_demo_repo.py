"""Create a throwaway Git repo with messy changes for demonstrating Lazy-Commit.

Usage: python scripts/make_demo_repo.py [TARGET_DIR]

All credentials written here are fake. The target directory must not exist.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()
        if root.exists():
            print(f"refusing to use existing path: {root}", file=sys.stderr)
            return 1
        root.mkdir(parents=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="lazycommit-demo-"))

    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Demo")
    git(root, "config", "user.email", "demo@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    write(root, "README.md", "# Weather CLI\n\nPrints the weather.\n")
    write(root, "weather/client.py", (
        "import urllib.request\n\n\n"
        "def fetch(city):\n"
        "    return urllib.request.urlopen(f'https://wttr.in/{city}').read()\n"
    ))
    write(root, "weather/cli.py", "import sys\nfrom weather.client import fetch\n\nprint(fetch(sys.argv[1]))\n")
    write(root, ".gitignore", "__pycache__/\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")

    # Staged: a bug fix with retry logic.
    write(root, "weather/client.py", (
        "import time\nimport urllib.request\n\n\n"
        "def fetch(city, retries=3):\n"
        "    for attempt in range(retries):\n"
        "        try:\n"
        "            return urllib.request.urlopen(f'https://wttr.in/{city}', timeout=10).read()\n"
        "        except OSError:\n"
        "            if attempt == retries - 1:\n"
        "                raise\n"
        "            time.sleep(2 ** attempt)\n"
    ))
    git(root, "add", "weather/client.py")
    # Unstaged: CLI usage message + docs.
    write(root, "weather/cli.py", (
        "import sys\nfrom weather.client import fetch\n\n"
        "if len(sys.argv) != 2:\n"
        "    sys.exit('usage: weather CITY')\n"
        "print(fetch(sys.argv[1]).decode())\n"
    ))
    write(root, "README.md", "# Weather CLI\n\nPrints the weather.\n\n## Usage\n\n    python -m weather.cli London\n")
    # Untracked: new tests.
    write(root, "tests/test_client.py", (
        "from unittest import mock\nfrom weather import client\n\n\n"
        "def test_fetch_retries():\n"
        "    with mock.patch('urllib.request.urlopen', side_effect=[OSError, mock.Mock(read=lambda: b'ok')]):\n"
        "        with mock.patch('time.sleep'):\n"
        "            assert client.fetch('x') == b'ok'\n"
    ))
    # Sensitive: these must be excluded from the model.
    write(root, ".env", "WEATHER_API_KEY=fake-demo-value-do-not-use\n")
    write(root, "weather/settings.py", 'API_TOKEN = "ghp_' + "DemoFakeToken0123456789DemoFakeTok" + '"\n')

    print(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
