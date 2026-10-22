"""Best-effort detection of secrets and sensitive files.

Findings carry a rule name and line number only, never the matched value, so
they are always safe to print or log. Detection is heuristic: a clean scan is
not a guarantee that content is secret-free.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Finding:
    rule: str
    line: int | None = None

    def describe(self) -> str:
        return f"{self.rule} (line {self.line})" if self.line else self.rule


# Filenames (matched case-insensitively against the basename) that are
# excluded from AI analysis regardless of content.
SENSITIVE_NAME_PATTERNS: tuple[tuple[str, str], ...] = (
    (".env", "environment file"),
    (".env.*", "environment file"),
    ("*.env", "environment file"),
    ("*.pem", "PEM key/certificate file"),
    ("*.key", "private key file"),
    ("*.p12", "PKCS#12 keystore"),
    ("*.pfx", "PKCS#12 keystore"),
    ("*.jks", "Java keystore"),
    ("*.keystore", "keystore file"),
    ("*.kdbx", "password database"),
    ("*.ppk", "PuTTY private key"),
    ("id_rsa", "SSH private key"),
    ("id_dsa", "SSH private key"),
    ("id_ecdsa", "SSH private key"),
    ("id_ed25519", "SSH private key"),
    ("credentials", "credentials file"),
    ("credentials.*", "credentials file"),
    ("*credentials*.json", "credentials file"),
    ("service-account*.json", "cloud service-account key"),
    ("secrets.yml", "secrets file"),
    ("secrets.yaml", "secrets file"),
    ("secrets.json", "secrets file"),
    (".netrc", "netrc credentials"),
    ("_netrc", "netrc credentials"),
    (".npmrc", "npm config (may hold tokens)"),
    (".pypirc", "PyPI config (may hold tokens)"),
    (".git-credentials", "git credential store"),
    (".htpasswd", "htpasswd file"),
)

# Content rules. Patterns are deliberately specific to keep false positives
# manageable; the generic assignment rule requires a quoted literal.
CONTENT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY(?: BLOCK)?-----")),
    ("AWS access key ID", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b")),
    (
        "AWS secret access key",
        re.compile(r"(?i)aws.{0,20}?(?:secret|key).{0,20}?[:=]\s*['\"]?[A-Za-z0-9/+=]{40}\b"),
    ),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("GitLab token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}")),
    ("Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("Slack webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+")),
    ("Stripe key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI-style API key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("npm token", re.compile(r"\bnpm_[A-Za-z0-9]{36}")),
    ("JSON Web Token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("credentials in URL", re.compile(r"[a-z][a-z0-9+.\-]*://[^\s/:@'\"]+:[^\s/@'\"]{3,}@", re.I)),
    (
        "hard-coded secret assignment",
        re.compile(
            r"(?i)\b[\w.\-]*(?:api[_\-]?key|secret|token|passwd|password|pwd|access[_\-]?key|auth[_\-]?key|client[_\-]?secret)[\w.\-]*"
            r"\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
        ),
    ),
)


def sensitive_filename(path: str) -> str | None:
    """Return a reason if the path names a sensitive file, else None."""
    name = PurePosixPath(path.replace("\\", "/")).name.lower()
    if name.endswith(".pub"):
        return None
    for pattern, reason in SENSITIVE_NAME_PATTERNS:
        if fnmatch.fnmatchcase(name, pattern):
            return reason
    return None


def scan_text(text: str) -> list[Finding]:
    """Scan text for secret patterns. Returns at most one finding per rule per line."""
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in CONTENT_RULES:
            if pattern.search(line):
                findings.append(Finding(rule, lineno))
    return findings


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for _, pattern in CONTENT_RULES)


def redact(text: str) -> str:
    """Replace anything matching a secret rule. Used for diagnostics only."""
    for _, pattern in CONTENT_RULES:
        text = pattern.sub("[REDACTED]", text)
    return text
