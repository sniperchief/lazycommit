"""Minimal client for a *local* Ollama server.

Only loopback hosts are accepted so repository content can never be sent to
another machine. There is no cloud fallback.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from lazycommit.secrets import redact

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 180.0


class ModelError(Exception):
    """A problem talking to the local model, with an actionable hint.

    ``category`` is a fixed, non-sensitive label used for telemetry.
    """

    def __init__(self, message: str, hint: str = "", category: str = "other") -> None:
        super().__init__(message)
        self.hint = hint
        self.category = category


def normalize_host(host: str) -> str:
    """Normalize OLLAMA_HOST-style values and enforce loopback."""
    host = host.strip()
    if "://" not in host:
        host = "http://" + host
    parsed = urllib.parse.urlparse(host)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ModelError(f"invalid Ollama host: {host!r}", "Use a URL like http://127.0.0.1:11434",
                         "model_refused_host")
    name = parsed.hostname
    loopback = name == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(name).is_loopback
        except ValueError:
            loopback = False
    if not loopback:
        raise ModelError(
            f"refusing non-local Ollama host {name!r}",
            "Lazy-Commit only sends repository content to a model on this machine "
            "(localhost / 127.0.0.1 / ::1).",
            "model_refused_host",
        )
    port = parsed.port or 11434
    netloc = f"[{name}]:{port}" if ":" in name else f"{name}:{port}"
    return f"{parsed.scheme}://{netloc}"


@dataclass
class OllamaClient:
    host: str
    timeout: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        self.host = normalize_host(self.host)

    def _request(self, method: str, path: str, body: dict[str, Any] | None, timeout: float) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.host + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        # Bypass any HTTP(S)_PROXY settings: traffic must stay on loopback.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            try:
                detail = str(json.loads(detail).get("error", detail))
            except (ValueError, AttributeError):
                pass
            raise ModelError(f"Ollama returned HTTP {exc.code}: {redact(detail)}",
                             category="model_http_error") from None
        except (TimeoutError, socket.timeout):
            raise ModelError(
                f"Ollama did not respond within {timeout:.0f}s",
                "Try a smaller/faster model, raise --timeout, or lower --max-total-chars.",
                "model_timeout",
            ) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ModelError(
                    f"Ollama did not respond within {timeout:.0f}s",
                    "Try a smaller/faster model, raise --timeout, or lower --max-total-chars.",
                    "model_timeout",
                ) from None
            raise ModelError(
                f"cannot reach Ollama at {self.host} ({exc.reason})",
                "Install Ollama from https://ollama.com/download and start it "
                "(`ollama serve`, or launch the Ollama app). Set OLLAMA_HOST if it "
                "listens on a different local port.",
                "model_unavailable",
            ) from None
        except (ConnectionError, OSError) as exc:
            raise ModelError(
                f"connection to Ollama at {self.host} failed or was interrupted ({type(exc).__name__})",
                "Check that Ollama is still running and retry.",
                "model_unavailable",
            ) from None
        try:
            return json.loads(raw)
        except ValueError:
            raise ModelError("Ollama returned a non-JSON HTTP response", category="model_bad_response") from None

    def installed_models(self) -> list[str]:
        data = self._request("GET", "/api/tags", None, timeout=min(self.timeout, 10.0))
        models = data.get("models", []) if isinstance(data, dict) else []
        return [m.get("name", "") for m in models if isinstance(m, dict) and m.get("name")]

    def ensure_model(self, model: str) -> None:
        installed = self.installed_models()
        names = set(installed)
        # "llama3.2" is shorthand for "llama3.2:latest".
        if model in names or (":" not in model and f"{model}:latest" in names):
            return
        listing = ", ".join(sorted(installed)) or "none"
        raise ModelError(
            f"model {model!r} is not installed in Ollama (installed: {listing})",
            f"Run `ollama pull {model}`, or pick an installed model with --model / LAZYCOMMIT_MODEL.",
            "model_not_installed",
        )

    def chat_json(self, model: str, system: str, user: str, schema: dict[str, Any]) -> str:
        body = {
            "model": model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = self._request("POST", "/api/chat", body, timeout=self.timeout)
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError):
            raise ModelError("Ollama response did not contain a chat message",
                             category="model_bad_response") from None
        if not isinstance(content, str):
            raise ModelError("Ollama response message content was not text", category="model_bad_response")
        return content
