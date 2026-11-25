"""Text generators behind one interface: a local Ollama model or an OpenAI-compatible API.

Configuration with environment variables (no code change needed):

    IDSS_LLM_BACKEND   ollama (default) | openai | template (= no LLM)
    IDSS_OLLAMA_MODEL  default llama3.2:3b
    OLLAMA_HOST        default http://localhost:11434
    IDSS_OPENAI_MODEL  default gpt-4o-mini; IDSS_OPENAI_BASE_URL default https://api.openai.com/v1
    OPENAI_API_KEY     required for the openai backend

Ollama is the default because the customer's financial data then never
leaves the bank's machine. If no backend is reachable, ``default_backend``
returns None and the writer uses the deterministic templates instead.
Only the standard library is used, so there is no new dependency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class TextBackend(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> str:
        ...


def _post_json(url: str, payload: dict, timeout: float, headers: dict | None = None) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class OllamaBackend:
    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 120.0,
                 temperature: float = 0.2):
        self.model = model or os.environ.get("IDSS_OLLAMA_MODEL", "llama3.2:3b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"ollama:{self.model}"

    def is_available(self) -> bool:
        """Server reachable and the model downloaded (`ollama pull <model>`)."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as response:
                names = [m["name"] for m in json.loads(response.read()).get("models", [])]
        except (urllib.error.URLError, OSError, ValueError):
            return False
        return any(n == self.model or n.split(":")[0] == self.model for n in names)

    def generate(self, system: str, prompt: str) -> str:
        reply = _post_json(f"{self.host}/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": 500},  # cap length (and time)
        }, self.timeout)
        return reply["message"]["content"].strip()


class OpenAICompatibleBackend:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 60.0, temperature: float = 0.2):
        self.model = model or os.environ.get("IDSS_OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("IDSS_OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.temperature = temperature
        self.name = f"openai:{self.model}"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, system: str, prompt: str) -> str:
        reply = _post_json(f"{self.base_url}/chat/completions", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }, self.timeout, headers={"Authorization": f"Bearer {self.api_key}"})
        return reply["choices"][0]["message"]["content"].strip()


def default_backend() -> TextBackend | None:
    """The configured backend if it is reachable, else None (templates only)."""
    choice = os.environ.get("IDSS_LLM_BACKEND", "ollama").lower()
    if choice == "template":
        return None
    backend = OpenAICompatibleBackend() if choice == "openai" else OllamaBackend()
    return backend if backend.is_available() else None
