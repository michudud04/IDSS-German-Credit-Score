"""Write the officer summary and the customer letter.

For each text: build the allowed facts -> ask the LLM -> check the text with
the guard -> if it fails, ask once more with the list of problems -> if it
still fails (or no LLM is available), use the template. The result records
which path was taken, so the app can show it and the report can measure it.
"""

from __future__ import annotations

import urllib.error
from dataclasses import asdict, dataclass, field
from typing import Literal

from Source.decision.narrative import guard, prompts, templates
from Source.decision.narrative.backends import TextBackend, default_backend
from Source.decision.narrative.facts import customer_facts, officer_facts
from Source.decision.types import Decision
from Source.explain.types import AdviceItem, KeyFactor, SimilarCases, UnusualCheck

MAX_ATTEMPTS = 2


@dataclass
class Narrative:
    text: str
    source: Literal["llm", "template"]
    backend: str | None                              # e.g. "ollama:llama3.2:3b"; None = no LLM configured
    attempts: int = 0                                 # LLM calls made
    issues: list[str] = field(default_factory=list)   # guard problems of the rejected drafts

    def to_dict(self) -> dict:
        return asdict(self)


_UNSET = object()


class NarrativeWriter:
    def __init__(self, backend: TextBackend | None | object = _UNSET):
        # No argument: use the configured backend if reachable. backend=None: templates only.
        self.backend = default_backend() if backend is _UNSET else backend

    def _write(self, system: str, facts: dict, check, fallback: str) -> Narrative:
        if self.backend is None:
            return Narrative(fallback, "template", None)
        issues: list[str] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                text = self.backend.generate(system, prompts.user_message(facts, issues or None))
            except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
                issues.append(f"LLM call failed: {exc}")
                break
            problems = check(text)
            if not problems:
                return Narrative(text, "llm", self.backend.name, attempt, issues)
            issues = problems
        return Narrative(fallback, "template", self.backend.name, attempt, issues)

    def officer_summary(self, decision: Decision, key_factors: list[KeyFactor] | None = None,
                        advice: list[AdviceItem] | None = None, similar: SimilarCases | None = None,
                        unusual: UnusualCheck | None = None, inputs: dict | None = None) -> Narrative:
        facts = officer_facts(decision, key_factors, advice, similar, unusual, inputs)
        return self._write(prompts.officer_system(facts), facts,
                           lambda text: guard.check_officer_summary(text, facts),
                           templates.officer_summary(facts))

    def customer_letter(self, decision: Decision, key_factors: list[KeyFactor] | None = None,
                        advice: list[AdviceItem] | None = None) -> Narrative:
        facts = customer_facts(decision, key_factors, advice)
        return self._write(prompts.CUSTOMER_SYSTEM, facts,
                           lambda text: guard.check_customer_letter(text, facts, decision.outcome),
                           templates.customer_letter(facts, decision.outcome))
