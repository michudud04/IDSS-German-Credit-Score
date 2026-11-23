"""Typed results of the decision step (Person D).

The overall case result (Person E) should hold a ``Decision`` in its
``decision`` field. Every outcome comes with machine-readable reason codes
and readable texts, so the app, the LLM and the report all use the same reasons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Outcome = Literal["approve", "review", "reject"]
Source_ = Literal["risk", "grey_zone", "model_disagreement", "unusual_case", "bank_rule"]


@dataclass
class Reason:
    code: str                 # stable identifier, e.g. "RULE_DEBT_EXCEEDS_INCOME"
    source: Source_           # which part of the decision logic produced it
    effect: Outcome           # what this reason alone asks for
    officer_text: str         # full explanation for the officer (may contain internal details)
    customer_text: str | None = None  # wording safe for the customer letter; None = never show the customer

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    outcome: Outcome
    probability: float                  # combined probability of default
    per_model: dict[str, float]         # probability of each model
    policy: dict                        # RiskPolicy.to_dict(): cost ratio, cut-off, grey zone
    expected_costs: dict[str, float]    # expected cost of approving / rejecting
    reasons: list[Reason] = field(default_factory=list)  # all reasons, the decisive ones first

    @property
    def decisive_reasons(self) -> list[Reason]:
        """Reasons whose effect equals the outcome."""
        return [r for r in self.reasons if r.effect == self.outcome]

    @property
    def reason_codes(self) -> list[str]:
        return [r.code for r in self.reasons]

    def to_dict(self) -> dict:
        return asdict(self) | {"reason_codes": self.reason_codes}
