from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from Source.explain.types import AdviceItem, KeyFactor, SimilarCases, UnusualCheck

DecisionLabel = Literal["approve", "review", "reject"]


@dataclass
class CaseResult:
    inputs: dict[str, float | None]
    prepared_readable: dict[str, float]

    probabilities: dict[str, float]
    combined_probability: float
    model_disagreement: float

    # Person C types from Source/explain/types.py
    key_factors: list[KeyFactor]
    advice: list[AdviceItem]
    similar_cases: SimilarCases
    unusual: UnusualCheck
    is_unusual: bool

    # Person D fields
    risk_cutoff: float
    review_margin: float
    decision: DecisionLabel
    decision_reasons: list[str]
    bank_rules_triggered: list[str]
    officer_summary: str
    customer_letter: str

    def to_dict(self) -> dict:
        return asdict(self)
