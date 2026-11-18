"""Typed results of the explanation modules.

These are the parts of the shared case result owned by Person C. The overall
``CaseResult`` (Person E) should hold them in these fields:

    key_factors     -> list[KeyFactor]
    advice          -> list[AdviceItem]
    similar_cases   -> SimilarCases
    is_unusual      -> UnusualCheck.is_unusual (reasons in UnusualCheck.reasons)

All probabilities are probabilities of default (0..1). Contributions are in
probability units: +0.04 means "raised the default probability by 4 points".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class KeyFactor:
    feature: str                    # input field, e.g. "RevUtil"
    label: str                      # readable name, e.g. "Revolving credit utilisation"
    value: float | None             # applicant's value in original units (None = not declared)
    value_text: str                 # formatted value, e.g. "95%" or "$4,100 (estimated)"
    direction: Literal["increases_risk", "decreases_risk"]
    contribution: float             # effect on the combined probability vs. an average applicant
    per_model: dict[str, float]     # the same effect for each model (shows disagreement)
    actionable: bool                # advice may target it
    sensitive: bool                 # age / family situation: never advise on it

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureChange:
    feature: str
    label: str
    old_value: float
    new_value: float
    description: str                # e.g. "Lower utilisation from 95% to 30%"


@dataclass
class AdviceItem:
    changes: list[FeatureChange]    # one or two simultaneous changes
    new_prob: float                 # combined probability after the change(s)
    prob_change: float              # new_prob - current combined probability (negative = better)
    passes_cutoff: bool             # new_prob below the current risk cut-off
    effort: float                   # relative size of the change(s); smaller = easier

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimilarCase:
    case_id: int                    # row index in the training data
    distance: float                 # distance on the model scale (smaller = more similar)
    inputs: dict[str, float]        # the past applicant's values in original units
    defaulted: bool                 # outcome: serious delinquency within 2 years


@dataclass
class SimilarCases:
    neighbours: list[SimilarCase]   # closest past applicants, shown in the UI
    neighbourhood_size: int         # how many neighbours the default rate is based on
    neighbourhood_default_rate: float
    base_rate: float                # default rate of all training applicants

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UnusualCheck:
    is_unusual: bool
    score: float                    # mean distance to the nearest training applicants
    threshold: float                # score above this is unusual (set on the validation split)
    reasons: list[str] = field(default_factory=list)  # readable reasons, e.g. range violations

    def to_dict(self) -> dict:
        return asdict(self)
