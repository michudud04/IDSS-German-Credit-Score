"""Data class holding all results shown in the app for one applicant."""

from dataclasses import asdict, dataclass

from Source.explain.types import AdviceItem, KeyFactor, SimilarCases, UnusualCheck


@dataclass
class CaseResult:
    # Preprocessed features (Part A)
    inputs: dict
    prepared_readable: dict

    # Model predictions (Part B)
    probabilities: dict
    combined_probability: float
    model_disagreement: float

    # Explanations and neighbour checks (Part C)
    key_factors: list[KeyFactor]
    advice: list[AdviceItem]
    similar_cases: SimilarCases
    unusual: UnusualCheck
    is_unusual: bool

    # Final decision and texts (Part D)
    risk_cutoff: float
    review_margin: float
    decision: str
    decision_reasons: list[str]
    officer_summary: str
    customer_letter: str
    policy_summary: dict

    def to_dict(self) -> dict:
        return asdict(self)
