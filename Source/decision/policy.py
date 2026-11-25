"""Risk slider: from a business cost ratio to a probability cut-off.

The officer (or the bank) sets one number with the slider:

    cost_ratio = cost of approving a customer who then defaults
                 / cost of rejecting a customer who would have repaid

With calibrated probabilities, approving costs ``p * cost_ratio`` on average
and rejecting costs ``(1 - p) * 1`` (in units of one lost good customer).
Approving is cheaper exactly when

    p < 1 / (1 + cost_ratio)          (the cut-off)

so the slider never needs a separate threshold. This only holds because the
models' probabilities are calibrated (no class weights, Brier 0.049 on test;
see notebooks/model_training.ipynb, sections 2.3 and 5).

Close to the cut-off the two actions cost almost the same and a small error in
the probability flips the decision, so a grey zone around it goes to review:

    p < cutoff / (1 + margin)               -> approve
    p >= cutoff * (1 + margin)              -> reject
    in between                              -> review

The default cost ratio 4 gives a cut-off of 20%, the illustrative threshold
used in the model evaluation, so the numbers in the models notebook stay valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

DEFAULT_COST_RATIO = 4.0     # cut-off 20%
MIN_COST_RATIO = 1.0         # cut-off 50%: a bad loan costs as much as a lost customer
MAX_COST_RATIO = 20.0        # cut-off ~4.8%
DEFAULT_REVIEW_MARGIN = 0.25  # grey zone: within 25% of the cut-off (16%-25% at the default)


def cutoff_from_cost_ratio(cost_ratio: float) -> float:
    """Probability of default above which rejecting is cheaper than approving."""
    if cost_ratio <= 0:
        raise ValueError("cost_ratio must be positive")
    return 1.0 / (1.0 + cost_ratio)


def cost_ratio_from_cutoff(cutoff: float) -> float:
    """Inverse of ``cutoff_from_cost_ratio`` (e.g. 0.20 -> 4)."""
    if not 0 < cutoff < 1:
        raise ValueError("cutoff must be between 0 and 1")
    return (1.0 - cutoff) / cutoff


@dataclass(frozen=True)
class RiskPolicy:
    """The slider setting and the probability bands it implies."""

    cost_ratio: float = DEFAULT_COST_RATIO
    review_margin: float = DEFAULT_REVIEW_MARGIN

    def __post_init__(self):
        if not MIN_COST_RATIO <= self.cost_ratio <= MAX_COST_RATIO:
            raise ValueError(f"cost_ratio must be in [{MIN_COST_RATIO}, {MAX_COST_RATIO}]")
        if self.review_margin < 0:
            raise ValueError("review_margin must be >= 0")

    @classmethod
    def from_cutoff(cls, cutoff: float, review_margin: float = DEFAULT_REVIEW_MARGIN) -> "RiskPolicy":
        return cls(cost_ratio_from_cutoff(cutoff), review_margin)

    @property
    def cutoff(self) -> float:
        return cutoff_from_cost_ratio(self.cost_ratio)

    @property
    def approve_below(self) -> float:
        """Probabilities below this are approved automatically (lower edge of the grey zone)."""
        return self.cutoff / (1.0 + self.review_margin)

    @property
    def reject_from(self) -> float:
        """Probabilities from this value on are rejected automatically (upper edge of the grey zone)."""
        return min(self.cutoff * (1.0 + self.review_margin), 1.0)

    def band(self, probability: float) -> str:
        """'approve', 'review' or 'reject' from the probability alone."""
        if probability < self.approve_below:
            return "approve"
        if probability >= self.reject_from:
            return "reject"
        return "review"

    def passes(self, probability: float) -> bool:
        """Would this probability be approved automatically? (for the advice's ``passes_cutoff``)"""
        return probability < self.approve_below

    def expected_costs(self, probability: float) -> dict[str, float]:
        """Expected cost of each action, in units of one lost good customer."""
        return {"approve": probability * self.cost_ratio, "reject": 1.0 - probability}

    def to_dict(self) -> dict:
        return asdict(self) | {"cutoff": self.cutoff, "approve_below": self.approve_below,
                               "reject_from": self.reject_from}
