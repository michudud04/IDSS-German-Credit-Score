"""Final recommendation: approve, review or reject, with every reason that led to it.

Order of precedence (first match decides the outcome; all reasons are kept):

1. A bank rule that rejects (``rules.py``)                   -> reject
2. Unusual applicant (C's warning): the model is unreliable  -> review
   (models notebook, test set: AUC 0.71 vs 0.86, Brier 0.22 vs 0.05 on flagged
   applicants. Validation, with the extreme_value trigger: 1.7% flagged, 33%
   default vs 6.2%. This applies even when the model would reject: only 18 of
   24,000 validation applicants move from reject to review because of a single
   extreme value, and the officer can still reject them.)
3. Probability in the reject band of the slider              -> reject
   Review rules do not soften a clear model rejection; they only stop
   an automatic approval.
4. Any review reason: grey zone around the cut-off, the models disagree
   about the side of the cut-off, or a bank rule asks for review  -> review
5. Otherwise                                                  -> approve

    engine = DecisionEngine.from_saved()            # once, at start-up
    decision = engine.assess(inputs)[0]             # per applicant (one row per applicant)
"""

from __future__ import annotations

import pandas as pd

from Source.decision.policy import RiskPolicy
from Source.decision.rules import check_rules
from Source.decision.types import Decision, Reason
from Source.explain.pipeline import COMBINED, RiskPipeline
from Source.explain.types import UnusualCheck

# The models disagree if they fall on both sides of the cut-off AND their
# probabilities are this far apart. Validation: 1.4% of applicants at the default setting.
DISAGREEMENT_SPREAD = 0.15


def _pct(p: float) -> str:
    return f"{p:.1%}"


def risk_reason(probability: float, policy: RiskPolicy) -> Reason:
    band = policy.band(probability)
    where = {
        "approve": f"below the automatic-approval limit of {_pct(policy.approve_below)}",
        "review": f"in the grey zone {_pct(policy.approve_below)}-{_pct(policy.reject_from)} around the "
                  f"cut-off of {_pct(policy.cutoff)}",
        "reject": f"above the automatic-rejection limit of {_pct(policy.reject_from)}",
    }[band]
    customer = {
        "approve": "Your overall credit profile meets our lending criteria.",
        "review": "Your application is close to our acceptance limit, so an advisor will look at it personally.",
        "reject": "Based on your overall credit profile, the estimated risk of repayment difficulties is "
                  "higher than we can accept for this loan.",
    }[band]
    code = {"approve": "RISK_LOW", "review": "RISK_GREY_ZONE", "reject": "RISK_HIGH"}[band]
    source = "grey_zone" if band == "review" else "risk"
    return Reason(code, source, band,
                  officer_text=f"Estimated probability of default {_pct(probability)}, {where} "
                               f"(cost ratio {policy.cost_ratio:g}).",
                  customer_text=customer)


def disagreement_reason(per_model: dict[str, float], policy: RiskPolicy) -> Reason | None:
    low, high = min(per_model.values()), max(per_model.values())
    if not (low < policy.cutoff <= high and high - low > DISAGREEMENT_SPREAD):
        return None
    detail = ", ".join(f"{name} {_pct(p)}" for name, p in per_model.items())
    return Reason("MODELS_DISAGREE", "model_disagreement", "review",
                  officer_text=f"The models disagree about this applicant ({detail}): some are below and some "
                               f"above the cut-off of {_pct(policy.cutoff)}.",
                  customer_text=None)  # internal detail, the letter only says that an advisor will review


def unusual_reason(check: UnusualCheck) -> Reason | None:
    if not check.is_unusual:
        return None
    return Reason("UNUSUAL_CASE", "unusual_case", "review",
                  officer_text="Unusual applicant, the model's estimate is unreliable ("
                               + ", ".join(check.triggers) + "). " + " ".join(check.reasons),
                  customer_text="Your profile differs from the applications we usually assess, so an advisor "
                                "will review it personally rather than relying on the automatic assessment.")


def decide(probabilities: dict[str, float], policy: RiskPolicy, unusual: UnusualCheck | None = None,
           rule_reasons: list[Reason] | None = None) -> Decision:
    """Combine probability, unusual-case check and bank rules into one Decision.

    ``probabilities`` holds one entry per model plus ``"combined"`` (one row of ``RiskPipeline.predict``).
    """
    p = float(probabilities[COMBINED])
    per_model = {name: float(v) for name, v in probabilities.items() if name != COMBINED}
    rule_reasons = rule_reasons or []

    risk = risk_reason(p, policy)
    candidates = [r for r in (unusual_reason(unusual) if unusual else None,
                              disagreement_reason(per_model, policy)) if r is not None]
    reasons = rule_reasons + candidates + [risk]

    if any(r.effect == "reject" for r in rule_reasons):
        outcome = "reject"
    elif any(r.code == "UNUSUAL_CASE" for r in candidates):
        outcome = "review"
    elif risk.effect == "reject":
        outcome = "reject"
    elif any(r.effect == "review" for r in reasons):
        outcome = "review"
    else:
        outcome = "approve"

    # Decisive reasons first, so the app and the LLM can lead with them.
    reasons.sort(key=lambda r: r.effect != outcome)
    return Decision(outcome=outcome, probability=p, per_model=per_model, policy=policy.to_dict(),
                    expected_costs=policy.expected_costs(p), reasons=reasons)


class DecisionEngine:
    """Runs the models, the unusual-case check and the rules for a table of applicants."""

    def __init__(self, pipeline: RiskPipeline, detector=None):
        self.pipeline = pipeline
        self.detector = detector  # UnusualCaseDetector; None = skip the check (tests only)

    @classmethod
    def from_saved(cls, with_unusual_check: bool = True) -> "DecisionEngine":
        from Source.decision.models import load_risk_pipeline
        detector = None
        if with_unusual_check:
            from Source.explain.unusual import load_detector
            detector = load_detector()
        return cls(load_risk_pipeline(), detector)

    def assess(self, inputs: pd.DataFrame, policy: RiskPolicy | None = None,
               unusual: list[UnusualCheck] | None = None) -> list[Decision]:
        """One Decision per row of ``inputs`` (columns: INPUT_FIELDS, optional LateCodeFlag).

        Pass ``unusual`` if the checks were already computed (e.g. by explain_case) to avoid redoing them.
        """
        policy = policy or RiskPolicy()
        probs = self.pipeline.predict(inputs)
        if unusual is None and self.detector is not None:
            unusual = self.detector.check_inputs(self.pipeline.preprocessor, inputs)
        return [
            decide(probs.iloc[i].to_dict(), policy,
                   unusual[i] if unusual is not None else None,
                   check_rules(inputs.iloc[i]))
            for i in range(len(inputs))
        ]
