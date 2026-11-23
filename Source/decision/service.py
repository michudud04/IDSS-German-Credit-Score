"""Entry point for the app (Person E): ``evaluate_decision``.

The orchestrator in ``Source/app/orchestrator.py`` already computes the
probabilities, the explanations and the unusual-case check, then calls

    Source.decision.evaluate_decision(inputs=..., readable_row=..., probabilities=...,
                                      risk_cutoff=..., review_margin=..., key_factors=...,
                                      advice=..., similar_cases=..., unusual=...)

and reads the keys ``decision``, ``decision_reasons``, ``bank_rules_triggered``,
``officer_summary`` and ``customer_letter`` from the returned dict. Extra keys
give the full ``Decision``, the policy and where each text came from.

Slider: pass either ``cost_ratio`` (preferred, see policy.py) or ``risk_cutoff``.
``review_margin`` follows the app's convention: an absolute half-width around
the cut-off (0.05 = +-5 points); None uses the default grey zone.
"""

from __future__ import annotations

import pandas as pd

from Source.decision.engine import decide
from Source.decision.narrative.writer import NarrativeWriter
from Source.decision.policy import MAX_COST_RATIO, MIN_COST_RATIO, RiskPolicy, cost_ratio_from_cutoff
from Source.decision.rules import check_rules
from Source.explain.types import AdviceItem, KeyFactor, SimilarCases, UnusualCheck

_WRITER: NarrativeWriter | None = None


def get_writer() -> NarrativeWriter:
    """One writer per process; the LLM backend is detected once."""
    global _WRITER
    if _WRITER is None:
        _WRITER = NarrativeWriter()
    return _WRITER


def make_policy(cost_ratio: float | None = None, risk_cutoff: float | None = None,
                review_margin: float | None = None) -> RiskPolicy:
    """Policy from the slider value (cost ratio or cut-off), clamped to the slider's range."""
    if cost_ratio is None:
        cost_ratio = cost_ratio_from_cutoff(risk_cutoff) if risk_cutoff is not None else RiskPolicy().cost_ratio
    cost_ratio = min(max(cost_ratio, MIN_COST_RATIO), MAX_COST_RATIO)
    if review_margin is None:
        return RiskPolicy(cost_ratio)
    cutoff = RiskPolicy(cost_ratio).cutoff
    return RiskPolicy(cost_ratio, review_margin=review_margin / cutoff)  # upper edge = cutoff + margin


def evaluate_decision(inputs: dict, probabilities: dict[str, float], unusual: UnusualCheck | None = None,
                      key_factors: list[KeyFactor] | None = None, advice: list[AdviceItem] | None = None,
                      similar_cases: SimilarCases | None = None, risk_cutoff: float | None = None,
                      review_margin: float | None = None, cost_ratio: float | None = None,
                      readable_row: dict | None = None, write_texts: bool = True,
                      writer: NarrativeWriter | None = None) -> dict:
    """Decision and texts for ONE applicant. ``inputs`` are the form values (blanks as NaN)."""
    policy = make_policy(cost_ratio, risk_cutoff, review_margin)
    decision = decide(probabilities, policy, unusual, check_rules(pd.Series(inputs, dtype=float)))

    result = {
        "decision": decision.outcome,
        "decision_reasons": [r.officer_text for r in decision.reasons],
        "bank_rules_triggered": [r.code for r in decision.reasons if r.source == "bank_rule"],
        "decision_details": decision,
        "policy": policy,
        "officer_summary": "",
        "customer_letter": "",
    }
    if write_texts:
        writer = writer or get_writer()
        summary = writer.officer_summary(decision, key_factors, advice, similar_cases, unusual, inputs)
        letter = writer.customer_letter(decision, key_factors, advice)
        result |= {"officer_summary": summary.text, "customer_letter": letter.text,
                   "officer_summary_meta": summary, "customer_letter_meta": letter}
    return result
