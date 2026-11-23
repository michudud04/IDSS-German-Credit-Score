"""The facts each text may use. Filtering happens HERE, before any LLM sees the data.

Officer facts: everything that helps the officer decide (all reasons, key
factors including sensitive ones, similar cases, the unusual-case warning).

Customer facts: only what the applicant may be told:
- reasons that have a ``customer_text`` (internal ones like model disagreement have none);
- key factors that are not sensitive (never age or dependents), as the main reasons;
- advice that only changes actionable, non-sensitive fields;
- never similar cases (they hold other customers' data), never probabilities.

Because the LLM never receives forbidden data, it cannot leak it; the guard
(guard.py) is a second line of defence against invented content.
"""

from __future__ import annotations

from Source.decision.types import Decision
from Source.explain.features import FIELDS, format_value
from Source.explain.types import AdviceItem, KeyFactor, SimilarCases, UnusualCheck

MAX_FACTORS = 5
MAX_CUSTOMER_FACTORS = 3
MAX_ADVICE = 3


def _pct(p: float) -> str:
    return f"{p:.1%}"


def _points(x: float) -> str:
    return f"{x * 100:+.1f} points"


def officer_facts(decision: Decision, key_factors: list[KeyFactor] | None = None,
                  advice: list[AdviceItem] | None = None, similar: SimilarCases | None = None,
                  unusual: UnusualCheck | None = None, inputs: dict | None = None) -> dict:
    policy = decision.policy
    facts = {
        "recommendation": decision.outcome.upper(),
        "probability_of_default": _pct(decision.probability),
        "cut_off": _pct(policy["cutoff"]),
        "automatic_approval_below": _pct(policy["approve_below"]),
        "automatic_rejection_from": _pct(policy["reject_from"]),
        "cost_ratio": f"{policy['cost_ratio']:g}",
        "per_model_probability": {name: _pct(p) for name, p in decision.per_model.items()},
        "reasons": [r.officer_text for r in decision.reasons],
    }
    # The raw form values are NOT passed: with them, small models add their own judgements
    # ("limited income", "may be a strain"). The values that matter come through the key factors,
    # with their measured effect. The app shows the form values to the officer anyway.
    if key_factors:
        ranked = sorted(key_factors, key=lambda f: abs(f.contribution), reverse=True)[:MAX_FACTORS]
        facts["key_factors"] = [
            {"factor": f.label, "value": f.value_text, "effect": _points(f.contribution),
             "sensitive_do_not_advise": f.sensitive}
            for f in ranked
        ]
    if similar:
        shown = similar.neighbours
        facts["similar_past_applicants"] = {
            "default_rate_among_most_similar": _pct(similar.neighbourhood_default_rate),
            "number_compared": similar.neighbourhood_size,
            "default_rate_all_applicants": _pct(similar.base_rate),
            "closest_shown": len(shown),
            "closest_who_defaulted": sum(c.defaulted for c in shown),
        }
    if unusual and unusual.is_unusual:
        facts["unusual_case_warning"] = unusual.reasons
    if advice:
        facts["what_if_advice"] = [
            {"changes": [c.description for c in a.changes], "new_probability": _pct(a.new_prob),
             "would_be_approved_automatically": a.passes_cutoff}
            for a in advice[:MAX_ADVICE]
        ]
    return facts


def _customer_value(f: KeyFactor) -> str:
    # An estimated income was never stated by the applicant, so do not show the estimate.
    return "not declared" if "estimated" in f.value_text or f.value is None else f.value_text


def _advice_is_allowed(item: AdviceItem) -> bool:
    return all(c.feature in FIELDS and FIELDS[c.feature].actionable and not FIELDS[c.feature].sensitive
               for c in item.changes)


def customer_facts(decision: Decision, key_factors: list[KeyFactor] | None = None,
                   advice: list[AdviceItem] | None = None) -> dict:
    reasons, seen = [], set()
    for r in decision.reasons:
        if r.customer_text and r.customer_text not in seen and (r.effect == decision.outcome or r.source == "bank_rule"):
            reasons.append(r.customer_text)
            seen.add(r.customer_text)

    # The applicant is entitled to the main reasons, actionable or not (e.g. past late payments),
    # but never to sensitive ones. Advice below is restricted further to actionable fields.
    allowed = [f for f in (key_factors or []) if not f.sensitive]
    ranked = sorted(allowed, key=lambda f: abs(f.contribution), reverse=True)
    facts = {
        "outcome": {"approve": "approved", "review": "under personal review by an advisor",
                    "reject": "not approved"}[decision.outcome],
        "reasons": reasons,
        "factors_raising_risk": [{"factor": f.label, "value": _customer_value(f)} for f in ranked
                                 if f.direction == "increases_risk"][:MAX_CUSTOMER_FACTORS],
        "factors_in_your_favour": [{"factor": f.label, "value": _customer_value(f)} for f in ranked
                                   if f.direction == "decreases_risk"][:2],
    }
    if decision.outcome != "approve" and advice:
        usable = [a for a in advice if _advice_is_allowed(a)]
        usable.sort(key=lambda a: (not a.passes_cutoff, a.effort))
        facts["what_could_improve_a_future_application"] = [
            {"changes": [c.description for c in a.changes],
             "enough_for_approval": a.passes_cutoff}
            for a in usable[:MAX_ADVICE]
        ]
    # Empty lists make the model write "no factors were identified..."; leave them out.
    return {key: value for key, value in facts.items() if value}
