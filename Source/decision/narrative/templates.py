"""Deterministic texts: used when no LLM is available or when the LLM's text fails the guard.

They use exactly the same facts as the LLM, so the content is identical; only
the wording is less natural.
"""

from __future__ import annotations

from Source.decision.narrative.facts import ADVICE_ENOUGH, ADVICE_NOT_ENOUGH

OUTCOME_SENTENCE = {
    "approve": "We are pleased to inform you that your loan application has been approved.",
    "review": "Your loan application will be reviewed personally by one of our advisors before a decision "
              "is made. No decision has been taken yet.",
    "reject": "After careful assessment, we are unable to approve your loan application at this time.",
}


def _bullets(factors: list[dict]) -> str:
    return "\n".join(f"- {f['factor']}: {f['value']}" for f in factors)


def customer_letter(facts: dict, outcome: str) -> str:
    parts = ["Dear applicant,", OUTCOME_SENTENCE[outcome]]
    if facts.get("reasons"):
        parts.append(" ".join(facts["reasons"]))
    raising = facts.get("factors_raising_risk") or []
    if raising and outcome != "approve":
        parts.append("The points that weighed most on our assessment were:\n" + _bullets(raising))
    favour = facts.get("factors_in_your_favour") or []
    if favour:
        parts.append("Points in your favour:\n" + _bullets(favour))
    enough = facts.get(ADVICE_ENOUGH) or []
    if enough:
        target = "your application" if outcome == "review" else "a future application"
        intro = (f"Based on our current assessment, any one of the following changes could bring {target} "
                 "within our lending criteria (an application is always assessed in full):")
        parts.append(intro + "\n" + "\n".join(f"- {change}" for change in enough))
    not_enough = facts.get(ADVICE_NOT_ENOUGH) or []
    if not_enough:
        intro = ("The following changes would lower the assessed risk, but would not be enough on their own"
                 + (":" if not enough else ", unlike the changes above:"))
        parts.append(intro + "\n" + "\n".join(f"- {change}" for change in not_enough))
    parts.append("If you have questions or wish to provide additional information, please contact your "
                 "branch.\n\nKind regards,\nThe Credit Team")
    return "\n\n".join(parts)


def officer_summary(facts: dict) -> str:
    lines = [
        f"Recommendation: {facts['recommendation']} (probability of default {facts['probability_of_default']}, "
        f"cut-off {facts['cut_off']}, cost ratio {facts['cost_ratio']}; automatic approval below "
        f"{facts['automatic_approval_below']}, automatic rejection from {facts['automatic_rejection_from']}).",
        "",
        "Why:",
        *[f"- {r}" for r in facts["reasons"]],
    ]
    if facts.get("key_factors"):
        lines += ["", "Key factors (effect on the probability of default):"]
        for f in facts["key_factors"]:
            note = " - sensitive, never use for advice" if f["sensitive_do_not_advise"] else ""
            lines.append(f"- {f['factor']}: {f['value']} ({f['effect']}){note}")
    if facts.get("similar_past_applicants"):
        s = facts["similar_past_applicants"]
        lines += ["", f"Similar past applicants: {s['default_rate_among_most_similar']} of the "
                      f"{s['number_compared']} most similar defaulted (all applicants: "
                      f"{s['default_rate_all_applicants']}); {s['closest_who_defaulted']} of the "
                      f"{s['closest_shown']} closest shown defaulted."]
    if facts.get("unusual_case_warning"):
        lines += ["", "Unusual-case warning:", *[f"- {r}" for r in facts["unusual_case_warning"]]]
    if facts.get("what_if_advice"):
        lines += ["", "What-if advice for the customer:"]
        for a in facts["what_if_advice"]:
            ok = "would be approved automatically" if a["would_be_approved_automatically"] else "not enough alone"
            lines.append(f"- {a['change']}: probability {a['new_probability']} ({ok})")
    lines += ["", "This is a recommendation; the final decision rests with the officer."]
    return "\n".join(lines)
