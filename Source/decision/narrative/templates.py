"""Deterministic texts: used when no LLM is available or when the LLM's text fails the guard.

They use exactly the same facts as the LLM, so the content is identical; only
the wording is less natural.
"""

from __future__ import annotations

OUTCOME_SENTENCE = {
    "approve": "We are pleased to inform you that your loan application has been approved.",
    "review": "Your loan application will be reviewed personally by one of our advisors before a decision "
              "is made. No decision has been taken yet.",
    "reject": "After careful assessment, we are unable to approve your loan application at this time.",
}


def customer_letter(facts: dict, outcome: str) -> str:
    parts = ["Dear applicant,", OUTCOME_SENTENCE[outcome]]
    if facts.get("reasons"):
        parts.append(" ".join(facts["reasons"]))
    raising = facts.get("factors_raising_risk") or []
    if raising and outcome != "approve":
        listed = "; ".join(f"{f['factor'].lower()} ({f['value']})" for f in raising)
        parts.append(f"The points that weighed most on our assessment were: {listed}.")
    favour = facts.get("factors_in_your_favour") or []
    if favour:
        listed = "; ".join(f"{f['factor'].lower()} ({f['value']})" for f in favour)
        parts.append(f"Points in your favour: {listed}.")
    advice = facts.get("what_could_improve_a_future_application") or []
    if advice:
        lines = [f"- {' and '.join(a['changes'])}" for a in advice]
        parts.append("The following changes could improve a future application:\n" + "\n".join(lines))
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
            lines.append(f"- {' and '.join(a['changes'])}: probability {a['new_probability']} ({ok})")
    lines += ["", "This is a recommendation; the final decision rests with the officer."]
    return "\n".join(lines)
