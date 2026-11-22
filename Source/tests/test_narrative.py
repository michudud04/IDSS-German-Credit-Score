"""Tests for the officer summary and customer letter (no LLM needed: fake backends)."""

import pytest

from Source.decision.engine import decide
from Source.decision.narrative import guard
from Source.decision.narrative.facts import customer_facts, officer_facts
from Source.decision.narrative.writer import NarrativeWriter
from Source.decision.policy import RiskPolicy
from Source.decision.service import evaluate_decision, make_policy
from Source.explain.types import AdviceItem, FeatureChange, KeyFactor, SimilarCase, SimilarCases, UnusualCheck

FACTORS = [
    KeyFactor("RevUtil", "Revolving credit utilisation", 0.95, "95%", "increases_risk", 0.06, {}, True, False),
    KeyFactor("Late30_59", "Payments 30-59 days late (last 2 years)", 2, "2", "increases_risk", 0.04, {}, False, False),
    KeyFactor("Age", "Age", 29, "29", "increases_risk", 0.02, {}, False, True),
    KeyFactor("MonthlyIncome", "Monthly income", 3200, "$3,200 (estimated)", "decreases_risk", -0.01, {}, True, False),
]
ADVICE = [
    AdviceItem([FeatureChange("RevUtil", "Revolving credit utilisation", 0.95, 0.30,
                              "Lower utilisation from 95% to 30%")], 0.09, -0.21, True, 0.5),
    AdviceItem([FeatureChange("Age", "Age", 29, 45, "Be older")], 0.10, -0.2, True, 0.1),  # must be dropped
]
SIMILAR = SimilarCases([SimilarCase(1, 0.3, {"Age": 31.0, "Dependents": 3.0}, True)], 50, 0.24, 0.067)
INPUTS = dict(RevUtil=0.95, Age=29, Late30_59=2, Late60_89=0, Late90=0, MonthlyIncome=float("nan"),
              MonthlyDebt=1500, OpenLines=5, RealEstateLines=0, Dependents=3, LateCodeFlag=0)


def probs(p):
    return {"logreg": p, "tree": p, "xgboost": p, "combined": p}


class FakeBackend:
    name = "fake"

    def __init__(self, *drafts):
        self.drafts, self.calls = list(drafts), []

    def generate(self, system, prompt):
        self.calls.append(prompt)
        return self.drafts.pop(0)


class BrokenBackend:
    name = "broken"

    def generate(self, system, prompt):
        raise OSError("connection refused")


GOOD_REJECTION = ("Dear applicant,\n\nAfter careful assessment, we are unable to approve your loan application at "
                  "this time. The main points were your revolving credit utilisation of 95% and two payments that "
                  "were 30-59 days late. Lowering your utilisation from 95% to 30% could improve a future "
                  "application.\n\nKind regards,\nThe Credit Team")


# -- facts filtering ----------------------------------------------------------

def test_customer_facts_never_contain_sensitive_data_or_other_customers():
    decision = decide(probs(0.6), RiskPolicy())
    text = str(customer_facts(decision, FACTORS, ADVICE)).lower()
    assert "age" not in text.replace("utilisation", "").replace("percentage", "")
    assert "dependents" not in text and "31.0" not in text and "be older" not in text
    assert "0.6" not in text and "60" not in text  # no probability


def test_customer_sees_estimated_income_as_not_declared():
    facts = customer_facts(decide(probs(0.6), RiskPolicy()), FACTORS, ADVICE)
    assert {"factor": "Monthly income", "value": "not declared"} in facts["factors_in_your_favour"]


def test_officer_facts_keep_everything_for_the_officer():
    facts = officer_facts(decide(probs(0.6), RiskPolicy()), FACTORS, ADVICE, SIMILAR, None, INPUTS)
    assert facts["similar_past_applicants"]["number_compared"] == 50
    assert any(f["sensitive_do_not_advise"] for f in facts["key_factors"])


def test_no_advice_in_an_approval_letter():
    assert "what_could_improve_a_future_application" not in customer_facts(decide(probs(0.02), RiskPolicy()),
                                                                           FACTORS, ADVICE)


# -- guard --------------------------------------------------------------------

def facts_for(outcome_p):
    decision = decide(probs(outcome_p), RiskPolicy())
    return decision, customer_facts(decision, FACTORS, ADVICE)


def test_guard_accepts_a_good_letter():
    decision, facts = facts_for(0.6)
    assert guard.check_customer_letter(GOOD_REJECTION, facts, decision.outcome) == []


@pytest.mark.parametrize("bad, problem", [
    (GOOD_REJECTION.replace("this time.", "this time, given your age."), "age"),
    (GOOD_REJECTION.replace("this time.", "this time. Similar applicants often defaulted."), "other customers"),
    (GOOD_REJECTION.replace("this time.", "this time. Our model rated you."), "internal model"),
    (GOOD_REJECTION.replace("95% to 30%", "95% to 25%"), "numbers"),
    (GOOD_REJECTION.replace("Dear applicant", "Dear [Customer Name]"), "placeholder"),
    (GOOD_REJECTION.replace("we are unable to approve your loan application at this time",
                            "your loan has been approved"), "outcome"),
])
def test_guard_catches_problems(bad, problem):
    decision, facts = facts_for(0.6)
    issues = guard.check_customer_letter(bad, facts, decision.outcome)
    assert any(problem in issue for issue in issues), issues


def test_templates_always_pass_the_guard():
    for p in (0.02, 0.18, 0.6):
        decision, facts = facts_for(p)
        letter = NarrativeWriter(backend=None).customer_letter(decision, FACTORS, ADVICE)
        assert guard.check_customer_letter(letter.text, facts, decision.outcome) == [], p
        summary = NarrativeWriter(backend=None).officer_summary(decision, FACTORS, ADVICE, SIMILAR, None, INPUTS)
        assert guard.check_officer_summary(summary.text, officer_facts(decision, FACTORS, ADVICE, SIMILAR, None,
                                                                       INPUTS)) == []


# -- writer -------------------------------------------------------------------

def test_good_llm_text_is_used():
    decision, _ = facts_for(0.6)
    letter = NarrativeWriter(FakeBackend(GOOD_REJECTION)).customer_letter(decision, FACTORS, ADVICE)
    assert (letter.source, letter.attempts, letter.text) == ("llm", 1, GOOD_REJECTION)


def test_bad_draft_is_retried_with_feedback():
    decision, _ = facts_for(0.6)
    backend = FakeBackend(GOOD_REJECTION.replace("this time.", "this time, given your age."), GOOD_REJECTION)
    letter = NarrativeWriter(backend).customer_letter(decision, FACTORS, ADVICE)
    assert letter.source == "llm" and letter.attempts == 2
    assert "broke these rules" in backend.calls[1]


def test_repeated_bad_drafts_fall_back_to_template():
    decision, _ = facts_for(0.6)
    bad = GOOD_REJECTION.replace("this time.", "this time, given your age.")
    letter = NarrativeWriter(FakeBackend(bad, bad)).customer_letter(decision, FACTORS, ADVICE)
    assert letter.source == "template" and letter.issues


def test_unreachable_llm_falls_back_to_template():
    decision, _ = facts_for(0.6)
    letter = NarrativeWriter(BrokenBackend()).customer_letter(decision, FACTORS, ADVICE)
    assert letter.source == "template" and "failed" in letter.issues[0]


# -- app entry point ------------------------------------------------------------

def test_evaluate_decision_matches_the_app_contract():
    res = evaluate_decision(inputs=INPUTS, readable_row={}, probabilities=probs(0.6), risk_cutoff=0.2,
                            review_margin=None, key_factors=FACTORS, advice=ADVICE, similar_cases=SIMILAR,
                            unusual=UnusualCheck(False, 1.0, 2.3), writer=NarrativeWriter(backend=None))
    assert res["decision"] == "reject"
    assert isinstance(res["decision_reasons"], list) and isinstance(res["bank_rules_triggered"], list)
    assert res["officer_summary"] and res["customer_letter"]


def test_make_policy_from_app_slider():
    assert make_policy(risk_cutoff=0.2).cutoff == pytest.approx(0.2)
    assert make_policy(risk_cutoff=0.2, review_margin=0.05).reject_from == pytest.approx(0.25)
    assert make_policy(risk_cutoff=0.9).cutoff == pytest.approx(0.5)  # clamped to the slider range


# -- real failures seen with llama3.2:3b (kept as regression tests) ----------

REAL_APPROVAL_LETTER = """Dear applicant,

We are pleased to inform you that your loan application has been approved.

Our decision is based on the fact that your overall credit profile meets our lending criteria.

We would like to take this opportunity to remind you that maintaining a good credit profile will be essential for future applications. To improve your chances of approval in the future, consider:

* Continuing to make timely payments on all debts
* Keeping credit utilization ratios low

Kind regards,
The Credit Team"""

REAL_OFFICER_SUMMARY = """Recommendation: APPROVE with probability of default 1.5% and cut-off 20.0%.

Why:
* Estimated probability of default is below the automatic-approval limit of 16.0% (cost ratio 4).

Key factors:
* Cost ratio: 4 (sensitive_do_not_advise=true)

Unusual-case warning: None.

The officer takes the final decision."""


def test_guard_rejects_invented_advice_in_letter():
    decision = decide(probs(0.015), RiskPolicy())
    issues = guard.check_customer_letter(REAL_APPROVAL_LETTER, customer_facts(decision), decision.outcome)
    assert any("advice" in issue for issue in issues), issues


def test_guard_rejects_invented_officer_sections():
    decision = decide(probs(0.015), RiskPolicy())
    issues = guard.check_officer_summary(REAL_OFFICER_SUMMARY, officer_facts(decision))
    assert any("key factors" in i for i in issues) and any("unusual" in i for i in issues), issues


def test_officer_prompt_lists_only_sections_with_facts():
    decision = decide(probs(0.015), RiskPolicy())
    system = __import__("Source.decision.narrative.prompts", fromlist=["x"]).officer_system(officer_facts(decision))
    assert "Key factors" not in system and "Unusual" not in system and "Why:" in system


def test_considered_is_not_advice():
    decision = decide(probs(0.6), RiskPolicy())
    letter = GOOD_REJECTION.replace("Lowering your utilisation from 95% to 30% could improve a future application.",
                                    "We have carefully considered your application.")
    facts = customer_facts(decision, FACTORS)  # no advice in the facts
    assert not any("advice" in i for i in guard.check_customer_letter(letter, facts, decision.outcome))


def test_officer_llm_facts_leave_out_raw_form_values():
    facts = officer_facts(decide(probs(0.2), RiskPolicy()), inputs=INPUTS)
    assert "applicant" not in facts


def test_empty_lists_are_not_sent_to_the_llm():
    facts = customer_facts(decide(probs(0.02), RiskPolicy()))
    assert all(value for value in facts.values())
