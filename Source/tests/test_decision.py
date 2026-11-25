"""Tests for the decision step (policy, rules, precedence). Most need no data."""

import math

import numpy as np
import pandas as pd
import pytest

from Source import config
from Source.decision.engine import decide
from Source.decision.policy import RiskPolicy, cost_ratio_from_cutoff, cutoff_from_cost_ratio
from Source.decision.rules import check_rules
from Source.explain.types import UnusualCheck

ORDINARY = dict(RevUtil=0.25, Age=48, Late30_59=0, Late60_89=0, Late90=0, MonthlyIncome=5500,
                MonthlyDebt=1700, OpenLines=8, RealEstateLines=1, Dependents=1, LateCodeFlag=0)


def probs(p, spread=0.0):
    return {"logreg": p - spread, "tree": p + spread, "xgboost": p, "combined": p}


def unusual(flag=True):
    return UnusualCheck(is_unusual=flag, score=3.0, threshold=2.3,
                        reasons=["No past applicant has a similar profile."] if flag else [],
                        triggers=["far_from_past_applicants"] if flag else [])


# -- policy -------------------------------------------------------------------

def test_default_cutoff_matches_model_evaluation():
    assert RiskPolicy().cutoff == pytest.approx(0.20)


@pytest.mark.parametrize("ratio", [1, 2.5, 4, 9, 20])
def test_cutoff_and_ratio_are_inverse(ratio):
    assert cost_ratio_from_cutoff(cutoff_from_cost_ratio(ratio)) == pytest.approx(ratio)


def test_cutoff_is_where_expected_costs_are_equal():
    policy = RiskPolicy(cost_ratio=7)
    costs = policy.expected_costs(policy.cutoff)
    assert costs["approve"] == pytest.approx(costs["reject"])


def test_higher_cost_ratio_is_stricter():
    assert RiskPolicy(cost_ratio=10).cutoff < RiskPolicy(cost_ratio=2).cutoff


def test_bands_and_passes():
    policy = RiskPolicy()
    assert policy.band(0.05) == "approve" and policy.passes(0.05)
    assert policy.band(0.19) == "review" and not policy.passes(0.19)
    assert policy.band(0.90) == "reject"
    assert policy.approve_below < policy.cutoff < policy.reject_from


def test_slider_limits():
    with pytest.raises(ValueError):
        RiskPolicy(cost_ratio=0.5)


# -- rules --------------------------------------------------------------------

def test_ordinary_applicant_fires_no_rule():
    assert check_rules(pd.Series(ORDINARY)) == []


def test_debt_above_declared_income_goes_to_review():
    hits = check_rules(pd.Series(ORDINARY | {"MonthlyIncome": 2000, "MonthlyDebt": 2500}))
    assert [r.code for r in hits] == ["RULE_DEBT_EXCEEDS_INCOME"]
    assert hits[0].effect == "review"


def test_undeclared_income_is_not_a_rule():
    assert check_rules(pd.Series(ORDINARY | {"MonthlyIncome": math.nan})) == []


def test_serious_delinquency_and_minor_are_rejected():
    assert check_rules(pd.Series(ORDINARY | {"Late90": 3}))[0].code == "RULE_SERIOUS_DELINQUENCY"
    assert check_rules(pd.Series(ORDINARY | {"Age": 17}))[0].code == "RULE_MIN_AGE"


def test_blank_history_asks_for_review_unless_lateness_code():
    assert check_rules(pd.Series(ORDINARY | {"Late90": math.nan}))[0].code == "RULE_MISSING_HISTORY"
    coded = ORDINARY | {"Late30_59": math.nan, "Late60_89": math.nan, "Late90": math.nan, "LateCodeFlag": 1}
    assert check_rules(pd.Series(coded)) == []


# -- precedence ---------------------------------------------------------------

def test_low_risk_is_approved():
    assert decide(probs(0.03), RiskPolicy(), unusual(False)).outcome == "approve"


def test_grey_zone_goes_to_review():
    assert decide(probs(0.18), RiskPolicy()).outcome == "review"


def test_unusual_overrides_a_low_probability():
    assert decide(probs(0.03), RiskPolicy(), unusual()).outcome == "review"


def test_unusual_overrides_a_high_probability():
    assert decide(probs(0.80), RiskPolicy(), unusual()).outcome == "review"


def test_reject_rule_overrides_everything():
    reasons = check_rules(pd.Series(ORDINARY | {"Late90": 4}))
    assert decide(probs(0.03), RiskPolicy(), unusual(), reasons).outcome == "reject"


def test_review_rule_does_not_soften_a_clear_rejection():
    reasons = check_rules(pd.Series(ORDINARY | {"MonthlyIncome": 2000, "MonthlyDebt": 2500}))
    assert decide(probs(0.60), RiskPolicy(), None, reasons).outcome == "reject"
    assert decide(probs(0.03), RiskPolicy(), None, reasons).outcome == "review"


def test_model_disagreement_goes_to_review():
    # combined 0.12 is an approval on its own, but one model is above the 20% cut-off
    decision = decide(probs(0.12, spread=0.10), RiskPolicy())
    assert decision.outcome == "review"
    assert "MODELS_DISAGREE" in decision.reason_codes


def test_decisive_reasons_come_first_and_are_serialisable():
    decision = decide(probs(0.03), RiskPolicy(), unusual())
    assert decision.reasons[0].code == "UNUSUAL_CASE"
    assert decision.to_dict()["outcome"] == "review"


def test_internal_reasons_never_reach_the_customer():
    decision = decide(probs(0.12, spread=0.10), RiskPolicy())
    disagreement = next(r for r in decision.reasons if r.code == "MODELS_DISAGREE")
    assert disagreement.customer_text is None


# -- end to end with the saved models ---------------------------------------

@pytest.mark.skipif(not config.PREPROCESSING_FILE.exists(), reason="run `python -m Source.data.build_dataset` first")
def test_engine_on_saved_models():
    from Source.decision.engine import DecisionEngine
    engine = DecisionEngine.from_saved()
    risky = ORDINARY | {"RevUtil": 1.2, "Late30_59": 4, "Late60_89": 2, "Late90": 2, "MonthlyIncome": 1800}
    decisions = engine.assess(pd.DataFrame([ORDINARY, risky]))
    assert [d.outcome for d in decisions] == ["approve", "reject"]
    combined = engine.pipeline.predict(pd.DataFrame([ORDINARY]))["combined"].iloc[0]
    assert decisions[0].probability == pytest.approx(combined)
    assert np.isclose(sum(engine.pipeline.weights.values()), 1.0)
