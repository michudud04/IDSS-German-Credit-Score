"""Tests for the what-if advice (on the final models)."""

import numpy as np
import pandas as pd
import pytest

from Source import config
from Source.explain.advice import ADVICE_FIELDS, EXAMPLES, MAX_ITEMS, generate_advice, single_changes
from Source.explain.features import FIELDS, as_explained_frame
from Source.explain.pipeline import COMBINED

pytestmark = pytest.mark.skipif(not config.PREPROCESSING_FILE.exists(),
                                reason="run `python -m Source.data.build_dataset` first")

LOW_RISK = dict(RevUtil=0.18, Age=44, Late30_59=0, Late60_89=0, Late90=0, MonthlyIncome=6800,
                MonthlyDebt=1450, OpenLines=7, RealEstateLines=1, Dependents=1)


@pytest.fixture(scope="module")
def pipeline():
    from Source.decision.models import load_risk_pipeline
    return load_risk_pipeline()


@pytest.fixture(scope="module")
def strict():
    from Source.decision.policy import RiskPolicy
    return RiskPolicy.from_cutoff(0.12)  # the app's default cut-off: the borderline example does not pass


def advise(pipeline, profile, policy):
    return generate_advice(pipeline, pd.DataFrame([profile]), policy=policy)[0]


def apply(profile, item):
    changed = dict(profile)
    for change in item.changes:
        changed[change.feature] = change.new_value
    return pd.DataFrame([changed])


def test_no_advice_when_the_applicant_already_passes(pipeline, strict):
    assert advise(pipeline, LOW_RISK, strict) == []


def test_borderline_applicant_gets_a_passing_option(pipeline, strict):
    advice = advise(pipeline, EXAMPLES["borderline"], strict)
    assert advice and advice[0].passes_cutoff


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_reported_numbers_are_reproducible(pipeline, strict, name):
    profile = EXAMPLES[name]
    base = pipeline.predict(pd.DataFrame([profile]))[COMBINED].iloc[0]
    for item in advise(pipeline, profile, strict):
        new = pipeline.predict(apply(profile, item))[COMBINED].iloc[0]
        assert item.new_prob == pytest.approx(new)
        assert item.prob_change == pytest.approx(new - base)
        assert item.passes_cutoff == strict.passes(new)
        assert item.prob_change < 0


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_only_actionable_fields_change_in_the_right_direction(pipeline, strict, name):
    profile = EXAMPLES[name]
    advice = advise(pipeline, profile, strict)
    assert len(advice) <= MAX_ITEMS
    for item in advice:
        assert 1 <= len(item.changes) <= 2
        for change in item.changes:
            assert change.feature in ADVICE_FIELDS
            assert FIELDS[change.feature].actionable and not FIELDS[change.feature].sensitive
            if change.feature == "MonthlyIncome":
                assert change.new_value > change.old_value
            else:
                assert change.new_value < change.old_value


def test_passing_options_come_first_easiest_first(pipeline, strict):
    advice = advise(pipeline, EXAMPLES["borderline"], strict)
    flags = [item.passes_cutoff for item in advice]
    assert flags == sorted(flags, reverse=True)
    efforts = [item.effort for item in advice if item.passes_cutoff]
    assert efforts == sorted(efforts)


def test_undeclared_income_is_never_advised(pipeline, strict):
    advice = advise(pipeline, EXAMPLES["income not declared"], strict)
    assert all(change.feature != "MonthlyIncome" for item in advice for change in item.changes)


def test_open_lines_never_go_below_real_estate_loans():
    row = as_explained_frame(pd.DataFrame([LOW_RISK | {"OpenLines": 5, "RealEstateLines": 3}])).iloc[0]
    lines = [option.changes[0].new_value for option in single_changes(row)["OpenLines"]]
    assert lines == [4, 3]


def test_utilisation_steps_are_round_and_reach_zero():
    row = as_explained_frame(pd.DataFrame([LOW_RISK | {"RevUtil": 0.72}])).iloc[0]
    targets = [option.changes[0].new_value for option in single_changes(row)["RevUtil"]]
    assert targets[0] == pytest.approx(0.70) and targets[-1] == 0.0
    assert np.allclose(np.round(np.array(targets) / 0.05), np.array(targets) / 0.05)


def test_app_style_call_with_cutoff(pipeline):
    """The app calls generate_advice(pipeline, inputs, cutoff=...)."""
    advice = generate_advice(pipeline, pd.DataFrame([EXAMPLES["borderline"]]), cutoff=0.12)[0]
    assert advice and advice[0].passes_cutoff
