"""Tests for the key factors (exact SHAP on the final models)."""

import numpy as np
import pandas as pd
import pytest

from Source import config
from Source.explain.factors import EXAMPLES, FactorExplainer, explain_factors, get_explainer
from Source.explain.features import EXPLAINED_FIELDS
from Source.explain.pipeline import COMBINED

pytestmark = pytest.mark.skipif(not config.PREPROCESSING_FILE.exists(),
                                reason="run `python -m Source.data.build_dataset` first")


@pytest.fixture(scope="module")
def pipeline():
    from Source.decision.models import load_risk_pipeline
    return load_risk_pipeline()


@pytest.fixture(scope="module")
def explainer(pipeline):
    # A smaller reference sample keeps the tests fast; every property tested holds for any size.
    return FactorExplainer.from_training_data(pipeline, n_reference=50)


def inputs_for(name):
    return pd.DataFrame([EXAMPLES[name]])


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_reference_plus_factors_equals_prediction(explainer, pipeline, name):
    table = explainer.contributions(inputs_for(name))[0]
    probs = pipeline.predict(inputs_for(name)).iloc[0]
    for column in list(pipeline.models) + [COMBINED]:
        assert table[column].sum() == pytest.approx(probs[column], abs=1e-6)


def test_merged_factors_are_the_weighted_model_factors(explainer, pipeline):
    table = explainer.contributions(inputs_for("borderline"))[0]
    expected = sum(pipeline.weights[m] * table[m] for m in pipeline.models)
    assert np.allclose(table[COMBINED], expected)


def test_top_factors_are_sorted_and_consistent(explainer):
    factors = explainer.explain(inputs_for("high risk"), top_n=5)[0]
    effects = [abs(f.contribution) for f in factors]
    assert len(factors) == 5 and effects == sorted(effects, reverse=True)
    for f in factors:
        assert (f.contribution > 0) == (f.direction == "increases_risk")
        assert set(f.per_model) == {"logreg", "tree", "xgboost"}


def test_high_utilisation_and_late_payments_raise_risk(explainer):
    factors = {f.feature: f for f in explainer.explain(inputs_for("high risk"), top_n=None)[0]}
    assert factors["RevUtil"].direction == "increases_risk"
    assert factors["Late90"].direction == "increases_risk"
    low = {f.feature: f for f in explainer.explain(inputs_for("low risk"), top_n=None)[0]}
    assert low["RevUtil"].direction == "decreases_risk"


def test_flags_follow_the_field_metadata(explainer):
    factors = {f.feature: f for f in explainer.explain(inputs_for("borderline"), top_n=None)[0]}
    assert factors["Age"].sensitive and not factors["Age"].actionable
    assert factors["RevUtil"].actionable and not factors["RevUtil"].sensitive
    assert not factors["Late30_59"].actionable


def test_blank_income_is_shown_as_estimate(explainer):
    factors = {f.feature: f for f in explainer.explain(inputs_for("income not declared"), top_n=None)[0]}
    assert factors["MonthlyIncome"].value is None
    assert "estimated" in factors["MonthlyIncome"].value_text


def test_every_input_field_can_be_explained(explainer):
    table = explainer.contributions(inputs_for("borderline"))[0]
    assert list(table.index) == EXPLAINED_FIELDS + ["reference"]


def test_explain_factors_builds_one_explainer_per_pipeline(pipeline):
    first = get_explainer(pipeline)
    assert get_explainer(pipeline) is first
    assert explain_factors(pipeline, inputs_for("borderline"))[0]
