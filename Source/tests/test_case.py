"""Tests for explain_case: one call for all four explanations, as the app uses it."""

import pandas as pd
import pytest

from Source import config
from Source.data.preprocessing import Preprocessor
from Source.explain.advice import EXAMPLES, generate_advice, resolve_policy
from Source.explain.factors import explain_factors
from Source.explain.types import CaseExplanation
from Source.explain.unusual import SYNTHETIC_PROFILES

pytestmark = pytest.mark.skipif(not config.PREPROCESSING_FILE.exists(),
                                reason="run `python -m Source.data.build_dataset` first")

PROFILES = {
    "borderline": EXAMPLES["borderline"],
    "high risk": EXAMPLES["high risk"],
    "income not declared": EXAMPLES["income not declared"],
    "very old (unusual)": SYNTHETIC_PROFILES["very old"],
    "lateness codes (unusual)": SYNTHETIC_PROFILES["lateness codes"],
}


@pytest.fixture(scope="module")
def parts():
    from Source.decision.models import load_risk_pipeline
    from Source.explain.similar import SimilarCaseFinder
    from Source.explain.unusual import load_detector
    pipeline = load_risk_pipeline()
    finder = SimilarCaseFinder.from_training_data()
    return pipeline, finder, load_detector(finder), Preprocessor.load()


@pytest.mark.parametrize("name", list(PROFILES))
def test_same_results_as_the_separate_calls(parts, name):
    from Source.explain import explain_case
    pipeline, finder, detector, prep = parts
    inputs = pd.DataFrame([PROFILES[name]])
    policy = resolve_policy(cutoff=0.12, review_margin=0.03)

    result = explain_case(pipeline, finder, detector, inputs, cutoff=0.12, review_margin=0.03)

    assert result.probabilities == pipeline.predict(inputs).iloc[0].to_dict()
    assert result.key_factors == explain_factors(pipeline, inputs)[0]
    assert result.advice == generate_advice(pipeline, inputs, policy=policy)[0]
    assert result.similar_cases == finder.find_for_inputs(prep, inputs)[0]
    assert result.unusual == detector.check_inputs(prep, inputs)[0]


def test_called_like_the_app(parts):
    """The app imports it from the package and passes the cut-off as a keyword."""
    from Source.explain import explain_case
    pipeline, finder, detector, _ = parts
    result = explain_case(pipeline, finder, detector, pd.DataFrame([PROFILES["borderline"]]), cutoff=0.12)
    assert isinstance(result, CaseExplanation)
    assert result.key_factors and result.advice and result.similar_cases.neighbours
    assert not result.unusual.is_unusual


def test_dict_input_and_one_applicant_only(parts):
    from Source.explain import explain_case
    pipeline, finder, detector, _ = parts
    assert isinstance(explain_case(pipeline, finder, detector, PROFILES["high risk"]), CaseExplanation)
    with pytest.raises(ValueError):
        explain_case(pipeline, finder, detector, pd.DataFrame([PROFILES["borderline"], PROFILES["high risk"]]))


def test_several_applicants_at_once(parts):
    from Source.explain import explain_case, explain_cases
    pipeline, finder, detector, _ = parts
    names = ["borderline", "very old (unusual)"]
    batch = explain_cases(pipeline, finder, detector, pd.DataFrame([PROFILES[n] for n in names]), cutoff=0.12)
    singles = [explain_case(pipeline, finder, detector, PROFILES[n], cutoff=0.12) for n in names]
    for many, one in zip(batch, singles):
        # Matrix products over 2 rows round the last digit differently than over 1 row (~1e-17).
        assert many.probabilities == pytest.approx(one.probabilities, abs=1e-12)
        assert (many.key_factors, many.advice, many.similar_cases, many.unusual) == \
            (one.key_factors, one.advice, one.similar_cases, one.unusual)


def test_slider_reaches_the_advice(parts):
    """At the decision default (20%) the borderline applicant passes, at 12% it gets advice."""
    from Source.explain import explain_case
    pipeline, finder, detector, _ = parts
    inputs = PROFILES["borderline"]
    assert explain_case(pipeline, finder, detector, inputs).advice == []
    assert explain_case(pipeline, finder, detector, inputs, cutoff=0.12).advice
