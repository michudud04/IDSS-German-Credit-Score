"""Tests for the RiskPipeline wrapper, run on the final models (Person B, loaded by Person D's loader)."""

import numpy as np
import pytest

from Source import config
from Source.data.preprocessing import INPUT_FIELDS, load_split
from Source.explain.pipeline import COMBINED

pytestmark = pytest.mark.skipif(not config.PREPROCESSING_FILE.exists(),
                                reason="run `python -m Source.data.build_dataset` first")


@pytest.fixture(scope="module")
def pipeline():
    from Source.decision.models import load_risk_pipeline
    return load_risk_pipeline()


@pytest.fixture(scope="module")
def applicants():
    return load_split("val")[INPUT_FIELDS + ["LateCodeFlag"]].head(500)


def test_probabilities_are_valid(pipeline, applicants):
    probs = pipeline.predict(applicants)
    assert list(probs.columns) == list(pipeline.models) + [COMBINED]
    assert ((probs >= 0) & (probs <= 1)).all().all()


def test_combined_is_weighted_average(pipeline, applicants):
    probs = pipeline.predict(applicants)
    expected = sum(pipeline.weights[m] * probs[m] for m in pipeline.models)
    assert np.allclose(probs[COMBINED], expected)
    assert sum(pipeline.weights.values()) == pytest.approx(1.0)


def test_prediction_is_deterministic(pipeline, applicants):
    assert pipeline.predict(applicants).equals(pipeline.predict(applicants))
