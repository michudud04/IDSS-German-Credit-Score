"""Tests for the similar-cases search."""

import numpy as np
import pandas as pd
import pytest

from Source import config
from Source.data.preprocessing import Preprocessor, load_split

pytestmark = pytest.mark.skipif(not config.PREPROCESSING_FILE.exists(),
                                reason="run `python -m Source.data.build_dataset` first")

APPLICANT = {
    "RevUtil": 0.30, "Age": 45, "Late30_59": 0, "Late60_89": 0, "Late90": 0, "MonthlyIncome": 5000,
    "MonthlyDebt": 1500, "OpenLines": 8, "RealEstateLines": 1, "Dependents": 1,
}


@pytest.fixture(scope="module")
def finder():
    from Source.explain.similar import SimilarCaseFinder
    return SimilarCaseFinder.from_training_data()


@pytest.fixture(scope="module")
def prep():
    return Preprocessor.load()


@pytest.fixture(scope="module")
def train():
    return load_split("train", "readable")


def test_result_shape_and_order(finder, prep):
    result = finder.find_for_inputs(prep, pd.DataFrame([APPLICANT]))[0]
    distances = [case.distance for case in result.neighbours]
    assert len(result.neighbours) == finder.n_shown
    assert distances == sorted(distances)
    assert result.neighbourhood_size == finder.neighbourhood_size
    assert 0.0 <= result.neighbourhood_default_rate <= 1.0


def test_neighbourhood_rate_matches_neighbour_outcomes(finder, prep, train):
    _, scaled = prep.prepare_customer(pd.DataFrame([APPLICANT]))
    _, positions = finder.neighbours(scaled, finder.neighbourhood_size)
    expected = train["Default2y"].iloc[positions[0]].mean()
    assert finder.find(scaled)[0].neighbourhood_default_rate == pytest.approx(expected)


def test_training_applicant_finds_itself(finder):
    scaled = load_split("train", "model").iloc[[123]]
    nearest = finder.find(scaled)[0].neighbours[0]
    assert nearest.distance == pytest.approx(0.0, abs=1e-9)


def test_only_training_applicants_are_returned(finder, train):
    val_scaled = load_split("val", "model").head(50)
    ids = {case.case_id for result in finder.find(val_scaled) for case in result.neighbours}
    assert ids <= set(train.index)


def test_age_and_dependents_are_part_of_the_profile(finder, prep):
    young = finder.find_for_inputs(prep, pd.DataFrame([APPLICANT | {"Age": 25, "Dependents": 0}]))[0]
    older = finder.find_for_inputs(prep, pd.DataFrame([APPLICANT | {"Age": 70, "Dependents": 3}]))[0]
    assert np.mean([c.inputs["Age"] for c in young.neighbours]) < np.mean([c.inputs["Age"] for c in older.neighbours])
    assert np.mean([c.inputs["Dependents"] for c in young.neighbours]) < \
        np.mean([c.inputs["Dependents"] for c in older.neighbours])


def test_lateness_code_applicants_match_each_other(finder, prep, train):
    coded = APPLICANT | {"Late30_59": np.nan, "Late60_89": np.nan, "Late90": np.nan, "LateCodeFlag": 1}
    _, scaled = prep.prepare_customer(pd.DataFrame([coded]))
    _, positions = finder.neighbours(scaled, finder.neighbourhood_size)
    assert train["LateCodeFlag"].iloc[positions[0]].eq(1).all()
