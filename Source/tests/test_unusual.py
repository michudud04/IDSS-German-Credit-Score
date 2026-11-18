"""Tests for the unusual-case warning."""

import numpy as np
import pandas as pd
import pytest

from Source import config
from Source.data.preprocessing import Preprocessor, load_split
from Source.explain.unusual import (DETECTOR_FILE, SYNTHETIC_PROFILES, TRIGGER_CODE, TRIGGER_FAR, TRIGGER_RANGE,
                                    UnusualCaseDetector)

pytestmark = pytest.mark.skipif(not (config.PREPROCESSING_FILE.exists() and DETECTOR_FILE.exists()),
                                reason="run `python -m Source.data.build_dataset` and `python -m Source.explain.unusual`")


@pytest.fixture(scope="module")
def detector():
    from Source.explain.unusual import load_detector
    return load_detector()


@pytest.fixture(scope="module")
def prep():
    return Preprocessor.load()


def check(detector, prep, profile):
    return detector.check_inputs(prep, pd.DataFrame([profile]))[0]


def test_ordinary_applicant_is_not_unusual(detector, prep):
    result = check(detector, prep, SYNTHETIC_PROFILES["ordinary"])
    assert not result.is_unusual and result.triggers == [] and result.reasons == []
    assert result.score < result.threshold


def test_extreme_history_is_far_from_past_applicants(detector, prep):
    result = check(detector, prep, SYNTHETIC_PROFILES["many late payments"])
    assert result.is_unusual and TRIGGER_FAR in result.triggers
    assert any("higher than 99.9%" in reason for reason in result.reasons)


def test_value_outside_observed_range(detector, prep):
    result = check(detector, prep, SYNTHETIC_PROFILES["too young for the data"])
    assert result.is_unusual and TRIGGER_RANGE in result.triggers
    assert any(reason.startswith("Age") for reason in result.reasons)


def test_lateness_codes_are_always_unusual(detector, prep):
    result = check(detector, prep, SYNTHETIC_PROFILES["lateness codes"])
    assert result.is_unusual and TRIGGER_CODE in result.triggers


def test_blank_income_is_not_range_checked(detector, prep):
    result = check(detector, prep, SYNTHETIC_PROFILES["ordinary"] | {"MonthlyIncome": np.nan})
    assert not any(reason.startswith("Monthly income") for reason in result.reasons)


def test_flagged_cases_always_have_reasons(detector):
    readable, scaled = load_split("val", "readable").head(3000), load_split("val", "model").head(3000)
    for result in detector.check(readable, scaled):
        assert result.is_unusual == bool(result.triggers) == bool(result.reasons)


def test_threshold_flags_about_one_percent_of_validation(detector):
    scores = detector.scores(load_split("val", "model"))
    assert np.mean(scores > detector.threshold) == pytest.approx(detector.settings["flag_rate"], abs=0.001)


def test_settings_round_trip(detector, tmp_path):
    path = tmp_path / "detector.json"
    detector.save(path)
    assert UnusualCaseDetector.load(detector.finder, path).settings == detector.settings
