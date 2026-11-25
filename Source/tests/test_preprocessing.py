"""Tests for the data split and run-time preprocessing.  Run: ``pytest Source/tests``."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from Source import config
from Source.data.preprocessing import (INCOME_FLOOR, INPUT_FIELDS, TARGET, Preprocessor, correct_errors,
                                       load_raw, load_split)

pytestmark = pytest.mark.skipif(not config.PREPROCESSING_FILE.exists(),
                                reason="run `python -m Source.data.build_dataset` first")


@pytest.fixture(scope="module")
def prep():
    return Preprocessor.load()


def test_splits_are_disjoint_and_sized():
    idx = {name: set(load_split(name).index) for name in ("train", "val", "test")}
    assert len(idx["train"]) == 96_000 and len(idx["val"]) == 24_000 and len(idx["test"]) == 30_000
    assert not (idx["train"] & idx["val"]) and not (idx["train"] & idx["test"]) and not (idx["val"] & idx["test"])


def test_test_split_matches_eda_notebook():
    """The test set must be the same rows the EDA notebook kept aside."""
    clean = correct_errors(load_raw())
    _, eda_test = train_test_split(clean, test_size=0.20, stratify=clean[TARGET], random_state=config.RANDOM_STATE)
    assert set(eda_test.index) == set(load_split("test").index)


def test_new_applicant_with_blank_income(prep):
    applicant = pd.DataFrame([{
        "RevUtil": 0.45, "Age": 38, "Late30_59": 1, "Late60_89": 0, "Late90": 0, "MonthlyIncome": np.nan,
        "MonthlyDebt": 1800, "OpenLines": 6, "RealEstateLines": 1, "Dependents": np.nan,
    }])
    readable, model = prep.prepare_customer(applicant)
    row = readable.iloc[0]
    assert row["IncomeMissing"] == 1 and row["DependentsMissing"] == 1
    assert row["MonthlyIncome"] > INCOME_FLOOR                       # income was estimated
    assert row["DebtRatio"] == pytest.approx(row["MonthlyDebt"] / row["MonthlyIncome"])
    assert np.isfinite(model.drop(columns=[c for c in model if model[c].dtype == object]).to_numpy(float)).all()


def test_income_below_floor_is_treated_as_missing(prep):
    applicant = pd.DataFrame([dict.fromkeys(INPUT_FIELDS, 0.0) | {"MonthlyIncome": 50.0, "Age": 40.0}])
    readable, _ = prep.prepare_customer(applicant)
    assert readable.iloc[0]["IncomeMissing"] == 1
    assert readable.iloc[0]["MonthlyIncome"] != 50.0
