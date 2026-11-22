from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Source import config
from Source.app.case_result import CaseResult
from Source.app.orchestrator import IDSSOrchestrator
from Source.app.use_cases import USE_CASES
from Source.data.preprocessing import INPUT_FIELDS, Preprocessor
from Source.explain.pipeline import RiskPipeline
from Source.explain.similar import SimilarCaseFinder
from Source.explain.unusual import DETECTOR_FILE, load_detector

pytestmark = pytest.mark.skipif(
    not (config.PREPROCESSING_FILE.exists() and DETECTOR_FILE.exists()),
    reason="run `python -m Source.data.build_dataset` and `python -m Source.explain.unusual`",
)


class FixedProbModel:
    """Deterministic stub model so tests do not depend on temporary stand-in model weights."""

    def __init__(self, prob: float) -> None:
        self.prob = prob

    def predict_default_proba(self, readable: pd.DataFrame, scaled: pd.DataFrame) -> np.ndarray:
        return np.full(len(readable), self.prob, dtype=float)


@pytest.fixture(scope="module")
def shared_components():
    prep = Preprocessor.load()
    finder = SimilarCaseFinder.from_training_data()
    detector = load_detector(finder)
    return prep, finder, detector


def make_orchestrator(shared_components, probs: tuple[float, float, float]) -> IDSSOrchestrator:
    prep, finder, detector = shared_components
    models = {
        "logreg": FixedProbModel(probs[0]),
        "xgboost": FixedProbModel(probs[1]),
        "tree": FixedProbModel(probs[2]),
    }
    pipeline = RiskPipeline(prep, models)
    return IDSSOrchestrator(
        preprocessor=prep,
        pipeline=pipeline,
        similar_finder=finder,
        unusual_detector=detector,
    )


def test_use_cases_have_valid_input_schema() -> None:
    for uc_id, uc in USE_CASES.items():
        assert set(INPUT_FIELDS).issubset(uc.inputs.keys()), uc_id
        assert uc.expected_outcome in ("approve", "review", "reject")


def test_case_result_contract_and_serialization(shared_components) -> None:
    orch = make_orchestrator(shared_components, (0.02, 0.03, 0.04))
    res = orch.evaluate_applicant(USE_CASES["uc1_prime_approval"].inputs, risk_cutoff=0.12)

    assert isinstance(res, CaseResult)
    assert res.combined_probability == pytest.approx(0.03)
    assert res.model_disagreement == pytest.approx(0.02)
    assert set(res.probabilities) == {"logreg", "xgboost", "tree", "combined"}
    assert len(res.similar_cases.neighbours) == 5
    assert res.is_unusual == res.unusual.is_unusual
    assert isinstance(res.to_dict(), dict)


def test_decision_thresholds_with_deterministic_models(shared_components) -> None:
    inputs = USE_CASES["uc1_prime_approval"].inputs

    low_orch = make_orchestrator(shared_components, (0.02, 0.03, 0.04))
    assert low_orch.evaluate_applicant(inputs, risk_cutoff=0.12, review_margin=0.03).decision == "approve"

    mid_orch = make_orchestrator(shared_components, (0.11, 0.12, 0.13))
    assert mid_orch.evaluate_applicant(inputs, risk_cutoff=0.12, review_margin=0.03).decision == "review"

    high_orch = make_orchestrator(shared_components, (0.25, 0.30, 0.35))
    assert high_orch.evaluate_applicant(inputs, risk_cutoff=0.12, review_margin=0.03).decision == "reject"


def test_unusual_profile_triggers_warning_and_review(shared_components) -> None:
    orch = make_orchestrator(shared_components, (0.02, 0.02, 0.02))
    res = orch.evaluate_applicant(USE_CASES["uc4_unusual_ood_review"].inputs, risk_cutoff=0.25)
    assert res.is_unusual is True
    assert len(res.unusual.reasons) > 0
    assert res.decision == "review"
