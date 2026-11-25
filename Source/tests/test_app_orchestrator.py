"""Tests for the app orchestrator, CaseResult, and PW3 folder structure."""

import numpy as np
import pandas as pd
import pytest

from Source import config
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
    def __init__(self, prob: float):
        self.prob = prob

    def predict_default_proba(self, readable: pd.DataFrame, scaled: pd.DataFrame) -> np.ndarray:
        return np.full(len(readable), self.prob, dtype=float)


@pytest.fixture(scope="module")
def shared_components():
    prep = Preprocessor.load()
    finder = SimilarCaseFinder.from_training_data()
    detector = load_detector(finder)
    return prep, finder, detector


@pytest.fixture(scope="module")
def production_orchestrator(shared_components) -> IDSSOrchestrator:
    prep, finder, detector = shared_components
    return IDSSOrchestrator(preprocessor=prep, similar_finder=finder, unusual_detector=detector)


def make_orchestrator(shared_components, probs: tuple[float, float, float]) -> IDSSOrchestrator:
    prep, finder, detector = shared_components
    models = {
        "logreg": FixedProbModel(probs[0]),
        "xgboost": FixedProbModel(probs[1]),
        "tree": FixedProbModel(probs[2]),
    }
    return IDSSOrchestrator(
        preprocessor=prep,
        pipeline=RiskPipeline(prep, models),
        similar_finder=finder,
        unusual_detector=detector,
    )


def test_pw3_deliverable_structure() -> None:
    root = config.ROOT
    for folder in ("Documentation", "Models", "Source", "Demo", "Presentation"):
        assert (root / folder).is_dir()
    assert (root / "README.txt").is_file()
    assert (config.MODELS_DIR / "logistic_regression_weights.json").is_file()
    assert (config.MODELS_DIR / "xgboost_feature_importance.json").is_file()


def test_use_cases_schema() -> None:
    for uc in USE_CASES.values():
        assert set(INPUT_FIELDS).issubset(uc.inputs.keys())
        assert uc.expected_outcome in ("approve", "review", "reject")


def test_case_result_fields(shared_components) -> None:
    orch = make_orchestrator(shared_components, (0.02, 0.03, 0.04))
    res = orch.evaluate_applicant(USE_CASES["uc1_prime_approval"].inputs, risk_cutoff=0.12)

    assert res.combined_probability == pytest.approx(0.03)
    assert res.model_disagreement == pytest.approx(0.02)
    assert len(res.similar_cases.neighbours) == 5
    assert "approve_below" in res.policy_summary
    assert res.to_dict()["decision"] == "approve"


def test_decision_thresholds(shared_components) -> None:
    inputs = USE_CASES["uc1_prime_approval"].inputs

    low_orch = make_orchestrator(shared_components, (0.02, 0.03, 0.04))
    assert low_orch.evaluate_applicant(inputs, risk_cutoff=0.12, review_margin=0.03).decision == "approve"

    mid_orch = make_orchestrator(shared_components, (0.11, 0.12, 0.13))
    assert mid_orch.evaluate_applicant(inputs, risk_cutoff=0.12, review_margin=0.03).decision == "review"

    high_orch = make_orchestrator(shared_components, (0.25, 0.30, 0.35))
    assert high_orch.evaluate_applicant(inputs, risk_cutoff=0.12, review_margin=0.03).decision == "reject"


def test_unusual_profile_goes_to_review(shared_components) -> None:
    orch = make_orchestrator(shared_components, (0.02, 0.02, 0.02))
    res = orch.evaluate_applicant(USE_CASES["uc4_unusual_ood_review"].inputs, risk_cutoff=0.25)
    assert res.is_unusual is True
    assert res.decision == "review"


def test_end_to_end_use_cases(production_orchestrator: IDSSOrchestrator) -> None:
    for uc in USE_CASES.values():
        res = production_orchestrator.evaluate_applicant(uc.inputs, risk_cutoff=uc.default_cutoff, review_margin=0.03)
        assert res.decision == uc.expected_outcome
        assert len(res.key_factors) == 5
        assert res.officer_summary and res.customer_letter


def test_streamlit_app_renders_all_cases() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "Source" / "app" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    # Changing the risk slider on Case 1 and switching to Case 2 must reset the slider to 12% and show Manual review
    at.sidebar.slider(key="risk_cutoff_pct").set_value(6.0).run()
    assert not at.exception
    at.sidebar.selectbox(key="selected_preset").select("uc2_borderline_review").run()
    assert not at.exception
    assert at.sidebar.slider(key="risk_cutoff_pct").value == 12.0
    assert any("Recommendation: Manual review" in m.value for m in at.markdown)

    # Switching across all cases and toggling missing-value checkboxes must never crash
    for preset in ("uc1_prime_approval", "uc2_borderline_review", "uc3_high_risk_reject", "uc4_unusual_ood_review", "custom"):
        at.sidebar.selectbox(key="selected_preset").select(preset).run()
        assert not at.exception

    at.sidebar.checkbox(key="income_declared").uncheck().run()
    assert not at.exception
    at.sidebar.checkbox(key="income_declared").check().run()
    assert not at.exception
    at.sidebar.checkbox(key="dependents_declared").uncheck().run()
    assert not at.exception
    at.sidebar.checkbox(key="dependents_declared").check().run()
    assert not at.exception

