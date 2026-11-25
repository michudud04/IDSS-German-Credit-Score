"""The prediction pipeline seen as one function of the applicant's input fields.

    RiskPipeline.predict(inputs) -> probability of default per model + combined

Every explanation module calls only this class, so it does not matter which
feature space each model uses or whether it is calibrated.

Interface for the models (Person B): any object with

    predict_default_proba(readable: DataFrame, scaled: DataFrame) -> 1-D array

where ``readable`` and ``scaled`` are the two tables returned by
``Preprocessor.prepare_customer``. A fitted scikit-learn / XGBoost classifier
can be wrapped with ``SklearnAdapter`` instead of writing a class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from Source.data.preprocessing import Preprocessor

MODEL_NAMES = ("logreg", "xgboost", "tree")
COMBINED = "combined"


class ProbabilityModel(Protocol):
    def predict_default_proba(self, readable: pd.DataFrame, scaled: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class SklearnAdapter:
    """Wraps any classifier with ``predict_proba`` and the columns it was trained on."""

    estimator: object
    features: list[str]
    space: Literal["readable", "model"]  # which prepared table the estimator expects

    def predict_default_proba(self, readable: pd.DataFrame, scaled: pd.DataFrame) -> np.ndarray:
        table = readable if self.space == "readable" else scaled
        return self.estimator.predict_proba(table[self.features])[:, 1]


class RiskPipeline:
    """inputs (cleaned schema, one row per applicant) -> probabilities of default."""

    def __init__(self, preprocessor: Preprocessor, models: dict[str, ProbabilityModel],
                 weights: dict[str, float] | None = None):
        self.preprocessor = preprocessor
        self.models = models
        # Combination = weighted average of probabilities (equal weights by default).
        # A weighted average keeps the merged SHAP factors exact (see factors.py).
        self.weights = weights or {name: 1.0 for name in models}
        total = sum(self.weights.values())
        self.weights = {name: w / total for name, w in self.weights.items()}

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """One column per model plus ``combined``, one row per applicant."""
        readable, scaled = self.preprocessor.prepare_customer(inputs)
        probs = pd.DataFrame(
            {name: model.predict_default_proba(readable, scaled) for name, model in self.models.items()},
            index=inputs.index,
        )
        probs[COMBINED] = sum(self.weights[name] * probs[name] for name in self.models)
        return probs

    def predict_combined(self, inputs: pd.DataFrame) -> np.ndarray:
        return self.predict(inputs)[COMBINED].to_numpy()
