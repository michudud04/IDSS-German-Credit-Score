"""Load the final models (Person B) as a RiskPipeline (Person C's interface).

``Source.model_loader.load_models`` returns the fitted classifiers and the
ensemble configuration (weights, features, which table each model uses).
Wrapping them with ``SklearnAdapter`` gives the same pipeline the explanation
modules use, so the decision is made on exactly the explained probability.
"""

from __future__ import annotations

from Source.data.preprocessing import Preprocessor
from Source.explain.pipeline import RiskPipeline, SklearnAdapter
from Source.model_loader import load_models


def load_risk_pipeline(preprocessor: Preprocessor | None = None) -> RiskPipeline:
    models, config = load_models()
    adapters = {
        name: SklearnAdapter(model, config["features"], config["feature_representation"][name])
        for name, model in models.items()
    }
    return RiskPipeline(preprocessor or Preprocessor.load(), adapters, config["weights"])
