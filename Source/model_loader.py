
from pathlib import Path
import json
import joblib

# Locate the Models folder relative to this Python file
MODELS_DIR = Path(__file__).resolve().parent.parent / "Models"


def load_models():
    """Load the trained classifiers and ensemble configuration."""

    models = {
        "logreg": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
        "tree": joblib.load(MODELS_DIR / "decision_tree.joblib"),
        "xgboost": joblib.load(MODELS_DIR / "xgboost.joblib"),
    }

    with open(MODELS_DIR / "ensemble_config.json", "r") as f:
        config = json.load(f)

    return models, config
