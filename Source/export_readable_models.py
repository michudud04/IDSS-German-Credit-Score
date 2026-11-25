"""Export readable model weights and feature importances to Models/."""

import json
import math

from Source import config
from Source.model_loader import load_models


def export_all() -> None:
    models, cfg = load_models()
    features = cfg["features"]
    out_dir = config.MODELS_DIR

    # Logistic regression standardized weights and odds multipliers
    logreg = models["logreg"]
    coef_items = sorted(
        [
            {
                "feature": feat,
                "standardized_weight": round(float(w), 4),
                "odds_multiplier": round(float(math.exp(w)), 4),
            }
            for feat, w in zip(features, logreg.coef_[0])
        ],
        key=lambda x: abs(x["standardized_weight"]),
        reverse=True,
    )
    lr_payload = {
        "model": "LogisticRegression",
        "feature_space": "model (log1p + StandardScaler)",
        "intercept": round(float(logreg.intercept_[0]), 4),
        "coefficients": coef_items,
    }
    (out_dir / "logistic_regression_weights.json").write_text(json.dumps(lr_payload, indent=2) + "\n")

    # XGBoost feature importances
    xgb = models["xgboost"]
    xgb_items = sorted(
        [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in zip(features, xgb.feature_importances_)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )
    xgb_payload = {
        "model": "XGBClassifier",
        "feature_space": "readable",
        "feature_importances": xgb_items,
    }
    (out_dir / "xgboost_feature_importance.json").write_text(json.dumps(xgb_payload, indent=2) + "\n")


if __name__ == "__main__":
    export_all()
    print(f"Saved readable model summaries to {config.MODELS_DIR}")
