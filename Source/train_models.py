"""Train the three credit risk models and save them to Models/."""

import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Source import config
from Source.data.preprocessing import SELECTED, TARGET, load_split

ENSEMBLE_WEIGHTS = {"logreg": 0.05, "tree": 0.05, "xgboost": 0.90}
FEATURE_REPRESENTATION = {"logreg": "model", "tree": "readable", "xgboost": "readable"}
DEFAULT_THRESHOLD = 0.20


def train_all() -> dict:
    train_readable = load_split("train", "readable")
    train_model = load_split("train", "model")
    y_train = train_readable[TARGET]

    logreg = LogisticRegression(C=1.0, max_iter=1000, random_state=config.RANDOM_STATE)
    logreg.fit(train_model[SELECTED], y_train)

    tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=200, random_state=config.RANDOM_STATE)
    tree.fit(train_readable[SELECTED], y_train)

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(train_readable[SELECTED], y_train)

    return {"logreg": logreg, "tree": tree, "xgboost": xgb}


def evaluate_models(models: dict, split: str = "val") -> dict:
    readable = load_split(split, "readable")
    scaled = load_split(split, "model")
    y = readable[TARGET]

    p_lr = models["logreg"].predict_proba(scaled[SELECTED])[:, 1]
    p_tree = models["tree"].predict_proba(readable[SELECTED])[:, 1]
    p_xgb = models["xgboost"].predict_proba(readable[SELECTED])[:, 1]
    p_ens = (
        ENSEMBLE_WEIGHTS["logreg"] * p_lr
        + ENSEMBLE_WEIGHTS["tree"] * p_tree
        + ENSEMBLE_WEIGHTS["xgboost"] * p_xgb
    )

    metrics = {}
    for name, probs in [("logreg", p_lr), ("tree", p_tree), ("xgboost", p_xgb), ("ensemble", p_ens)]:
        metrics[name] = {
            "roc_auc": round(float(roc_auc_score(y, probs)), 4),
            "brier": round(float(brier_score_loss(y, probs)), 4),
        }
    return metrics


def save_models(models: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["logreg"], config.MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(models["tree"], config.MODELS_DIR / "decision_tree.joblib")
    joblib.dump(models["xgboost"], config.MODELS_DIR / "xgboost.joblib")

    ensemble_cfg = {
        "weights": ENSEMBLE_WEIGHTS,
        "features": SELECTED,
        "feature_representation": FEATURE_REPRESENTATION,
        "decision_threshold": DEFAULT_THRESHOLD,
    }
    (config.MODELS_DIR / "ensemble_config.json").write_text(json.dumps(ensemble_cfg, indent=4) + "\n")


if __name__ == "__main__":
    trained = train_all()
    for split_name in ("val", "test"):
        print(f"=== {split_name.upper()} ===")
        for model_name, m in evaluate_models(trained, split_name).items():
            print(f"  {model_name:10s}  ROC-AUC: {m['roc_auc']:.4f}  Brier: {m['brier']:.4f}")
    save_models(trained)
    print(f"Saved models to {config.MODELS_DIR}")
