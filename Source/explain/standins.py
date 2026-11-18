"""TEMPORARY stand-in models so the explanation modules can be built before
Person B delivers the real, tuned and calibrated models.

They are deliberately simple and untuned, trained without class weights so
that their probabilities are roughly honest. They are cached under
``data/cache/`` (git-ignored) and must never be shipped or evaluated as the
system's models.

    python -m Source.explain.standins     # train, cache, print validation AUC
"""

from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from Source import config
from Source.data.preprocessing import INPUT_FIELDS, SELECTED, TARGET, Preprocessor, load_split
from Source.explain.pipeline import COMBINED, RiskPipeline, SklearnAdapter

STANDIN_FILE = config.DATA_DIR / "cache" / "standin_models.joblib"


def train_standins() -> dict[str, SklearnAdapter]:
    readable, scaled = load_split("train", "readable"), load_split("train", "model")
    y = readable[TARGET]

    logreg = LogisticRegression(max_iter=2000).fit(scaled[SELECTED], y)
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=200,
                                  random_state=config.RANDOM_STATE).fit(readable[SELECTED], y)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=config.RANDOM_STATE, n_jobs=4)
    xgb.fit(readable[SELECTED], y)

    return {
        "logreg": SklearnAdapter(logreg, SELECTED, "model"),
        "xgboost": SklearnAdapter(xgb, SELECTED, "readable"),
        "tree": SklearnAdapter(tree, SELECTED, "readable"),
    }


def load_standin_pipeline(retrain: bool = False) -> RiskPipeline:
    """Stand-in pipeline with equal weights; trains and caches the models on first use."""
    if retrain or not STANDIN_FILE.exists():
        STANDIN_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(train_standins(), STANDIN_FILE)
    return RiskPipeline(Preprocessor.load(), joblib.load(STANDIN_FILE))


def main() -> None:
    pipeline = load_standin_pipeline(retrain=True)
    # Sanity check on validation applicants, fed through the same path as a new applicant.
    val = load_split("val", "readable")
    form = val[INPUT_FIELDS + ["LateCodeFlag"]].copy()
    form.loc[val["IncomeMissing"] == 1, "MonthlyIncome"] = float("nan")  # imputed in the table, blank on a form
    form.loc[val["DependentsMissing"] == 1, "Dependents"] = float("nan")
    probs = pipeline.predict(form)
    for name in list(pipeline.models) + [COMBINED]:
        print(f"{name:8s} val AUC {roc_auc_score(val[TARGET], probs[name]):.4f}  "
              f"mean prob {probs[name].mean():.4f}")
    print(f"actual default rate {val[TARGET].mean():.4f}")


if __name__ == "__main__":
    main()
