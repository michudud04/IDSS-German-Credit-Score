"""Key factors: which of the applicant's values pushed the risk up or down.

Method: exact Shapley values (SHAP) on the applicant's own inputs
- The explained "model" is the whole pipeline from the officer's 11 input
  fields (10 form fields plus the lateness-code flag) to each model's
  probability of default. Derived features such as DebtRatio or
  SavingCapacity are therefore credited to the inputs they come from, and the
  officer reads "monthly debt of $3,400" instead of three related ratios.
- Reference: 200 randomly drawn training applicants (seed 42), with values
  they did not declare left blank, so they represent the real population.
  A factor of +4 points means: this value raised the probability of default
  by 4 percentage points compared with the reference applicants. The
  reference value plus all factors equals the prediction exactly.
- Exact, not sampled: with 11 inputs there are 2^11 = 2,048 combinations,
  which ``shap.ExactExplainer`` evaluates in about 0.5 s per applicant. There
  is no sampling noise, so the same applicant always gets the same factors.
- Merged across models: the three models are explained together with the
  same reference rows. Shapley values are linear in the model, so the
  ensemble's weighted average of the three models' factors is exactly the
  explanation of the combined probability the decision uses.

Usage (the app calls ``explain_factors``):

    factors = explain_factors(pipeline, inputs)[0]   # top 5 KeyFactors for the first applicant

    python -m Source.explain.factors                  # examples and sanity checks
"""

from __future__ import annotations

import weakref

import numpy as np
import pandas as pd
import shap

from Source import config
from Source.data.preprocessing import LATE_COLS, load_split
from Source.explain.features import EXPLAINED_FIELDS, FIELDS, format_value
from Source.explain.pipeline import COMBINED, RiskPipeline
from Source.explain.types import KeyFactor

N_REFERENCE = 200   # reference applicants; 100 doubled the noise in the factors, 200 takes ~0.5 s
TOP_N = 5           # factors shown per applicant (the brief asks for the top 3-5)
MIN_EFFECT = 1e-6   # contributions smaller than this are not reported as factors


def to_form(readable: pd.DataFrame) -> pd.DataFrame:
    """Prepared rows -> what the applicant entered: estimated values back to blanks."""
    form = readable[EXPLAINED_FIELDS].copy()
    form.loc[readable["IncomeMissing"] == 1, "MonthlyIncome"] = np.nan
    form.loc[readable["DependentsMissing"] == 1, "Dependents"] = np.nan
    form.loc[readable["LateCodeFlag"] == 1, LATE_COLS] = np.nan
    return form


def as_explained_frame(inputs: pd.DataFrame) -> pd.DataFrame:
    """Applicant inputs (INPUT_FIELDS, optional LateCodeFlag) in the explained column order."""
    frame = inputs.copy()
    if "LateCodeFlag" not in frame:
        frame["LateCodeFlag"] = 0.0
    return frame[EXPLAINED_FIELDS].astype(float)


class FactorExplainer:
    """Exact SHAP values of every model and of the combined probability, per input field."""

    def __init__(self, pipeline: RiskPipeline, reference_forms: pd.DataFrame):
        self.pipeline = pipeline
        self.model_names = list(pipeline.models)
        self.weights = np.array([pipeline.weights[name] for name in self.model_names])
        reference = as_explained_frame(reference_forms).to_numpy(float)
        masker = shap.maskers.Independent(reference, max_samples=len(reference))
        self.explainer = shap.ExactExplainer(self._predict, masker)

    @classmethod
    def from_training_data(cls, pipeline: RiskPipeline, n_reference: int = N_REFERENCE,
                           seed: int = config.RANDOM_STATE) -> "FactorExplainer":
        sample = load_split("train", "readable").sample(n_reference, random_state=seed)
        return cls(pipeline, to_form(sample))

    def _predict(self, values: np.ndarray) -> np.ndarray:
        """(rows x 11 inputs) -> (rows x models) probabilities of default."""
        frame = pd.DataFrame(values, columns=EXPLAINED_FIELDS)
        return self.pipeline.predict(frame)[self.model_names].to_numpy()

    def contributions(self, inputs: pd.DataFrame) -> list[pd.DataFrame]:
        """Full table per applicant: one row per input field (plus 'reference'), one column per
        model and 'combined'. Each column sums to that model's prediction."""
        explanation = self.explainer(as_explained_frame(inputs).to_numpy(float), silent=True)
        tables = []
        for values, base in zip(explanation.values, explanation.base_values):
            table = pd.DataFrame(values, index=EXPLAINED_FIELDS, columns=self.model_names)
            table.loc["reference"] = base
            table[COMBINED] = table[self.model_names].to_numpy() @ self.weights
            tables.append(table)
        return tables

    def explain(self, inputs: pd.DataFrame, top_n: int | None = TOP_N) -> list[list[KeyFactor]]:
        """Top factors per applicant, strongest first, merged across the models."""
        readable, _ = self.pipeline.preprocessor.prepare_customer(inputs)
        entered = as_explained_frame(inputs)
        results = []
        for i, table in enumerate(self.contributions(inputs)):
            fields = table.drop(index="reference")
            fields = fields[fields[COMBINED].abs() > MIN_EFFECT]
            order = fields[COMBINED].abs().sort_values(ascending=False).index[:top_n]
            results.append([
                self._factor(name, fields.loc[name], entered.iloc[i][name], readable.iloc[i][name])
                for name in order
            ])
        return results

    def _factor(self, name: str, effects: pd.Series, entered: float, used: float) -> KeyFactor:
        info = FIELDS[name]
        blank = pd.isna(entered)
        if blank and name in LATE_COLS:
            value_text = "not available (lateness code)"
        else:
            # A blank income or dependents field was estimated; show the estimate the models used.
            value_text = format_value(name, used if blank else entered, estimated=blank)
        contribution = float(effects[COMBINED])
        return KeyFactor(
            feature=name,
            label=info.label,
            value=None if blank else float(entered),
            value_text=value_text,
            direction="increases_risk" if contribution > 0 else "decreases_risk",
            contribution=contribution,
            per_model={model: float(effects[model]) for model in self.model_names},
            actionable=info.actionable,
            sensitive=info.sensitive,
        )


# One explainer per pipeline, built on first use (building it reads the training split).
_EXPLAINERS: "weakref.WeakKeyDictionary[RiskPipeline, FactorExplainer]" = weakref.WeakKeyDictionary()


def get_explainer(pipeline: RiskPipeline) -> FactorExplainer:
    if pipeline not in _EXPLAINERS:
        _EXPLAINERS[pipeline] = FactorExplainer.from_training_data(pipeline)
    return _EXPLAINERS[pipeline]


def explain_factors(pipeline: RiskPipeline, inputs: pd.DataFrame, top_n: int | None = TOP_N) -> list[list[KeyFactor]]:
    """Entry point used by the app: top key factors for each applicant (row) in ``inputs``."""
    return get_explainer(pipeline).explain(inputs, top_n=top_n)


# ---------------------------------------------------------------------------
# Examples and sanity checks (validation split only)
# ---------------------------------------------------------------------------

EXAMPLES = {
    "low risk": dict(RevUtil=0.18, Age=44, Late30_59=0, Late60_89=0, Late90=0, MonthlyIncome=6800,
                     MonthlyDebt=1450, OpenLines=7, RealEstateLines=1, Dependents=1),
    "borderline": dict(RevUtil=0.75, Age=42, Late30_59=1, Late60_89=0, Late90=0, MonthlyIncome=4800,
                       MonthlyDebt=1850, OpenLines=7, RealEstateLines=1, Dependents=1),
    "high risk": dict(RevUtil=1.15, Age=33, Late30_59=2, Late60_89=1, Late90=2, MonthlyIncome=3100,
                      MonthlyDebt=2600, OpenLines=5, RealEstateLines=0, Dependents=1),
    "income not declared": dict(RevUtil=0.45, Age=38, Late30_59=1, Late60_89=0, Late90=0, MonthlyIncome=np.nan,
                                MonthlyDebt=1800, OpenLines=6, RealEstateLines=1, Dependents=np.nan),
}


def main() -> None:
    import time
    import warnings

    from scipy.stats import spearmanr

    from Source.decision.models import load_risk_pipeline

    warnings.filterwarnings("ignore", category=UserWarning)
    pipeline = load_risk_pipeline()
    explainer = FactorExplainer.from_training_data(pipeline)

    for name, profile in EXAMPLES.items():
        inputs = pd.DataFrame([profile])
        start = time.perf_counter()
        factors = explainer.explain(inputs)[0]
        seconds = time.perf_counter() - start
        reference = explainer.contributions(inputs)[0].loc["reference", COMBINED]
        prob = pipeline.predict(inputs)[COMBINED].iloc[0]
        print(f"\n{name}: probability {prob:.1%} (reference applicants {reference:.1%}), {seconds:.2f}s")
        for f in factors:
            flags = " [sensitive]" if f.sensitive else (" [actionable]" if f.actionable else "")
            models = ", ".join(f"{m} {v * 100:+.1f}" for m, v in f.per_model.items())
            print(f"  {f.contribution * 100:+5.1f} pts  {f.label}: {f.value_text}{flags}   ({models})")

    # Checks on validation applicants.
    sample = load_split("val", "readable").sample(100, random_state=config.RANDOM_STATE)
    forms = to_form(sample)
    tables = explainer.contributions(forms)
    probs = pipeline.predict(forms)
    error = max(abs(t[COMBINED].sum() - p) for t, p in zip(tables, probs[COMBINED]))
    print(f"\nAdditivity on {len(tables)} validation applicants: max |reference + factors - prediction| = {error:.1e}")

    # Do the models agree on which inputs matter? Rank correlation of their factors per applicant.
    names = explainer.model_names
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    for a, b in pairs:
        rho = np.nanmean([spearmanr(t.drop(index="reference")[a], t.drop(index="reference")[b])[0] for t in tables])
        print(f"Mean rank correlation of factors, {a} vs {b}: {rho:.2f}")

    importance = pd.concat([t.drop(index="reference")[COMBINED].abs() for t in tables], axis=1).mean(axis=1)
    print("\nMean absolute effect on the combined probability (points):")
    print((importance.sort_values(ascending=False) * 100).round(2).to_string())


if __name__ == "__main__":
    main()
