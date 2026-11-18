"""Unusual-case warning: flag applicants who look unlike the past applicants.

Why it matters: the models learned only from past applicants. For someone
unlike any of them, the predicted probability is an extrapolation and should
not be trusted on its own, so the decision step (Person D) sends the case to
review.

An applicant is unusual if any of three triggers fires:

1. ``far_from_past_applicants``: the mean distance to the ``k`` most similar
   training applicants is above a threshold. It uses the same comparison
   space as the similar cases (see similar.py). The threshold is the 99th
   percentile of this score on the validation split, so about 1% of ordinary
   applicants are flagged by design. It catches single extreme values (on the
   log scale) as well as rare combinations of ordinary values.
2. ``outside_observed_range``: a value is lower or higher than anything in
   the training data (e.g. age 19, while the data starts at 21).
3. ``lateness_code``: the credit record holds the codes 96/98, whose meaning
   is undocumented (EDA section 4.2).

Each flagged case gets plain-language reasons: the values that are extreme
compared with past applicants or, if every value is ordinary, the fields in
which the applicant differs most from the most similar past applicants.

The fitted settings are saved as readable JSON in
``Models/explain/unusual_detector.json``. Refit after any change to the data
or the preprocessing:

    python -m Source.explain.unusual     # fit, save, print sanity checks
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from Source import config
from Source.data.preprocessing import INPUT_FIELDS, TARGET, Preprocessor, load_split
from Source.explain.features import FIELDS, format_value
from Source.explain.similar import PROFILE_COLUMNS, SimilarCaseFinder
from Source.explain.types import UnusualCheck

DETECTOR_FILE = config.MODELS_DIR / "explain" / "unusual_detector.json"

TRIGGER_FAR = "far_from_past_applicants"
TRIGGER_RANGE = "outside_observed_range"
TRIGGER_CODE = "lateness_code"

# Labels for the profile columns that are not officer input fields.
PROFILE_LABELS = {name: info.label for name, info in FIELDS.items()} | {"IncomeMissing": "Income not declared"}


class UnusualCaseDetector:
    """Out-of-distribution check built on the similar-cases search."""

    def __init__(self, finder: SimilarCaseFinder, settings: dict):
        self.finder = finder
        self.settings = settings

    # -- fitting and persistence ------------------------------------------
    @classmethod
    def fit(cls, finder: SimilarCaseFinder, train_readable: pd.DataFrame, val_scaled: pd.DataFrame,
            k: int = 10, flag_rate: float = 0.01) -> "UnusualCaseDetector":
        """Set the distance threshold on validation applicants and record training ranges."""
        distances, _ = finder.neighbours(val_scaled, k)
        threshold = float(np.quantile(distances.mean(axis=1), 1 - flag_rate))

        values = train_readable[INPUT_FIELDS]
        fields = {
            name: {
                "min": float(values[name].min()),
                "max": float(values[name].max()),
                "q_low": float(values[name].quantile(0.001)),
                "q_high": float(values[name].quantile(0.999)),
            }
            for name in INPUT_FIELDS
        }
        coded = train_readable["LateCodeFlag"] == 1
        settings = {
            "description": "Unusual-case warning. A score above distance_threshold, a value outside "
                           "[min, max], or a lateness code makes an applicant unusual.",
            "k": k,
            "flag_rate": flag_rate,
            "distance_threshold": threshold,
            "profile_columns": PROFILE_COLUMNS,
            "fields": fields,
            "lateness_code_default_rate": float(train_readable.loc[coded, TARGET].mean()),
            "fitted_on": {"ranges_and_index": "train", "threshold": "val"},
        }
        return cls(finder, settings)

    def save(self, path=DETECTOR_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.settings, indent=2) + "\n")

    @classmethod
    def load(cls, finder: SimilarCaseFinder, path=DETECTOR_FILE) -> "UnusualCaseDetector":
        return cls(finder, json.loads(path.read_text()))

    # -- checking ---------------------------------------------------------
    @property
    def threshold(self) -> float:
        return self.settings["distance_threshold"]

    def scores(self, scaled: pd.DataFrame) -> np.ndarray:
        """Mean distance to the k most similar training applicants (higher = more unusual)."""
        distances, _ = self.finder.neighbours(scaled, self.settings["k"])
        return distances.mean(axis=1)

    def check(self, readable: pd.DataFrame, scaled: pd.DataFrame) -> list[UnusualCheck]:
        """One UnusualCheck per applicant, from the two tables of ``prepare_customer``."""
        distances, positions = self.finder.neighbours(scaled, self.settings["k"])
        scores = distances.mean(axis=1)
        points = self.finder._embed(scaled)

        results = []
        for i, (_, row) in enumerate(readable.iterrows()):
            # Values the applicant left blank were estimated, so they say nothing about the applicant.
            blank = {"MonthlyIncome": row["IncomeMissing"] == 1, "Dependents": row["DependentsMissing"] == 1}
            known = [name for name in INPUT_FIELDS if not blank.get(name, False)]
            triggers, reasons = [], []

            out_of_range = self._range_reasons(row, known)
            if out_of_range:
                triggers.append(TRIGGER_RANGE)
                reasons += out_of_range

            if scores[i] > self.threshold:
                triggers.append(TRIGGER_FAR)
                reasons.append(f"No past applicant has a similar profile (distance score {scores[i]:.2f}; "
                               f"99% of ordinary applicants score below {self.threshold:.2f}).")
                extreme = self._extreme_reasons(row, known)
                if extreme:
                    reasons += extreme
                elif not out_of_range:
                    reasons.append(self._combination_reason(points[i], positions[i]))

            if row["LateCodeFlag"] == 1:
                triggers.append(TRIGGER_CODE)
                reasons.append("The credit record contains the lateness codes 96/98, whose meaning is "
                               f"undocumented. {self.settings['lateness_code_default_rate']:.0%} of past "
                               "applicants with these codes defaulted.")

            results.append(UnusualCheck(is_unusual=bool(triggers), score=float(scores[i]),
                                        threshold=self.threshold, reasons=reasons, triggers=triggers))
        return results

    def check_inputs(self, preprocessor: Preprocessor, inputs: pd.DataFrame) -> list[UnusualCheck]:
        """Convenience: raw form inputs -> unusual-case checks."""
        readable, scaled = preprocessor.prepare_customer(inputs)
        return self.check(readable, scaled)

    # -- wording ----------------------------------------------------------
    def _range_reasons(self, row: pd.Series, known: list[str]) -> list[str]:
        reasons = []
        for name in known:
            limits, value, label = self.settings["fields"][name], row[name], FIELDS[name].label
            if value < limits["min"]:
                reasons.append(f"{label}: {format_value(name, value)}, below any past applicant "
                               f"(lowest: {format_value(name, limits['min'])}).")
            elif value > limits["max"]:
                reasons.append(f"{label}: {format_value(name, value)}, above any past applicant "
                               f"(highest: {format_value(name, limits['max'])}).")
        return reasons

    def _extreme_reasons(self, row: pd.Series, known: list[str]) -> list[str]:
        """Values inside the observed range but beyond 99.9% of past applicants."""
        reasons = []
        for name in known:
            limits, value, label = self.settings["fields"][name], row[name], FIELDS[name].label
            if limits["q_high"] < value <= limits["max"]:
                reasons.append(f"{label}: {format_value(name, value)}, higher than 99.9% of past applicants.")
            elif limits["min"] <= value < limits["q_low"]:
                reasons.append(f"{label}: {format_value(name, value)}, lower than 99.9% of past applicants.")
        return reasons

    def _combination_reason(self, point: np.ndarray, neighbour_positions: np.ndarray) -> str:
        """Name the two profile columns where the applicant differs most from its neighbours."""
        gaps = (point - self.finder.embedded[neighbour_positions].mean(axis=0)) ** 2
        top = [PROFILE_LABELS[PROFILE_COLUMNS[j]] for j in np.argsort(gaps)[::-1][:2]]
        return ("Each value is common on its own, but this combination is rare. It differs most from "
                f"the most similar past applicants in: {top[0].lower()} and {top[1].lower()}.")


def load_detector(finder: SimilarCaseFinder | None = None) -> UnusualCaseDetector:
    """Detector with the saved settings (builds the similar-cases index if not given)."""
    return UnusualCaseDetector.load(finder or SimilarCaseFinder.from_training_data())


# ---------------------------------------------------------------------------
# Fitting script with sanity checks (validation split only, never test)
# ---------------------------------------------------------------------------

SYNTHETIC_PROFILES = {
    "ordinary": dict(RevUtil=0.25, Age=48, Late30_59=0, Late60_89=0, Late90=0, MonthlyIncome=5500,
                     MonthlyDebt=1700, OpenLines=8, RealEstateLines=1, Dependents=1),
    "very high income": dict(RevUtil=0.10, Age=50, Late30_59=0, Late60_89=0, Late90=0, MonthlyIncome=400_000,
                             MonthlyDebt=3000, OpenLines=9, RealEstateLines=2, Dependents=2),
    "too young for the data": dict(RevUtil=0.40, Age=19, Late30_59=0, Late60_89=0, Late90=0, MonthlyIncome=1800,
                                   MonthlyDebt=300, OpenLines=2, RealEstateLines=0, Dependents=0),
    "many late payments": dict(RevUtil=0.95, Age=35, Late30_59=10, Late60_89=9, Late90=15, MonthlyIncome=3000,
                               MonthlyDebt=1200, OpenLines=4, RealEstateLines=0, Dependents=2),
    "young with many mortgages": dict(RevUtil=0.20, Age=23, Late30_59=0, Late60_89=0, Late90=0, MonthlyIncome=2500,
                                      MonthlyDebt=900, OpenLines=12, RealEstateLines=8, Dependents=0),
    "lateness codes": dict(RevUtil=1.0, Age=30, Late30_59=np.nan, Late60_89=np.nan, Late90=np.nan,
                           MonthlyIncome=2200, MonthlyDebt=0, OpenLines=0, RealEstateLines=0, Dependents=0,
                           LateCodeFlag=1),
}


def main() -> None:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from Source.explain.standins import load_standin_pipeline

    finder = SimilarCaseFinder.from_training_data()
    train_readable = load_split("train", "readable")
    val_readable, val_scaled = load_split("val", "readable"), load_split("val", "model")

    detector = UnusualCaseDetector.fit(finder, train_readable, val_scaled)
    detector.save()
    print(f"Saved {DETECTOR_FILE.relative_to(config.ROOT)}; distance threshold {detector.threshold:.3f}\n")

    checks = detector.check(val_readable, val_scaled)
    flagged = np.array([c.is_unusual for c in checks])
    print(f"Validation applicants flagged: {flagged.mean():.2%} ({flagged.sum()} of {len(flagged):,})")
    for trigger in (TRIGGER_FAR, TRIGGER_RANGE, TRIGGER_CODE):
        print(f"  {trigger:26s} {np.mean([trigger in c.triggers for c in checks]):.2%}")

    # Are the models less reliable on flagged applicants? (stand-in models for now)
    probs = load_standin_pipeline().predict(val_readable[INPUT_FIELDS + ["LateCodeFlag"]])["combined"]
    y = val_readable[TARGET].to_numpy()
    print("\nStand-in combined model on validation applicants:")
    for name, mask in [("not flagged", ~flagged), ("flagged", flagged)]:
        print(f"  {name:12s} n={mask.sum():6,d}  default rate {y[mask].mean():.1%}  "
              f"mean predicted {probs[mask].mean():.1%}  AUC {roc_auc_score(y[mask], probs[mask]):.3f}  "
              f"Brier {brier_score_loss(y[mask], probs[mask]):.4f}")

    print("\nHand-made profiles:")
    prep = Preprocessor.load()
    for name, profile in SYNTHETIC_PROFILES.items():
        result = detector.check_inputs(prep, pd.DataFrame([profile]))[0]
        print(f"- {name}: unusual={result.is_unusual}, score {result.score:.2f}, triggers {result.triggers}")
        for reason in result.reasons:
            print(f"    {reason}")


if __name__ == "__main__":
    main()
