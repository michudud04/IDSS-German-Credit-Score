"""Similar past applicants: the nearest training customers and how they turned out.

How similarity is measured
- Each applicant is described by the full customer profile: the 10 fields the
  officer enters (including age and dependents) plus two data-quality flags,
  "income not declared" and "lateness code in the records".
- The comparison uses the model-scale table from preprocessing, where the
  skewed amounts are log-transformed. A gap between $3,000 and $6,000 of
  income therefore counts as much as one between $6,000 and $12,000.
- Every profile column is standardised on the training data, so no single
  column dominates the distance just because of its unit.
- The flags get large weights automatically: a rare flag is many standard
  deviations away from 0. Applicants with lateness codes are therefore only
  matched with each other, and declared income mostly with declared income.

What the officer sees
- The ``n_shown`` closest past applicants with their outcome.
- The default rate among the ``neighbourhood_size`` closest applicants,
  next to the overall rate. With a base rate of 6.7%, five neighbours
  almost always show zero defaults, so the larger group is what carries
  information.

Only the training split is searched, so validation and test applicants
never see their own outcome.

    python -m Source.explain.similar     # example output and a sanity check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from Source.data.preprocessing import INPUT_FIELDS, TARGET, Preprocessor, load_split
from Source.explain.types import SimilarCase, SimilarCases

PROFILE_COLUMNS = INPUT_FIELDS + ["IncomeMissing", "LateCodeFlag"]


class SimilarCaseFinder:
    """Nearest-neighbour search over the training applicants."""

    def __init__(self, train_readable: pd.DataFrame, train_scaled: pd.DataFrame,
                 n_shown: int = 5, neighbourhood_size: int = 50):
        self.n_shown = n_shown
        self.neighbourhood_size = neighbourhood_size

        # The numeric profile columns are already standardised by the preprocessing,
        # so this scaler only changes the two 0/1 flags.
        self.scaler = StandardScaler().fit(train_scaled[PROFILE_COLUMNS])
        self.embedded = self._embed(train_scaled)  # training profiles in the comparison space
        self.index = NearestNeighbors().fit(self.embedded)

        self.case_ids = train_scaled.index.to_numpy()
        self.outcomes = train_readable[TARGET].to_numpy().astype(bool)
        self.profiles = train_readable[INPUT_FIELDS].to_numpy(float)
        self.income_estimated = train_readable["IncomeMissing"].to_numpy().astype(bool)
        self.base_rate = float(self.outcomes.mean())

    @classmethod
    def from_training_data(cls, **kwargs) -> "SimilarCaseFinder":
        """Build the search index from the prepared training split (takes about a second)."""
        return cls(load_split("train", "readable"), load_split("train", "model"), **kwargs)

    def _embed(self, scaled: pd.DataFrame) -> np.ndarray:
        """Model-scale table -> points in the comparison space (one row per applicant)."""
        return self.scaler.transform(scaled[PROFILE_COLUMNS])

    def neighbours(self, scaled: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Distances and row positions of the ``k`` nearest training applicants, per query row."""
        return self.index.kneighbors(self._embed(scaled), n_neighbors=k)

    def find(self, scaled: pd.DataFrame) -> list[SimilarCases]:
        """Similar cases for each applicant in a model-scale table (from ``prepare_customer``)."""
        distances, positions = self.neighbours(scaled, self.neighbourhood_size)
        results = []
        for dist_row, pos_row in zip(distances, positions):
            shown = [
                SimilarCase(
                    case_id=int(self.case_ids[pos]),
                    distance=float(dist),
                    inputs={field: float(value) for field, value in zip(INPUT_FIELDS, self.profiles[pos])},
                    defaulted=bool(self.outcomes[pos]),
                    income_estimated=bool(self.income_estimated[pos]),
                )
                for dist, pos in zip(dist_row[:self.n_shown], pos_row[:self.n_shown])
            ]
            results.append(SimilarCases(
                neighbours=shown,
                neighbourhood_size=self.neighbourhood_size,
                neighbourhood_default_rate=float(self.outcomes[pos_row].mean()),
                base_rate=self.base_rate,
            ))
        return results

    def find_for_inputs(self, preprocessor: Preprocessor, inputs: pd.DataFrame) -> list[SimilarCases]:
        """Convenience: raw form inputs -> similar cases."""
        _, scaled = preprocessor.prepare_customer(inputs)
        return self.find(scaled)


def main() -> None:
    import time

    from sklearn.metrics import roc_auc_score

    start = time.perf_counter()
    finder = SimilarCaseFinder.from_training_data()
    print(f"Index built in {time.perf_counter() - start:.2f}s")

    prep = Preprocessor.load()
    applicant = pd.DataFrame([{
        "RevUtil": 0.85, "Age": 29, "Late30_59": 2, "Late60_89": 0, "Late90": 0, "MonthlyIncome": 3200,
        "MonthlyDebt": 1500, "OpenLines": 5, "RealEstateLines": 0, "Dependents": 1,
    }])
    start = time.perf_counter()
    result = finder.find_for_inputs(prep, applicant)[0]
    print(f"One query in {time.perf_counter() - start:.3f}s\n")
    print(pd.DataFrame([{**c.inputs, "distance": c.distance, "defaulted": c.defaulted} for c in result.neighbours]))
    print(f"\nDefault rate among the {result.neighbourhood_size} most similar: "
          f"{result.neighbourhood_default_rate:.1%} (all applicants: {result.base_rate:.1%})")

    # Sanity check: if neighbours are meaningful, their default rate should rank
    # validation applicants by risk well above chance (AUC 0.5).
    val_readable, val_scaled = load_split("val", "readable"), load_split("val", "model")
    start = time.perf_counter()
    rates = [r.neighbourhood_default_rate for r in finder.find(val_scaled)]
    print(f"\nValidation ({len(val_scaled):,} applicants, {time.perf_counter() - start:.1f}s): "
          f"AUC of the neighbourhood default rate = {roc_auc_score(val_readable[TARGET], rates):.3f}")


if __name__ == "__main__":
    main()
