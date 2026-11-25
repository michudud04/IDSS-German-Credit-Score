"""Evaluate the decision policy on the validation split (never the test split).

    python -m Source.decision.evaluate              # validation, prints tables
    python -m Source.decision.evaluate --split test # final numbers for the report, once settled

For each slider setting it reports how many applicants are approved, sent to
review and rejected, how many defaulters slip through the automatic approvals,
and the cost of the automatic decisions in units of one lost good customer.

Validation at the default setting (cost ratio 4, cut-off 20%): 86.3% approved,
8.0% review (grey zone 3.5%, debt above income 3.2%, unusual 1.5%, models
disagree 1.4%; an applicant can have several reasons), 5.7% rejected.
Defaulters among the automatic approvals: 3.2% (6.7% overall).
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from Source.data.preprocessing import INPUT_FIELDS, TARGET, load_split
from Source.decision.engine import DecisionEngine
from Source.decision.policy import RiskPolicy

COST_RATIOS = [1, 2, 3, 4, 6, 8, 10, 15, 20]


def applicant_forms(readable: pd.DataFrame) -> pd.DataFrame:
    """Prepared split -> what an officer would have typed (estimated values back to blanks)."""
    form = readable[INPUT_FIELDS + ["LateCodeFlag"]].copy()
    form.loc[readable["IncomeMissing"] == 1, "MonthlyIncome"] = np.nan
    form.loc[readable["DependentsMissing"] == 1, "Dependents"] = np.nan
    return form


def summarise(decisions, y: np.ndarray, policy: RiskPolicy) -> dict:
    outcome = np.array([d.outcome for d in decisions])
    approved, review, rejected = (outcome == o for o in ("approve", "review", "reject"))
    auto = ~review
    # Cost of the automatic decisions only: a wrong approval costs cost_ratio, a wrong rejection 1.
    cost = (approved & (y == 1)).sum() * policy.cost_ratio + (rejected & (y == 0)).sum()
    return {
        "cost_ratio": policy.cost_ratio,
        "cutoff": policy.cutoff,
        "approve %": approved.mean() * 100,
        "review %": review.mean() * 100,
        "reject %": rejected.mean() * 100,
        "default rate approved %": y[approved].mean() * 100 if approved.any() else np.nan,
        "default rate review %": y[review].mean() * 100 if review.any() else np.nan,
        "default rate rejected %": y[rejected].mean() * 100 if rejected.any() else np.nan,
        "defaulters caught %": (y[~approved] == 1).sum() / max((y == 1).sum(), 1) * 100,
        "auto cost / 1k": cost / auto.sum() * 1000 if auto.any() else np.nan,
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    split = parser.parse_args().split

    readable, scaled = load_split(split, "readable"), load_split(split, "model")
    forms, y = applicant_forms(readable), readable[TARGET].to_numpy()

    engine = DecisionEngine.from_saved()
    unusual = engine.detector.check(readable, scaled)  # computed once, reused for every setting

    rows, default_decisions = [], None
    for ratio in COST_RATIOS:
        policy = RiskPolicy(cost_ratio=ratio)
        decisions = engine.assess(forms, policy, unusual=unusual)
        rows.append(summarise(decisions, y, policy))
        if ratio == RiskPolicy().cost_ratio:
            default_decisions = decisions

    pd.set_option("display.width", 200)
    print(f"Slider sweep on the {split} split ({len(y):,} applicants, default rate {y.mean():.2%}):\n")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print(f"\nWhy applicants go to review at the default setting (cost ratio {RiskPolicy().cost_ratio:g}); "
          "an applicant can have several reasons:")
    review = [d for d in default_decisions if d.outcome == "review"]
    counts = Counter(r.code for d in review for r in d.reasons if r.effect == "review")
    for code, n in counts.most_common():
        print(f"  {code:26s} {n:6,d}  ({n / len(y):.2%} of all applicants)")
    print(f"  {'total in review':26s} {len(review):6,d}  ({len(review) / len(y):.2%})")

    rules = Counter(r.code for d in default_decisions for r in d.reasons if r.source == "bank_rule")
    print("\nBank rules fired (all outcomes):")
    for code, n in rules.most_common():
        print(f"  {code:26s} {n:6,d}  ({n / len(y):.2%})")


if __name__ == "__main__":
    main()
