"""What-if advice: which changes to the applicant's own situation would lower the risk enough.

Rules (team decisions)
- Only actionable fields change, each in its sensible direction: revolving
  utilisation (lower), monthly debt payments (lower), monthly income (higher)
  and open credit lines (fewer). Age, dependents, past late payments,
  real-estate loans and the lateness-code flag are never changed.
- No limits on the size of a change. The search still needs finite ranges:
  utilisation and debt go down to 0, open credit lines down to the number of
  real-estate loans (a mortgage is an open credit line), and income up to
  three times the current income.
- Income is only advised when the applicant declared it. An undeclared
  income was estimated by us, and "raise it from $X" would rest on that guess.

Search
- Each field gets candidate values in round steps: utilisation in 5-point
  steps, debt cut by 10%, 20%, ..., 100%, income raised by 10% to 200%,
  open lines reduced one at a time. Amounts are rounded to $10.
- Every single change and every pair of changes is run through the full
  pipeline (preprocessing, 3 models, ensemble), so derived values such as the
  debt ratio are recomputed. That is about 1,000 candidates in about 0.1 s.
- effort = sum of the relative changes, e.g. utilisation 80% -> 40% = 0.5,
  income +20% = 0.2, closing 2 of 8 lines = 0.25.

What is reported (at most 4 options)
- "passes" uses the decision step's own rule (D's RiskPolicy.passes): the new
  probability is below the automatic-approval limit of the risk slider.
- For each field, the smallest single change that passes.
- The cheapest pair that passes, if it needs less effort than every single
  change that passes (or no single change passes).
- If nothing passes: for each field the single change that lowers the risk
  most, and the best pair, all marked as not enough.
- Options that do not pass are only reported if they lower the risk by at
  least 0.5 points. The tree models are step functions, so some changes do
  nothing or even raise the risk. Options that pass are always reported.
- An applicant who already passes gets no advice.

Usage (the app calls ``generate_advice``):

    advice = generate_advice(pipeline, inputs, cutoff=0.12, review_margin=0.03)[0]

    python -m Source.explain.advice      # examples and a validation check
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Source.data.preprocessing import INCOME_FLOOR
from Source.explain.features import EXPLAINED_FIELDS, FIELDS, as_explained_frame, format_value
from Source.explain.pipeline import COMBINED, RiskPipeline
from Source.explain.types import AdviceItem, FeatureChange

ADVICE_FIELDS = ["RevUtil", "MonthlyDebt", "MonthlyIncome", "OpenLines"]
UTIL_STEP = 0.05
DEBT_CUTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
INCOME_RAISES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
MIN_IMPROVEMENT = 0.005  # at least 0.5 points lower probability of default
MAX_ITEMS = 4

# Safety net for the team rule "never advise on age, dependents or past events".
assert all(FIELDS[f].actionable and not FIELDS[f].sensitive for f in ADVICE_FIELDS)


@dataclass(frozen=True)
class Option:
    """A candidate: one or two simultaneous changes and their total effort."""

    changes: tuple[FeatureChange, ...]
    effort: float

    @property
    def fields(self) -> set[str]:
        return {change.feature for change in self.changes}


def _known(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _change(field: str, old: float, new: float) -> FeatureChange:
    label = FIELDS[field].label
    if field == "OpenLines":
        closed = int(old - new)
        text = f"Close {closed} open credit line{'s' if closed > 1 else ''} (from {old:.0f} to {new:.0f})"
    else:
        verb = "Raise" if new > old else "Lower"
        text = f"{verb} {label[0].lower() + label[1:]} from {format_value(field, old)} to {format_value(field, new)}"
    return FeatureChange(field, label, float(old), float(new), text)


def single_changes(row: pd.Series) -> dict[str, list[Option]]:
    """Candidate single changes per field, smallest change first."""
    options: dict[str, list[Option]] = {}

    util = row["RevUtil"]
    if _known(util) and util > 0:
        steps_below = math.ceil(util / UTIL_STEP - 1e-9)
        targets = [round(k * UTIL_STEP, 2) for k in range(steps_below - 1, -1, -1)]
        options["RevUtil"] = [Option((_change("RevUtil", util, t),), (util - t) / util) for t in targets]

    debt = row["MonthlyDebt"]
    if _known(debt) and debt > 0:
        targets = sorted({round(debt * (1 - cut), -1) for cut in DEBT_CUTS} - {debt}, reverse=True)
        options["MonthlyDebt"] = [Option((_change("MonthlyDebt", debt, t),), (debt - t) / debt)
                                  for t in targets if t < debt]

    income = row["MonthlyIncome"]
    if _known(income) and income > INCOME_FLOOR:
        targets = sorted({round(income * (1 + raise_), -1) for raise_ in INCOME_RAISES})
        options["MonthlyIncome"] = [Option((_change("MonthlyIncome", income, t),), (t - income) / income)
                                    for t in targets if t > income]

    lines, mortgages = row["OpenLines"], row["RealEstateLines"]
    if _known(lines) and lines > 0:
        lowest = int(mortgages) if _known(mortgages) else 0
        options["OpenLines"] = [Option((_change("OpenLines", lines, t),), (lines - t) / lines)
                                for t in range(int(lines) - 1, lowest - 1, -1)]

    return {field: opts for field, opts in options.items() if opts}


def candidate_options(row: pd.Series) -> list[Option]:
    """All single changes plus every pair of changes to two different fields."""
    singles = single_changes(row)
    options = [option for field_options in singles.values() for option in field_options]
    for first, second in itertools.combinations(singles, 2):
        options += [Option(a.changes + b.changes, a.effort + b.effort)
                    for a in singles[first] for b in singles[second]]
    return options


def evaluate_options(pipeline: RiskPipeline, row: pd.Series, options: list[Option]) -> np.ndarray:
    """Combined probability of default after each option (one pipeline call)."""
    values = np.repeat(row[EXPLAINED_FIELDS].to_numpy(float)[None, :], len(options), axis=0)
    column = {field: j for j, field in enumerate(EXPLAINED_FIELDS)}
    for i, option in enumerate(options):
        for change in option.changes:
            values[i, column[change.feature]] = change.new_value
    return pipeline.predict(pd.DataFrame(values, columns=EXPLAINED_FIELDS))[COMBINED].to_numpy()


def select_options(options: list[Option], probs: np.ndarray, base: float, policy,
                   max_items: int = MAX_ITEMS) -> list[tuple[Option, float]]:
    """Pick the few options worth showing (see the module docstring)."""
    # Any option that passes is worth showing (the applicant does not pass yet, so it lowers the risk);
    # options that do not pass must lower the risk noticeably.
    passing = [(o, p) for o, p in zip(options, probs) if policy.passes(p)]
    helpful = [(o, p) for o, p in zip(options, probs) if base - p >= MIN_IMPROVEMENT]

    def cheapest(pool):
        return min(pool, key=lambda item: (item[0].effort, item[1]), default=None)

    def strongest(pool):
        return min(pool, key=lambda item: (item[1], item[0].effort), default=None)

    chosen = [best for field in ADVICE_FIELDS
              if (best := cheapest([(o, p) for o, p in passing if o.fields == {field}]))]
    pair = cheapest([(o, p) for o, p in passing if len(o.changes) == 2])
    if pair and (not chosen or pair[0].effort < min(o.effort for o, _ in chosen)):
        chosen.append(pair)

    if not chosen:  # nothing is enough: show what helps most
        chosen = [best for field in ADVICE_FIELDS
                  if (best := strongest([(o, p) for o, p in helpful if o.fields == {field}]))]
        pair = strongest([(o, p) for o, p in helpful if len(o.changes) == 2])
        if pair:
            chosen.append(pair)

    # Passing options first (easiest first), then the others (strongest first).
    chosen.sort(key=lambda item: (not policy.passes(item[1]),
                                  item[0].effort if policy.passes(item[1]) else item[1]))
    return chosen[:max_items]


def resolve_policy(policy=None, cutoff: float | None = None, review_margin: float | None = None):
    """The risk policy to test against. Uses D's slider conversion, so advice and decision agree."""
    if policy is not None:
        return policy
    from Source.decision.service import make_policy
    return make_policy(risk_cutoff=cutoff, review_margin=review_margin)


def generate_advice(pipeline: RiskPipeline, inputs: pd.DataFrame, cutoff: float | None = None,
                    review_margin: float | None = None, policy=None,
                    max_items: int = MAX_ITEMS) -> list[list[AdviceItem]]:
    """Entry point used by the app: advice for each applicant (row) in ``inputs``.

    The slider is given either as ``policy`` (a RiskPolicy) or as ``cutoff`` and
    ``review_margin`` with the app's convention (absolute half-width, e.g. 0.03).
    Without any of them, the decision step's default policy is used.
    """
    policy = resolve_policy(policy, cutoff, review_margin)
    frame = as_explained_frame(inputs)
    base_probs = pipeline.predict(frame)[COMBINED].to_numpy()

    results = []
    for (_, row), base in zip(frame.iterrows(), base_probs):
        options = [] if policy.passes(base) else candidate_options(row)
        if not options:
            results.append([])
            continue
        probs = evaluate_options(pipeline, row, options)
        results.append([
            AdviceItem(changes=list(option.changes), new_prob=float(prob), prob_change=float(prob - base),
                       passes_cutoff=bool(policy.passes(prob)), effort=round(float(option.effort), 4))
            for option, prob in select_options(options, probs, float(base), policy, max_items)
        ])
    return results


# ---------------------------------------------------------------------------
# Examples and a validation check (never the test split)
# ---------------------------------------------------------------------------

EXAMPLES = {
    "borderline": dict(RevUtil=0.75, Age=42, Late30_59=1, Late60_89=0, Late90=0, MonthlyIncome=4800,
                       MonthlyDebt=1850, OpenLines=7, RealEstateLines=1, Dependents=1),
    "high risk": dict(RevUtil=1.15, Age=33, Late30_59=2, Late60_89=1, Late90=2, MonthlyIncome=3100,
                      MonthlyDebt=2600, OpenLines=5, RealEstateLines=0, Dependents=1),
    "income not declared": dict(RevUtil=0.95, Age=29, Late30_59=1, Late60_89=0, Late90=0, MonthlyIncome=np.nan,
                                MonthlyDebt=1500, OpenLines=5, RealEstateLines=0, Dependents=0),
}


def main() -> None:
    import time
    import warnings

    from Source import config
    from Source.data.preprocessing import TARGET, load_split
    from Source.decision.models import load_risk_pipeline
    from Source.decision.policy import RiskPolicy
    from Source.explain.factors import to_form

    warnings.filterwarnings("ignore", category=UserWarning)
    pipeline = load_risk_pipeline()

    for label, settings in [("app default: cut-off 12%, margin 3 points", dict(cutoff=0.12, review_margin=0.03)),
                            ("decision default: cut-off 20%", dict(policy=RiskPolicy()))]:
        policy = resolve_policy(**settings)
        print(f"\n=== {label} (passes below {policy.approve_below:.1%})")
        for name, profile in EXAMPLES.items():
            inputs = pd.DataFrame([profile])
            start = time.perf_counter()
            advice = generate_advice(pipeline, inputs, **settings)[0]
            seconds = time.perf_counter() - start
            prob = pipeline.predict(inputs)[COMBINED].iloc[0]
            print(f"- {name}: probability {prob:.1%}, {len(advice)} options ({seconds:.2f}s)")
            for item in advice:
                status = "passes" if item.passes_cutoff else "not enough"
                print(f"    {item.new_prob:6.1%} ({item.prob_change * 100:+.1f} pts, {status}, effort {item.effort:.2f}): "
                      + " + ".join(c.description for c in item.changes))

    # How often can advice help? Validation applicants who do not pass at the decision default.
    policy = RiskPolicy()
    readable = load_split("val", "readable")
    forms = to_form(readable)
    probs = pipeline.predict(forms)[COMBINED].to_numpy()
    needing = readable.index[~np.array([policy.passes(p) for p in probs])]
    sample = readable.loc[needing].sample(min(400, len(needing)), random_state=config.RANDOM_STATE)
    start = time.perf_counter()
    advice = generate_advice(pipeline, to_form(sample), policy=policy)
    seconds = (time.perf_counter() - start) / len(sample)

    has_pass = np.array([any(a.passes_cutoff for a in items) for items in advice])
    single = np.array([any(a.passes_cutoff and len(a.changes) == 1 for a in items) for items in advice])
    efforts = [min(a.effort for a in items if a.passes_cutoff) for items in advice if any(a.passes_cutoff for a in items)]
    print(f"\nValidation: {len(needing):,} of {len(readable):,} applicants do not pass at the default; "
          f"sample of {len(sample)} ({seconds:.2f}s per applicant)")
    print(f"  some change passes: {has_pass.mean():.0%} (a single change: {single.mean():.0%}); "
          f"median effort of the easiest passing option: {np.median(efforts):.2f}")
    fields = pd.Series([c.feature for items in advice for a in items if a.passes_cutoff for c in a.changes])
    print("  fields in passing options:", fields.value_counts().to_dict())
    print(f"  default rate of the sample: with passing advice {sample[TARGET][has_pass].mean():.1%}, "
          f"without {sample[TARGET][~has_pass].mean():.1%}")

    # Fairness check for A: is passing advice equally available across age groups?
    groups = pd.cut(sample["Age"], [0, 35, 55, 200], labels=["<=35", "36-55", ">55"])
    table = pd.DataFrame({"age group": groups, "passing advice": has_pass}).groupby("age group", observed=True)
    print("  passing advice by age group:",
          {str(k): f"{v.mean():.0%} (n={len(v)})" for k, v in table["passing advice"]})


if __name__ == "__main__":
    main()
