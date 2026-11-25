"""One call for all four explanations of an applicant: the app's entry point to part C.

    explanation = explain_case(pipeline, finder, detector, inputs, cutoff=0.12, review_margin=0.03)
    explanation.key_factors      # top 5 KeyFactors
    explanation.advice           # up to 4 AdviceItems (empty if the applicant already passes)
    explanation.similar_cases    # SimilarCases
    explanation.unusual          # UnusualCheck
    explanation.probabilities    # probability of default per model and "combined"

``pipeline`` is the RiskPipeline with the final models, ``finder`` a
SimilarCaseFinder and ``detector`` an UnusualCaseDetector. Build them once at
start-up and reuse them for every applicant.

The results are the same as calling the four parts one by one, but this call
- runs the nearest-neighbour search once and shares it: the similar cases
  use the 50 closest past applicants, the unusual-case check the 10 closest;
- predicts the applicant's probabilities once and hands them to the advice;
- passes the risk slider (cut-off and review margin) to the advice in one
  place, converted with the decision step's own rule, so advice and decision
  agree on what "passes" means.
"""

from __future__ import annotations

import pandas as pd

from Source.explain.advice import generate_advice, resolve_policy
from Source.explain.factors import explain_factors
from Source.explain.pipeline import RiskPipeline
from Source.explain.similar import SimilarCaseFinder
from Source.explain.types import CaseExplanation
from Source.explain.unusual import UnusualCaseDetector


def explain_cases(pipeline: RiskPipeline, finder: SimilarCaseFinder, detector: UnusualCaseDetector,
                  inputs: pd.DataFrame, cutoff: float | None = None, review_margin: float | None = None,
                  policy=None) -> list[CaseExplanation]:
    """All four explanations for each applicant (row) in ``inputs``.

    The risk slider is given as ``policy`` (a RiskPolicy) or as ``cutoff`` and
    ``review_margin`` (the app's convention); see ``advice.generate_advice``.
    """
    policy = resolve_policy(policy, cutoff, review_margin)
    readable, scaled = pipeline.preprocessor.prepare_customer(inputs)
    probs = pipeline.predict(inputs)

    # One nearest-neighbour search, deep enough for both the similar cases and the unusual-case check.
    neighbours = finder.neighbours(scaled, max(finder.neighbourhood_size, detector.settings["k"]))
    similar = finder.find(scaled, neighbours)
    unusual = detector.check(readable, scaled, neighbours)

    factors = explain_factors(pipeline, inputs)
    advice = generate_advice(pipeline, inputs, policy=policy, base_probs=probs["combined"].to_numpy())

    return [
        CaseExplanation(
            probabilities={name: float(value) for name, value in probs.iloc[i].items()},
            key_factors=factors[i],
            advice=advice[i],
            similar_cases=similar[i],
            unusual=unusual[i],
        )
        for i in range(len(inputs))
    ]


def explain_case(pipeline: RiskPipeline, finder: SimilarCaseFinder, detector: UnusualCaseDetector,
                 inputs: pd.DataFrame | dict, cutoff: float | None = None, review_margin: float | None = None,
                 policy=None) -> CaseExplanation:
    """All four explanations for ONE applicant: a one-row DataFrame or a dict of input values."""
    if isinstance(inputs, dict):
        inputs = pd.DataFrame([inputs])
    if len(inputs) != 1:
        raise ValueError(f"explain_case expects one applicant, got {len(inputs)}; use explain_cases")
    return explain_cases(pipeline, finder, detector, inputs, cutoff, review_margin, policy)[0]
