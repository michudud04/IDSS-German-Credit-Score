"""Try the texts on three applicants with the real models and the configured LLM.

    python -m Source.decision.narrative.demo             # LLM if reachable, else templates
    IDSS_LLM_BACKEND=template python -m Source.decision.narrative.demo

Prints the decision, both texts, and whether each text came from the LLM or
the template (and why). Uses C's key factors, advice, similar cases and
unusual-case warning, exactly as the app does.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd

from Source.decision.models import load_risk_pipeline
from Source.decision.narrative.writer import NarrativeWriter
from Source.decision.policy import RiskPolicy
from Source.decision.service import evaluate_decision
from Source.explain.advice import generate_advice
from Source.explain.factors import explain_factors
from Source.explain.similar import SimilarCaseFinder
from Source.explain.unusual import load_detector

APPLICANTS = {
    "low risk": dict(RevUtil=0.18, Age=44, Late30_59=0, Late60_89=0, Late90=0, MonthlyIncome=6800,
                     MonthlyDebt=1450, OpenLines=8, RealEstateLines=1, Dependents=1, LateCodeFlag=0),
    "grey zone": dict(RevUtil=0.9, Age=35, Late30_59=1, Late60_89=0, Late90=0, MonthlyIncome=3000,
                      MonthlyDebt=900, OpenLines=5, RealEstateLines=0, Dependents=1, LateCodeFlag=0),
    "high risk": dict(RevUtil=1.15, Age=33, Late30_59=2, Late60_89=1, Late90=2, MonthlyIncome=3100,
                      MonthlyDebt=2600, OpenLines=5, RealEstateLines=0, Dependents=1, LateCodeFlag=0),
}


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    pipeline = load_risk_pipeline()
    finder = SimilarCaseFinder.from_training_data()
    detector = load_detector(finder)
    writer = NarrativeWriter()
    print(f"LLM backend: {writer.backend.name if writer.backend else 'none reachable -> templates'}\n")

    for name, inputs in APPLICANTS.items():
        frame = pd.DataFrame([inputs])
        readable, scaled = pipeline.preprocessor.prepare_customer(frame)
        start = time.perf_counter()
        policy = RiskPolicy()
        result = evaluate_decision(inputs=inputs, probabilities=pipeline.predict(frame).iloc[0].to_dict(),
                                   unusual=detector.check(readable, scaled)[0], similar_cases=finder.find(scaled)[0],
                                   key_factors=explain_factors(pipeline, frame)[0],
                                   advice=generate_advice(pipeline, frame, policy=policy)[0],
                                   cost_ratio=policy.cost_ratio, writer=writer)
        print("=" * 80)
        print(f"{name}: {result['decision'].upper()}  ({time.perf_counter() - start:.1f}s)")
        for key in ("officer_summary", "customer_letter"):
            meta = result[f"{key}_meta"]
            print(f"\n--- {key} [{meta.source}, {meta.attempts} LLM call(s)]")
            if meta.issues:
                print("    rejected drafts: " + " | ".join(meta.issues))
            print(meta.text)
        print()


if __name__ == "__main__":
    main()
