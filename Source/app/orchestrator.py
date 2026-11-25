"""Main orchestrator connecting preprocessing, models, explanations, and decision."""

import math
import numpy as np
import pandas as pd

from Source.app.case_result import CaseResult
from Source.data.preprocessing import INPUT_FIELDS, Preprocessor
from Source.decision.models import load_risk_pipeline
from Source.decision.service import evaluate_decision
from Source.explain.case import explain_case
from Source.explain.pipeline import COMBINED, MODEL_NAMES, RiskPipeline
from Source.explain.similar import SimilarCaseFinder
from Source.explain.unusual import UnusualCaseDetector, load_detector


class IDSSOrchestrator:
    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        pipeline: RiskPipeline | None = None,
        similar_finder: SimilarCaseFinder | None = None,
        unusual_detector: UnusualCaseDetector | None = None,
    ):
        self.preprocessor = preprocessor or Preprocessor.load()
        self.pipeline = pipeline or load_risk_pipeline(self.preprocessor)
        self.similar_finder = similar_finder or SimilarCaseFinder.from_training_data()
        self.unusual_detector = unusual_detector or load_detector(self.similar_finder)

    def evaluate_applicant(
        self,
        inputs: dict,
        risk_cutoff: float = 0.12,
        review_margin: float = 0.03,
    ) -> CaseResult:
        clean_inputs = self._clean_inputs(inputs)
        input_df = pd.DataFrame([clean_inputs])

        # 1. Preprocessed readable figures for the officer table
        readable_df, _ = self.preprocessor.prepare_customer(input_df)
        readable_row = {col: float(readable_df.iloc[0][col]) for col in readable_df.columns}

        # 2. Predictions, SHAP key factors, what-if advice, similar cases, and unusual check
        exp = explain_case(
            self.pipeline,
            self.similar_finder,
            self.unusual_detector,
            input_df,
            cutoff=risk_cutoff,
            review_margin=review_margin,
        )
        probabilities = exp.probabilities
        combined_prob = probabilities[COMBINED]
        per_model_vals = [probabilities[m] for m in MODEL_NAMES]
        disagreement = float(max(per_model_vals) - min(per_model_vals))

        # 3. Final recommendation and officer/customer texts
        dec = evaluate_decision(
            inputs=clean_inputs,
            readable_row=readable_row,
            probabilities=probabilities,
            risk_cutoff=risk_cutoff,
            review_margin=review_margin,
            key_factors=exp.key_factors,
            advice=exp.advice,
            similar_cases=exp.similar_cases,
            unusual=exp.unusual,
        )
        policy_summary = dec["policy"].to_dict()

        return CaseResult(
            inputs=clean_inputs,
            prepared_readable=readable_row,
            probabilities=probabilities,
            combined_probability=combined_prob,
            model_disagreement=disagreement,
            key_factors=exp.key_factors,
            advice=exp.advice,
            similar_cases=exp.similar_cases,
            unusual=exp.unusual,
            is_unusual=exp.unusual.is_unusual,
            risk_cutoff=float(policy_summary["cutoff"]),
            review_margin=review_margin,
            decision=dec["decision"],
            decision_reasons=dec["decision_reasons"],
            officer_summary=dec["officer_summary"],
            customer_letter=dec["customer_letter"],
            policy_summary=policy_summary,
        )

    @staticmethod
    def _clean_inputs(inputs: dict) -> dict:
        """Convert missing form values to NaN and cast numbers to float."""
        out = {}
        for name in INPUT_FIELDS:
            val = inputs.get(name)
            out[name] = np.nan if val is None or math.isnan(float(val)) else float(val)
        flag = inputs.get("LateCodeFlag")
        out["LateCodeFlag"] = 0.0 if flag is None or math.isnan(float(flag)) else float(flag)
        return out
