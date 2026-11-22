from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from Source.app.case_result import CaseResult, DecisionLabel
from Source.data.preprocessing import INPUT_FIELDS, Preprocessor
from Source.explain.pipeline import COMBINED, MODEL_NAMES, RiskPipeline, SklearnAdapter
from Source.explain.similar import SimilarCaseFinder
from Source.explain.standins import load_standin_pipeline
from Source.explain.types import AdviceItem, KeyFactor, SimilarCases, UnusualCheck
from Source.explain.unusual import UnusualCaseDetector, load_detector


def _load_optional_callable(module_path: str, attr_name: str):
    """Import an optional teammate function; only returns None if the module/function does not exist yet."""
    try:
        module = __import__(module_path, fromlist=[attr_name])
    except ModuleNotFoundError as exc:
        if exc.name and module_path.startswith(exc.name):
            return None
        raise
    return getattr(module, attr_name, None)


class IDSSOrchestrator:
    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        pipeline: RiskPipeline | None = None,
        similar_finder: SimilarCaseFinder | None = None,
        unusual_detector: UnusualCaseDetector | None = None,
    ) -> None:
        self.preprocessor = preprocessor or Preprocessor.load()
        self.pipeline = pipeline or self._load_pipeline(self.preprocessor)
        self.similar_finder = similar_finder or SimilarCaseFinder.from_training_data()
        self.unusual_detector = unusual_detector or load_detector(self.similar_finder)

    @staticmethod
    def _load_pipeline(preprocessor: Preprocessor) -> RiskPipeline:
        load_models = _load_optional_callable("Source.model_loader", "load_models")
        if load_models is not None:
            raw_models, cfg = load_models()
            features = cfg["features"]
            spaces = cfg["feature_representation"]
            adapters = {
                name: SklearnAdapter(est, features, spaces[name])
                for name, est in raw_models.items()
            }
            return RiskPipeline(preprocessor, adapters, weights=cfg["weights"])
        return load_standin_pipeline()

    def evaluate_applicant(
        self,
        inputs: dict[str, float | None],
        risk_cutoff: float = 0.12,
        review_margin: float = 0.03,
    ) -> CaseResult:
        clean_inputs = self._clean_inputs(inputs)
        input_df = pd.DataFrame([clean_inputs])

        readable_df, scaled_df = self.preprocessor.prepare_customer(input_df)
        readable_row = {col: float(readable_df.iloc[0][col]) for col in readable_df.columns}

        probs_df = self.pipeline.predict(input_df)
        probabilities = {col: float(probs_df.iloc[0][col]) for col in probs_df.columns}
        combined_prob = probabilities[COMBINED]
        per_model_vals = [probabilities[m] for m in MODEL_NAMES if m in probabilities]
        disagreement = float(max(per_model_vals) - min(per_model_vals)) if per_model_vals else 0.0

        key_factors, advice, similar_cases, unusual = self._run_explanations(
            input_df, readable_df, scaled_df, risk_cutoff
        )

        decision, reasons, rules, officer_summary, customer_letter = self._run_decision(
            inputs=clean_inputs,
            readable_row=readable_row,
            probabilities=probabilities,
            combined_prob=combined_prob,
            risk_cutoff=risk_cutoff,
            review_margin=review_margin,
            key_factors=key_factors,
            advice=advice,
            similar_cases=similar_cases,
            unusual=unusual,
        )

        return CaseResult(
            inputs=clean_inputs,
            prepared_readable=readable_row,
            probabilities=probabilities,
            combined_probability=combined_prob,
            model_disagreement=disagreement,
            key_factors=key_factors,
            advice=advice,
            similar_cases=similar_cases,
            unusual=unusual,
            is_unusual=unusual.is_unusual,
            risk_cutoff=risk_cutoff,
            review_margin=review_margin,
            decision=decision,
            decision_reasons=reasons,
            bank_rules_triggered=rules,
            officer_summary=officer_summary,
            customer_letter=customer_letter,
        )

    @staticmethod
    def _clean_inputs(inputs: dict[str, Any]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for name in INPUT_FIELDS:
            val = inputs.get(name)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                out[name] = np.nan
            else:
                out[name] = float(val)
        late_flag = inputs.get("LateCodeFlag", 0.0)
        out["LateCodeFlag"] = 0.0 if late_flag is None or math.isnan(float(late_flag)) else float(late_flag)
        return out

    def _run_explanations(
        self,
        input_df: pd.DataFrame,
        readable_df: pd.DataFrame,
        scaled_df: pd.DataFrame,
        risk_cutoff: float,
    ) -> tuple[list[KeyFactor], list[AdviceItem], SimilarCases, UnusualCheck]:
        explain_case = _load_optional_callable("Source.explain", "explain_case")
        if explain_case is not None:
            exp = explain_case(
                self.pipeline,
                self.similar_finder,
                self.unusual_detector,
                input_df,
                cutoff=risk_cutoff,
            )
            return exp.key_factors, exp.advice, exp.similar_cases, exp.unusual

        similar_cases = self.similar_finder.find(scaled_df)[0]
        unusual = self.unusual_detector.check(readable_df, scaled_df)[0]

        explain_factors = _load_optional_callable("Source.explain.factors", "explain_factors")
        key_factors: list[KeyFactor] = (
            explain_factors(self.pipeline, input_df)[0] if explain_factors is not None else []
        )

        generate_advice = _load_optional_callable("Source.explain.advice", "generate_advice")
        advice: list[AdviceItem] = (
            generate_advice(self.pipeline, input_df, cutoff=risk_cutoff)[0]
            if generate_advice is not None
            else []
        )

        return key_factors, advice, similar_cases, unusual

    @staticmethod
    def _run_decision(
        inputs: dict[str, float | None],
        readable_row: dict[str, float],
        probabilities: dict[str, float],
        combined_prob: float,
        risk_cutoff: float,
        review_margin: float,
        key_factors: list[KeyFactor],
        advice: list[AdviceItem],
        similar_cases: SimilarCases,
        unusual: UnusualCheck,
    ) -> tuple[DecisionLabel, list[str], list[str], str, str]:
        evaluate_decision = _load_optional_callable("Source.decision", "evaluate_decision")
        if evaluate_decision is not None:
            res = evaluate_decision(
                inputs=inputs,
                readable_row=readable_row,
                probabilities=probabilities,
                risk_cutoff=risk_cutoff,
                review_margin=review_margin,
                key_factors=key_factors,
                advice=advice,
                similar_cases=similar_cases,
                unusual=unusual,
            )
            return (
                res["decision"],
                res["decision_reasons"],
                res["bank_rules_triggered"],
                res["officer_summary"],
                res["customer_letter"],
            )

        # Minimal threshold wiring until Person D's Source.decision module is merged
        low_cutoff = max(0.01, risk_cutoff - review_margin)
        high_cutoff = risk_cutoff + review_margin

        if unusual.is_unusual:
            decision: DecisionLabel = "review"
            reasons = ["Unusual applicant profile flagged for manual review."]
        elif combined_prob > high_cutoff:
            decision = "reject"
            reasons = [f"Combined default risk ({combined_prob:.1%}) is above the cut-off ({high_cutoff:.1%})."]
        elif combined_prob >= low_cutoff:
            decision = "review"
            reasons = [f"Combined default risk ({combined_prob:.1%}) is near the cut-off ({risk_cutoff:.1%})."]
        else:
            decision = "approve"
            reasons = [f"Combined default risk ({combined_prob:.1%}) is below the cut-off ({low_cutoff:.1%})."]

        return decision, reasons, [], "", ""


_ORCHESTRATOR: IDSSOrchestrator | None = None


def evaluate_applicant(
    inputs: dict[str, float | None],
    risk_cutoff: float = 0.12,
    review_margin: float = 0.03,
) -> CaseResult:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = IDSSOrchestrator()
    return _ORCHESTRATOR.evaluate_applicant(inputs, risk_cutoff=risk_cutoff, review_margin=review_margin)
