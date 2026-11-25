"""Streamlit web interface for the Credit Decision Support System.

Run from the repository root:
    PYTHONPATH=. streamlit run Source/app/app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Source import config
from Source.app.case_result import CaseResult
from Source.app.orchestrator import IDSSOrchestrator
from Source.app.use_cases import USE_CASES
from Source.data.preprocessing import Preprocessor
from Source.explain.features import FIELDS, format_value
from Source.explain.similar import SimilarCaseFinder
from Source.explain.unusual import load_detector

st.set_page_config(
    page_title="Credit Decision Support",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading models and training data...")
def get_orchestrator() -> IDSSOrchestrator:
    """Load preprocessing, models, and nearest-neighbour index once at startup."""
    prep = Preprocessor.load()
    finder = SimilarCaseFinder.from_training_data()
    detector = load_detector(finder)
    return IDSSOrchestrator(preprocessor=prep, similar_finder=finder, unusual_detector=detector)


@st.cache_data(show_spinner=False)
def get_test_cases_table() -> pd.DataFrame:
    """Run the 4 pre-defined test cases once at their default settings."""
    orchestrator = get_orchestrator()
    rows = []
    for uc_obj in USE_CASES.values():
        uc_res = orchestrator.evaluate_applicant(
            uc_obj.inputs,
            risk_cutoff=uc_obj.default_cutoff,
            review_margin=0.03,
        )
        rows.append({
            "Case": uc_obj.title,
            "Expected": uc_obj.expected_outcome,
            "Decision": uc_res.decision,
            "Combined risk": f"{uc_res.combined_probability:.1%}",
            "LogReg / XGBoost / Tree": (
                f"{uc_res.probabilities['logreg']:.1%} / "
                f"{uc_res.probabilities['xgboost']:.1%} / "
                f"{uc_res.probabilities['tree']:.1%}"
            ),
            "50-neighbour default rate": f"{uc_res.similar_cases.neighbourhood_default_rate:.1%}",
            "Unusual": "Yes" if uc_res.is_unusual else "No",
        })
    return pd.DataFrame(rows)


def _load_preset_into_state(preset_key: str) -> None:
    """Populate all sidebar widgets from the selected example case."""
    if preset_key == "custom":
        return
    uc = USE_CASES[preset_key]
    inp = uc.inputs
    st.session_state["risk_cutoff_pct"] = float(uc.default_cutoff * 100)
    st.session_state["review_margin_pct"] = 3.0
    st.session_state["rev_util_pct"] = float(inp["RevUtil"] * 100.0)
    st.session_state["income_declared"] = inp["MonthlyIncome"] is not None
    st.session_state["monthly_income"] = float(inp["MonthlyIncome"] if inp["MonthlyIncome"] is not None else 4500.0)
    st.session_state["monthly_debt"] = float(inp["MonthlyDebt"])
    st.session_state["open_lines"] = int(inp["OpenLines"])
    st.session_state["real_estate"] = int(inp["RealEstateLines"])
    st.session_state["late30"] = int(inp["Late30_59"])
    st.session_state["late60"] = int(inp["Late60_89"])
    st.session_state["late90"] = int(inp["Late90"])
    st.session_state["late_code_flag"] = bool(inp["LateCodeFlag"])
    st.session_state["age"] = int(inp["Age"])
    st.session_state["dependents_declared"] = inp["Dependents"] is not None
    st.session_state["dependents"] = int(inp["Dependents"] if inp["Dependents"] is not None else 0)


def render_decision_box(result: CaseResult) -> None:
    """Display the top recommendation banner with probability bounds and reasons."""
    styles = {
        "approve": ("#15803d", "#f0fdf4", "Recommendation: Approve"),
        "review": ("#b45309", "#fffbeb", "Recommendation: Manual review"),
        "reject": ("#b91c1c", "#fef2f2", "Recommendation: Reject"),
    }
    border, bg, heading = styles[result.decision]
    reasons_html = "".join(f"<li>{r}</li>" for r in result.decision_reasons)
    pol = result.policy_summary

    st.markdown(
        f"""
        <div style="border-left: 4px solid {border}; background: {bg};
                    padding: 14px 18px; border-radius: 4px; margin-bottom: 16px;">
            <div style="font-size: 1.15rem; font-weight: 600; color: {border}; margin-bottom: 4px;">
                {heading}
            </div>
            <div style="font-size: 0.92rem; color: #334155; margin-bottom: 8px;">
                Combined default risk: <strong>{result.combined_probability:.1%}</strong>
                &nbsp;&middot;&nbsp; Approve below: <strong>{pol['approve_below']:.1%}</strong>
                &nbsp;&middot;&nbsp; Cut-off: <strong>{result.risk_cutoff:.1%}</strong>
                &nbsp;&middot;&nbsp; Reject from: <strong>{pol['reject_from']:.1%}</strong>
                &nbsp;&middot;&nbsp; Model spread: <strong>{result.model_disagreement:.1%}</strong>
            </div>
            <ul style="margin: 0; padding-left: 18px; font-size: 0.9rem; color: #1e293b;">
                {reasons_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_models_tab(orchestrator: IDSSOrchestrator) -> None:
    """Show ensemble weights and global feature weights/importances."""
    st.subheader("Ensemble weights")
    weights = orchestrator.pipeline.weights
    w_df = pd.DataFrame([
        {"Model": "XGBoost Classifier", "Feature space": "Readable units", "Weight": f"{weights['xgboost']:.0%}"},
        {"Model": "Logistic Regression", "Feature space": "Log-transformed & standardised", "Weight": f"{weights['logreg']:.0%}"},
        {"Model": "Decision Tree Classifier", "Feature space": "Readable units", "Weight": f"{weights['tree']:.0%}"},
    ])
    st.dataframe(w_df, width="stretch", hide_index=True)

    lr_data = json.loads((config.MODELS_DIR / "logistic_regression_weights.json").read_text())
    xgb_data = json.loads((config.MODELS_DIR / "xgboost_feature_importance.json").read_text())

    col_lr, col_xgb = st.columns(2)
    with col_lr:
        st.subheader("Logistic regression weights")
        st.dataframe(pd.DataFrame(lr_data["coefficients"]), width="stretch", hide_index=True)

    with col_xgb:
        st.subheader("XGBoost feature importances")
        st.dataframe(pd.DataFrame(xgb_data["feature_importances"]), width="stretch", hide_index=True)


def main() -> None:
    orchestrator = get_orchestrator()

    # Initialise widget state on first page load
    if "selected_preset" not in st.session_state:
        st.session_state["selected_preset"] = "uc1_prime_approval"
        _load_preset_into_state("uc1_prime_approval")

    # Sidebar: example selector, risk slider, and applicant input form
    st.sidebar.header("Applicant details")

    preset_keys = ["custom"] + list(USE_CASES.keys())
    preset_names = {
        "custom": "Custom input",
        **{k: v.title for k, v in USE_CASES.items()},
    }
    selected_preset = st.sidebar.selectbox(
        "Example case",
        options=preset_keys,
        key="selected_preset",
        format_func=lambda k: preset_names[k],
        on_change=lambda: _load_preset_into_state(st.session_state["selected_preset"]),
    )

    if selected_preset != "custom":
        st.sidebar.caption(USE_CASES[selected_preset].subtitle)

    st.sidebar.subheader("Decision settings")
    risk_cutoff = (
        st.sidebar.slider(
            "Risk cut-off (%)",
            min_value=5.0,
            max_value=35.0,
            step=0.5,
            key="risk_cutoff_pct",
        )
        / 100.0
    )
    review_margin = (
        st.sidebar.slider(
            "Review margin (+/- %)",
            min_value=1.0,
            max_value=8.0,
            step=0.5,
            key="review_margin_pct",
        )
        / 100.0
    )

    st.sidebar.subheader("Financial profile")
    rev_util_pct = st.sidebar.number_input(
        "Revolving credit utilisation (%)",
        min_value=0.0,
        max_value=500.0,
        step=5.0,
        key="rev_util_pct",
    )

    income_declared = st.sidebar.checkbox("Monthly income declared", key="income_declared")
    if income_declared:
        st.session_state.setdefault("monthly_income", 4500.0)
        monthly_income = st.sidebar.number_input(
            "Monthly income ($)",
            min_value=0.0,
            max_value=1_000_000.0,
            step=250.0,
            key="monthly_income",
        )
    else:
        monthly_income = np.nan

    monthly_debt = st.sidebar.number_input(
        "Monthly debt payments ($)",
        min_value=0.0,
        max_value=100_000.0,
        step=100.0,
        key="monthly_debt",
    )
    open_lines = st.sidebar.number_input(
        "Open credit lines and loans",
        min_value=0,
        max_value=60,
        step=1,
        key="open_lines",
    )
    real_estate = st.sidebar.number_input(
        "Real-estate loans",
        min_value=0,
        max_value=30,
        step=1,
        key="real_estate",
    )

    st.sidebar.subheader("Payment history (last 2 years)")
    late30 = st.sidebar.number_input("Payments 30-59 days late", min_value=0, max_value=30, step=1, key="late30")
    late60 = st.sidebar.number_input("Payments 60-89 days late", min_value=0, max_value=30, step=1, key="late60")
    late90 = st.sidebar.number_input("Payments 90+ days late", min_value=0, max_value=30, step=1, key="late90")
    late_code_flag = st.sidebar.checkbox("Unexplained lateness code in records (96/98)", key="late_code_flag")

    st.sidebar.subheader("Personal details")
    age = st.sidebar.number_input("Age (years)", min_value=18, max_value=105, step=1, key="age")
    dependents_declared = st.sidebar.checkbox("Dependents declared", key="dependents_declared")
    if dependents_declared:
        st.session_state.setdefault("dependents", 0)
        dependents = float(
            st.sidebar.number_input(
                "Number of dependents",
                min_value=0,
                max_value=20,
                step=1,
                key="dependents",
            )
        )
    else:
        dependents = np.nan

    applicant_inputs = {
        "RevUtil": rev_util_pct / 100.0,
        "Age": float(age),
        "Late30_59": float(late30),
        "Late60_89": float(late60),
        "Late90": float(late90),
        "MonthlyIncome": monthly_income,
        "MonthlyDebt": float(monthly_debt),
        "OpenLines": float(open_lines),
        "RealEstateLines": float(real_estate),
        "Dependents": dependents,
        "LateCodeFlag": 1.0 if late_code_flag else 0.0,
    }

    result = orchestrator.evaluate_applicant(
        applicant_inputs,
        risk_cutoff=risk_cutoff,
        review_margin=review_margin,
    )

    # Main header and decision banner
    st.title("Credit Decision Support System")
    render_decision_box(result)

    # Warning box when the applicant differs strongly from training data
    if result.is_unusual:
        trigger_labels = {
            "far_from_past_applicants": "Far from past applicants",
            "extreme_value": "Extreme value",
            "outside_observed_range": "Outside observed range",
            "lateness_code": "Unexplained lateness code (96/98)",
        }
        triggers_text = ", ".join(trigger_labels.get(t, t) for t in result.unusual.triggers)
        st.warning(
            "**Unusual applicant profile**\n\n"
            + f"Triggers: {triggers_text}\n\n"
            + "\n".join(f"- {r}" for r in result.unusual.reasons)
            + f"\n\nDistance score: {result.unusual.score:.2f} (threshold: {result.unusual.threshold:.2f})."
        )

    tab_officer, tab_customer, tab_cases, tab_models = st.tabs([
        "Officer view",
        "Customer view",
        "Test cases",
        "Models",
    ])

    # Tab 1: Officer diagnostics
    with tab_officer:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Combined risk", f"{result.combined_probability:.1%}")
        c2.metric("Logistic regression", f"{result.probabilities['logreg']:.1%}")
        c3.metric("XGBoost", f"{result.probabilities['xgboost']:.1%}")
        c4.metric("Decision tree", f"{result.probabilities['tree']:.1%}")
        c5.metric("50-neighbour default rate", f"{result.similar_cases.neighbourhood_default_rate:.1%}")

        st.subheader("Officer summary")
        st.write(result.officer_summary)

        col_left, col_right = st.columns([1.3, 1.0])

        with col_left:
            st.subheader("Main factors")
            factor_rows = []
            for kf in result.key_factors:
                factor_rows.append({
                    "Field": kf.label,
                    "Value": kf.value_text,
                    "Effect on risk": f"{kf.contribution * 100:+.1f} pts",
                    "LogReg": f"{kf.per_model['logreg'] * 100:+.1f} pts",
                    "XGBoost": f"{kf.per_model['xgboost'] * 100:+.1f} pts",
                    "Tree": f"{kf.per_model['tree'] * 100:+.1f} pts",
                    "Type": "Sensitive" if kf.sensitive else ("Actionable" if kf.actionable else "History"),
                })
            st.dataframe(pd.DataFrame(factor_rows), width="stretch", hide_index=True)

        with col_right:
            st.subheader("Derived figures")
            prep = result.prepared_readable
            ratios_df = pd.DataFrame([
                {"Measure": "Debt-to-income ratio", "Value": f"{prep['DebtRatio']:.0%}"},
                {"Measure": "Monthly saving capacity", "Value": f"${prep['SavingCapacity']:,.0f}"},
                {"Measure": "Income per household member", "Value": f"${prep['IncomePerPerson']:,.0f}"},
                {
                    "Measure": "Monthly income used",
                    "Value": format_value("MonthlyIncome", prep["MonthlyIncome"], estimated=prep["IncomeMissing"] == 1),
                },
                {"Measure": "Total late payments (2 years)", "Value": f"{int(prep['LateTotal'])}"},
            ])
            st.dataframe(ratios_df, width="stretch", hide_index=True)

        st.subheader("Similar past applicants")
        st.caption(
            f"Default rate among the {result.similar_cases.neighbourhood_size} most similar past applicants: "
            f"{result.similar_cases.neighbourhood_default_rate:.1%} "
            f"(overall training base rate: {result.similar_cases.base_rate:.1%})."
        )
        sim_rows = []
        for sc in result.similar_cases.neighbours:
            row_dict = {
                "Case ID": sc.case_id,
                "Distance": f"{sc.distance:.2f}",
                "Outcome": "Defaulted" if sc.defaulted else "No default",
            }
            for f_name in (
                "RevUtil", "MonthlyIncome", "MonthlyDebt", "Late30_59",
                "Late60_89", "Late90", "OpenLines", "RealEstateLines", "Age", "Dependents"
            ):
                est = f_name == "MonthlyIncome" and sc.income_estimated
                row_dict[FIELDS[f_name].label] = format_value(f_name, sc.inputs[f_name], estimated=est)
            sim_rows.append(row_dict)
        st.dataframe(pd.DataFrame(sim_rows), width="stretch", hide_index=True)

    # Tab 2: Customer letter and counterfactual advice
    with tab_customer:
        col_letter, col_advice = st.columns([1.1, 1.0])

        with col_letter:
            st.subheader("Customer letter")
            st.markdown(
                f"""
                <div style="background: #f8fafc; border: 1px solid #cbd5e1;
                            padding: 18px; border-radius: 4px; color: #0f172a; line-height: 1.55;">
                {result.customer_letter.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_advice:
            st.subheader("What would need to change")
            if result.decision == "approve":
                st.info("The application is below the automatic-approval limit. No changes needed.")
            elif result.advice:
                for idx, item in enumerate(result.advice, start=1):
                    status = "Passes approval limit" if item.passes_cutoff else "Lowers risk"
                    steps = "; ".join(c.description for c in item.changes)
                    st.markdown(
                        f"**Option {idx}** ({status} — new risk **{item.new_prob:.1%}**, "
                        f"{item.prob_change * 100:+.1f} pts):  \n{steps}"
                    )
            elif result.combined_probability < result.policy_summary["approve_below"]:
                st.info(
                    "The estimated default risk is already below the approval limit; "
                    "this file is under review for a non-risk reason (such as an unusual profile or affordability check)."
                )
            else:
                st.write(
                    "No single or dual adjustment to current debt, utilisation, or income is sufficient "
                    "to reach the approval threshold."
                )

    # Tab 3: Cached overview of the 4 test scenarios
    with tab_cases:
        st.subheader("Summary of test cases")
        st.dataframe(get_test_cases_table(), width="stretch", hide_index=True)

    # Tab 4: Global model weights and feature importances
    with tab_models:
        render_models_tab(orchestrator)


if __name__ == "__main__":
    main()
