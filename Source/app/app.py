from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    prep = Preprocessor.load()
    finder = SimilarCaseFinder.from_training_data()
    detector = load_detector(finder)
    return IDSSOrchestrator(preprocessor=prep, similar_finder=finder, unusual_detector=detector)


def render_decision_box(result: CaseResult) -> None:
    styles = {
        "approve": ("#15803d", "#f0fdf4", "Recommendation: Approve"),
        "review": ("#b45309", "#fffbeb", "Recommendation: Manual review"),
        "reject": ("#b91c1c", "#fef2f2", "Recommendation: Reject"),
    }
    border, bg, heading = styles[result.decision]
    reasons_html = "".join(f"<li>{r}</li>" for r in result.decision_reasons)
    st.markdown(
        f"""
        <div style="border-left: 4px solid {border}; background: {bg};
                    padding: 14px 18px; border-radius: 4px; margin-bottom: 16px;">
            <div style="font-size: 1.15rem; font-weight: 600; color: {border}; margin-bottom: 4px;">
                {heading}
            </div>
            <div style="font-size: 0.92rem; color: #334155; margin-bottom: 8px;">
                Combined default risk: <strong>{result.combined_probability:.1%}</strong>
                &nbsp;&middot;&nbsp; Risk cut-off: <strong>{result.risk_cutoff:.1%}</strong>
                &nbsp;&middot;&nbsp; Model spread: <strong>{result.model_disagreement:.1%}</strong>
            </div>
            <ul style="margin: 0; padding-left: 18px; font-size: 0.9rem; color: #1e293b;">
                {reasons_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    orchestrator = get_orchestrator()

    st.sidebar.header("Applicant details")

    preset_keys = ["custom"] + list(USE_CASES.keys())
    preset_names = {
        "custom": "Custom input",
        **{k: v.title for k, v in USE_CASES.items()},
    }
    selected_preset = st.sidebar.selectbox(
        "Example case",
        options=preset_keys,
        index=1,
        format_func=lambda k: preset_names[k],
    )

    if selected_preset != "custom":
        uc = USE_CASES[selected_preset]
        defaults = uc.inputs
        default_cutoff = uc.default_cutoff
        st.sidebar.caption(uc.subtitle)
    else:
        defaults = USE_CASES["uc1_prime_approval"].inputs
        default_cutoff = 0.12

    st.sidebar.subheader("Decision settings")
    risk_cutoff = (
        st.sidebar.slider(
            "Risk cut-off (%)",
            min_value=3.0,
            max_value=35.0,
            value=float(default_cutoff * 100),
            step=0.5,
        )
        / 100.0
    )
    review_margin = (
        st.sidebar.slider(
            "Review margin (+/- %)",
            min_value=1.0,
            max_value=8.0,
            value=3.0,
            step=0.5,
        )
        / 100.0
    )

    st.sidebar.subheader("Financial profile")
    rev_util_pct = st.sidebar.number_input(
        "Revolving credit utilisation (%)",
        min_value=0.0,
        max_value=500.0,
        value=float((defaults["RevUtil"] or 0.0) * 100.0),
        step=5.0,
    )

    income_declared = st.sidebar.checkbox(
        "Monthly income declared",
        value=defaults["MonthlyIncome"] is not None,
    )
    if income_declared:
        monthly_income: float | None = st.sidebar.number_input(
            "Monthly income ($)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=float(defaults["MonthlyIncome"] or 4500.0),
            step=250.0,
        )
    else:
        monthly_income = np.nan

    monthly_debt = st.sidebar.number_input(
        "Monthly debt payments ($)",
        min_value=0.0,
        max_value=100_000.0,
        value=float(defaults["MonthlyDebt"] or 0.0),
        step=100.0,
    )
    open_lines = st.sidebar.number_input(
        "Open credit lines and loans",
        min_value=0,
        max_value=60,
        value=int(defaults["OpenLines"] or 0),
        step=1,
    )
    real_estate = st.sidebar.number_input(
        "Real-estate loans",
        min_value=0,
        max_value=30,
        value=int(defaults["RealEstateLines"] or 0),
        step=1,
    )

    st.sidebar.subheader("Payment history (last 2 years)")
    late30 = st.sidebar.number_input(
        "Payments 30-59 days late",
        min_value=0,
        max_value=30,
        value=int(defaults["Late30_59"] or 0),
        step=1,
    )
    late60 = st.sidebar.number_input(
        "Payments 60-89 days late",
        min_value=0,
        max_value=30,
        value=int(defaults["Late60_89"] or 0),
        step=1,
    )
    late90 = st.sidebar.number_input(
        "Payments 90+ days late",
        min_value=0,
        max_value=30,
        value=int(defaults["Late90"] or 0),
        step=1,
    )
    late_code_flag = st.sidebar.checkbox(
        "Unexplained lateness code in records (96/98)",
        value=bool(defaults.get("LateCodeFlag", 0)),
    )

    st.sidebar.subheader("Personal details")
    age = st.sidebar.number_input(
        "Age (years)",
        min_value=18,
        max_value=105,
        value=int(defaults["Age"] or 40),
        step=1,
    )
    dependents_declared = st.sidebar.checkbox(
        "Dependents declared",
        value=defaults["Dependents"] is not None,
    )
    if dependents_declared:
        dependents: float | None = float(
            st.sidebar.number_input(
                "Number of dependents",
                min_value=0,
                max_value=20,
                value=int(defaults["Dependents"] or 0),
                step=1,
            )
        )
    else:
        dependents = np.nan

    applicant_inputs: dict[str, float | None] = {
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

    st.title("Credit Decision Support System")
    render_decision_box(result)

    if result.is_unusual:
        st.warning(
            "**Unusual applicant profile**\n\n"
            + "\n".join(f"- {r}" for r in result.unusual.reasons)
            + f"\n\nDistance score: {result.unusual.score:.2f} (threshold: {result.unusual.threshold:.2f})."
        )

    tab_officer, tab_customer, tab_cases = st.tabs([
        "Officer view",
        "Customer view",
        "Test cases",
    ])

    with tab_officer:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Combined risk", f"{result.combined_probability:.1%}")
        c2.metric("Logistic regression", f"{result.probabilities.get('logreg', 0):.1%}")
        c3.metric("XGBoost", f"{result.probabilities.get('xgboost', 0):.1%}")
        c4.metric("Decision tree", f"{result.probabilities.get('tree', 0):.1%}")
        c5.metric("50-neighbour default rate", f"{result.similar_cases.neighbourhood_default_rate:.1%}")

        if result.officer_summary:
            st.subheader("Officer summary")
            st.write(result.officer_summary)

        col_left, col_right = st.columns([1.3, 1.0])

        with col_left:
            st.subheader("Main factors")
            if result.key_factors:
                factor_rows = []
                for kf in result.key_factors:
                    factor_rows.append({
                        "Field": kf.label,
                        "Value": kf.value_text,
                        "Effect on risk": f"{kf.contribution * 100:+.1f} pts",
                        "LogReg": f"{kf.per_model.get('logreg', 0) * 100:+.1f} pts",
                        "XGBoost": f"{kf.per_model.get('xgboost', 0) * 100:+.1f} pts",
                        "Tree": f"{kf.per_model.get('tree', 0) * 100:+.1f} pts",
                        "Type": "Sensitive" if kf.sensitive else ("Actionable" if kf.actionable else "History"),
                    })
                st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("Key factors module (Person C) not connected yet.")

        with col_right:
            st.subheader("Derived figures")
            prep = result.prepared_readable
            ratios_df = pd.DataFrame([
                {"Measure": "Debt-to-income ratio", "Value": f"{prep.get('DebtRatio', 0):.0%}"},
                {"Measure": "Monthly saving capacity", "Value": f"${prep.get('SavingCapacity', 0):,.0f}"},
                {"Measure": "Income per household member", "Value": f"${prep.get('IncomePerPerson', 0):,.0f}"},
                {
                    "Measure": "Monthly income used",
                    "Value": format_value("MonthlyIncome", prep.get("MonthlyIncome"), estimated=prep.get("IncomeMissing") == 1),
                },
                {"Measure": "Total late payments (2 years)", "Value": f"{int(prep.get('LateTotal', 0))}"},
            ])
            st.dataframe(ratios_df, use_container_width=True, hide_index=True)

            if result.bank_rules_triggered:
                st.subheader("Bank rules triggered")
                for rule in result.bank_rules_triggered:
                    st.markdown(f"- {rule}")

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
                row_dict[FIELDS[f_name].label] = format_value(f_name, sc.inputs.get(f_name), estimated=est)
            sim_rows.append(row_dict)
        st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

    with tab_customer:
        col_letter, col_advice = st.columns([1.1, 1.0])

        with col_letter:
            st.subheader("Customer letter")
            if result.customer_letter:
                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border: 1px solid #cbd5e1;
                                padding: 18px; border-radius: 4px; color: #0f172a; line-height: 1.55;">
                    {result.customer_letter.replace(chr(10), '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Customer letter module (Person D) not connected yet.")

        with col_advice:
            st.subheader("What would need to change")
            if result.decision == "approve":
                st.info("The application is below the risk cut-off. No changes needed.")
            elif result.advice:
                for idx, item in enumerate(result.advice, start=1):
                    status = "Passes cut-off" if item.passes_cutoff else "Lowers risk"
                    steps = "; ".join(c.description for c in item.changes)
                    st.markdown(
                        f"**Option {idx}** ({status} — new risk **{item.new_prob:.1%}**, "
                        f"{item.prob_change * 100:+.1f} pts):  \n{steps}"
                    )
            else:
                st.caption("Counterfactual advice module (Person C) not connected yet.")

    with tab_cases:
        st.subheader("Summary of test cases")
        uc_rows = []
        for uc_obj in USE_CASES.values():
            uc_res = orchestrator.evaluate_applicant(uc_obj.inputs, risk_cutoff=uc_obj.default_cutoff)
            uc_rows.append({
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
        st.dataframe(pd.DataFrame(uc_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
