"""Pre-defined applicant profiles used to test and demonstrate the IDSS."""

from dataclasses import dataclass


@dataclass
class UseCase:
    id: str
    title: str
    subtitle: str
    expected_outcome: str
    narrative: str
    inputs: dict
    default_cutoff: float = 0.12


USE_CASES = {
    "uc1_prime_approval": UseCase(
        id="uc1_prime_approval",
        title="Case 1: Low-risk applicant (Approve)",
        subtitle="Low utilisation, stable income, no late payments.",
        expected_outcome="approve",
        narrative=(
            "44-year-old applicant earning $6,800/month with $1,450 monthly debt (21% debt ratio), "
            "18% revolving utilisation, 1 mortgage, and no late payments."
        ),
        inputs={
            "RevUtil": 0.18,
            "Age": 44.0,
            "Late30_59": 0.0,
            "Late60_89": 0.0,
            "Late90": 0.0,
            "MonthlyIncome": 6800.0,
            "MonthlyDebt": 1450.0,
            "OpenLines": 7.0,
            "RealEstateLines": 1.0,
            "Dependents": 1.0,
            "LateCodeFlag": 0.0,
        },
        default_cutoff=0.12,
    ),
    "uc2_borderline_review": UseCase(
        id="uc2_borderline_review",
        title="Case 2: Borderline applicant (Review)",
        subtitle="Near the cut-off due to 75% revolving utilisation and one minor late payment.",
        expected_outcome="review",
        narrative=(
            "42-year-old applicant earning $4,800/month with 75% revolving utilisation, "
            "$1,850 monthly debt, and one 30-59 day late payment. Risk is close to the 12% cut-off."
        ),
        inputs={
            "RevUtil": 0.75,
            "Age": 42.0,
            "Late30_59": 1.0,
            "Late60_89": 0.0,
            "Late90": 0.0,
            "MonthlyIncome": 4800.0,
            "MonthlyDebt": 1850.0,
            "OpenLines": 7.0,
            "RealEstateLines": 1.0,
            "Dependents": 1.0,
            "LateCodeFlag": 0.0,
        },
        default_cutoff=0.12,
    ),
    "uc3_high_risk_reject": UseCase(
        id="uc3_high_risk_reject",
        title="Case 3: High-risk applicant (Reject)",
        subtitle="Over-limit credit cards (115%), repeated 90+ day late payments, high debt ratio.",
        expected_outcome="reject",
        narrative=(
            "33-year-old applicant with 115% revolving utilisation, two 90+ day late payments, "
            "and $2,600 monthly debt on $3,100 income (84% debt ratio)."
        ),
        inputs={
            "RevUtil": 1.15,
            "Age": 33.0,
            "Late30_59": 2.0,
            "Late60_89": 1.0,
            "Late90": 2.0,
            "MonthlyIncome": 3100.0,
            "MonthlyDebt": 2600.0,
            "OpenLines": 5.0,
            "RealEstateLines": 0.0,
            "Dependents": 1.0,
            "LateCodeFlag": 0.0,
        },
        default_cutoff=0.12,
    ),
    "uc4_unusual_ood_review": UseCase(
        id="uc4_unusual_ood_review",
        title="Case 4: Unusual profile (Review)",
        subtitle="Low predicted risk, but unusual combination of age, income, and 12 mortgages.",
        expected_outcome="review",
        narrative=(
            "20-year-old applicant with no late payments and 15% utilisation, but holding "
            "12 mortgages and 18 open credit lines on $2,500 monthly income. Flagged as unusual."
        ),
        inputs={
            "RevUtil": 0.15,
            "Age": 20.0,
            "Late30_59": 0.0,
            "Late60_89": 0.0,
            "Late90": 0.0,
            "MonthlyIncome": 2500.0,
            "MonthlyDebt": 750.0,
            "OpenLines": 18.0,
            "RealEstateLines": 12.0,
            "Dependents": 0.0,
            "LateCodeFlag": 0.0,
        },
        default_cutoff=0.12,
    ),
}
