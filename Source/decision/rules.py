"""Bank-policy rules applied on top of the model.

The model estimates risk; these rules encode policy that should hold whatever
the model says and whatever the slider setting is. Each rule is checked on the
applicant's own form values, so it can be explained in one sentence.

Rates on the validation split (24,000 applicants, see ``evaluate.py``):

    RULE_SERIOUS_DELINQUENCY  0.8%   reject   59% of them defaulted
    RULE_DEBT_EXCEEDS_INCOME  3.5%   review   affordability check, 11% defaulted
    RULE_MIN_AGE / RULE_MISSING_HISTORY: 0% (form errors that cannot occur in the data)

Deliberately NOT a rule:
- Lateness codes 96/98: the unusual-case warning (C) already sends them to review.
- Income not declared: 21% of applicants, who default less (5.4%) than average.
  The model already uses the "income missing" flag; a rule would add review
  load without adding information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from Source.decision.types import Outcome, Reason
from Source.explain.features import format_value

LEGAL_MIN_AGE = 18
MAX_SERIOUS_LATE = 3          # 90+ days late this many times -> reject, whatever the slider says
REQUIRED_FIELDS = ["RevUtil", "Age", "Late30_59", "Late60_89", "Late90",
                   "MonthlyDebt", "OpenLines", "RealEstateLines"]  # income and dependents may be blank


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


@dataclass(frozen=True)
class Rule:
    code: str
    effect: Outcome
    description: str                                  # one line for the report / app
    check: Callable[[pd.Series], Reason | None]       # applicant form row -> Reason if the rule fires


def _min_age(row: pd.Series) -> Reason | None:
    age = row.get("Age")
    if _blank(age) or age >= LEGAL_MIN_AGE:
        return None
    return Reason("RULE_MIN_AGE", "bank_rule", "reject",
                  officer_text=f"Applicant is {age:.0f}; credit requires full legal age ({LEGAL_MIN_AGE}).",
                  customer_text=f"Credit can only be granted to applicants aged {LEGAL_MIN_AGE} or over.")


def _missing_history(row: pd.Series) -> Reason | None:
    missing = [f for f in REQUIRED_FIELDS if _blank(row.get(f))]
    # Lateness counts are blank on purpose when the record holds the codes 96/98 (handled by C).
    if row.get("LateCodeFlag", 0) == 1:
        missing = [f for f in missing if not f.startswith("Late")]
    if not missing:
        return None
    return Reason("RULE_MISSING_HISTORY", "bank_rule", "review",
                  officer_text="Required fields are blank and were estimated: " + ", ".join(missing)
                               + ". Complete them before deciding.",
                  customer_text="Some information in your application is incomplete; an advisor will "
                                "contact you to complete it.")


def _serious_delinquency(row: pd.Series) -> Reason | None:
    late90 = row.get("Late90")
    if _blank(late90) or late90 < MAX_SERIOUS_LATE:
        return None
    return Reason("RULE_SERIOUS_DELINQUENCY", "bank_rule", "reject",
                  officer_text=f"{late90:.0f} payments were 90+ days late; bank policy rejects from "
                               f"{MAX_SERIOUS_LATE} on, whatever the risk setting.",
                  customer_text="Your credit history shows several payments that were more than 90 days late.")


def _debt_exceeds_income(row: pd.Series) -> Reason | None:
    income, debt = row.get("MonthlyIncome"), row.get("MonthlyDebt")
    if _blank(income) or _blank(debt) or income <= 100 or debt <= income:
        return None  # income <= 100 counts as not declared (preprocessing.INCOME_FLOOR)
    return Reason("RULE_DEBT_EXCEEDS_INCOME", "bank_rule", "review",
                  officer_text=f"Monthly debt payments ({format_value('MonthlyDebt', debt)}) exceed the declared "
                               f"monthly income ({format_value('MonthlyIncome', income)}). Check affordability "
                               "before any approval.",
                  customer_text="Your monthly debt payments are higher than your declared monthly income, so an "
                                "advisor will check together with you whether the loan is affordable.")


RULES: list[Rule] = [
    Rule("RULE_MIN_AGE", "reject", f"Applicant younger than {LEGAL_MIN_AGE}.", _min_age),
    Rule("RULE_SERIOUS_DELINQUENCY", "reject",
         f"{MAX_SERIOUS_LATE} or more payments 90+ days late.", _serious_delinquency),
    Rule("RULE_MISSING_HISTORY", "review", "A required field of the credit history is blank.", _missing_history),
    Rule("RULE_DEBT_EXCEEDS_INCOME", "review", "Monthly debt payments above the declared income.",
         _debt_exceeds_income),
]


def check_rules(row: pd.Series, rules: list[Rule] = RULES) -> list[Reason]:
    """All rules that fire for one applicant (form values in the cleaned schema, blanks as NaN)."""
    return [reason for rule in rules if (reason := rule.check(row)) is not None]
