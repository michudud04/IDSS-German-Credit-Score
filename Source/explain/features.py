"""Metadata about the input fields: readable names, formatting and what may be changed.

``actionable`` decides whether the what-if advice may ever change a field.
``sensitive`` marks fields that must never appear as advice and that the
LLM explainer has to treat with care (age, family situation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldInfo:
    label: str          # name shown to officer and customer
    unit: str           # "ratio", "money", "count" or "years"
    actionable: bool    # may the advice suggest changing it?
    sensitive: bool     # protected / personal attribute


FIELDS: dict[str, FieldInfo] = {
    "RevUtil":         FieldInfo("Revolving credit utilisation", "ratio", actionable=True, sensitive=False),
    "MonthlyDebt":     FieldInfo("Monthly debt payments", "money", actionable=True, sensitive=False),
    "MonthlyIncome":   FieldInfo("Monthly income", "money", actionable=True, sensitive=False),
    "OpenLines":       FieldInfo("Open credit lines and loans", "count", actionable=True, sensitive=False),
    "RealEstateLines": FieldInfo("Real-estate loans", "count", actionable=False, sensitive=False),
    "Late30_59":       FieldInfo("Payments 30-59 days late (last 2 years)", "count", actionable=False, sensitive=False),
    "Late60_89":       FieldInfo("Payments 60-89 days late (last 2 years)", "count", actionable=False, sensitive=False),
    "Late90":          FieldInfo("Payments 90+ days late", "count", actionable=False, sensitive=False),
    "LateCodeFlag":    FieldInfo("Unexplained lateness code in records", "count", actionable=False, sensitive=False),
    "Age":             FieldInfo("Age", "years", actionable=False, sensitive=True),
    "Dependents":      FieldInfo("Number of dependents", "count", actionable=False, sensitive=True),
}

# Fields explained by the key-factor module (officer form fields + the data-quality flag).
EXPLAINED_FIELDS = list(FIELDS)
ACTIONABLE_FIELDS = [name for name, info in FIELDS.items() if info.actionable]


def format_value(field: str, value: float | None, estimated: bool = False) -> str:
    """Human-readable value, e.g. '95%', '$3,400', '2'. Marks estimated values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "not declared"
    unit = FIELDS[field].unit
    if unit == "ratio":
        text = f"{value:.0%}"
    elif unit == "money":
        text = f"${value:,.0f}"
    else:
        text = f"{value:,.0f}"
    return f"{text} (estimated)" if estimated else text
