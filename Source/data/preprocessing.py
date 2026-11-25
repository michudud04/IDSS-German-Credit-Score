"""Cleaning and preprocessing for the Give Me Some Credit data.

This module is the importable version of the rules developed in
``notebooks/EDA.ipynb`` (sections 1, 5, 7, 9 and 10). The notebook explains
*why* each step exists; this file only implements the steps so the IDSS can
apply exactly the same treatment to a new applicant at run time.

Pipeline for one table of customers:

    raw CSV schema --load_raw / rename--> raw schema (short names)
                   --correct_errors-----> cleaned schema (rule-based fixes, flags)
                   --Preprocessor.impute-> no missing values (learned on train)
                   --derive-------------> readable features (original units)
                   --to_model_scale-----> model features (log + standardised)

Rule-based steps (``correct_errors``, ``derive``, ``log_transform``) learn
nothing from the data. Learned steps (medians, income model, scaler) live in
``Preprocessor`` and are fitted on the training split only.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from Source import config

# ---------------------------------------------------------------------------
# Schema (EDA section 1)
# ---------------------------------------------------------------------------

RENAME = {
    "SeriousDlqin2yrs": "Default2y",
    "RevolvingUtilizationOfUnsecuredLines": "RevUtil",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Late30_59",
    "DebtRatio": "DebtRatioRaw",
    "NumberOfOpenCreditLinesAndLoans": "OpenLines",
    "NumberOfTimes90DaysLate": "Late90",
    "NumberRealEstateLoansOrLines": "RealEstateLines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Late60_89",
    "NumberOfDependents": "Dependents",
}
TARGET = "Default2y"
LATE_COLS = ["Late30_59", "Late60_89", "Late90"]

# Fields an officer enters for a new applicant (cleaned schema, blanks allowed).
INPUT_FIELDS = [
    "RevUtil", "Age", "Late30_59", "Late60_89", "Late90", "MonthlyIncome",
    "MonthlyDebt", "OpenLines", "RealEstateLines", "Dependents",
]

# ---------------------------------------------------------------------------
# Error-correction rules (EDA section 5)
# ---------------------------------------------------------------------------

INCOME_FLOOR = 100   # incomes <= 100 behave like "not declared" (EDA 4.4)
UTIL_CEILING = 10    # RevUtil > 10 holds dollar balances, not ratios (EDA 4.3)
MIN_AGE = 18         # age 0 is a data-entry mistake (EDA 4.5)
LATE_CODES = [96, 98]  # special codes in the lateness counts (EDA 4.2)

# ---------------------------------------------------------------------------
# Feature lists (EDA sections 7, 9, 10, 11)
# ---------------------------------------------------------------------------

SIMPLE_MEDIAN_COLS = ["Age", "RevUtil", "Dependents"] + LATE_COLS
IMPUTE_INPUTS = ["Age", "RevUtil_log", "MonthlyDebt_log", "OpenLines", "RealEstateLines", "LateAny", "Dependents"]

BINARY = ["LateCodeFlag", "IncomeMissing", "DependentsMissing", "SeriousLate", "OverLimit", "UtilAtOne"]
NUMERIC = [
    "RevUtil", "Age", "Late30_59", "MonthlyIncome", "OpenLines", "Late90", "RealEstateLines",
    "Late60_89", "Dependents", "MonthlyDebt", "DebtRatio", "SavingCapacity", "IncomePerPerson",
    "LateTotal", "NonRealEstateLines",
]
ALL_FEATURES = NUMERIC + BINARY

# The 15 features chosen in EDA section 11 for the models.
SELECTED = [
    "RevUtil", "OverLimit",
    "Late30_59", "Late60_89", "Late90", "LateCodeFlag",
    "MonthlyIncome", "MonthlyDebt", "DebtRatio", "SavingCapacity", "IncomePerPerson", "IncomeMissing",
    "OpenLines", "RealEstateLines",
    "Age",
]
SENSITIVE = ["Age", "Dependents", "IncomePerPerson", "DependentsMissing"]

LOG_COLS = ["RevUtil", "MonthlyIncome", "MonthlyDebt", "DebtRatio", "IncomePerPerson",
            "LateTotal", "Dependents", "RealEstateLines"] + LATE_COLS
SIGNED_LOG_COLS = ["SavingCapacity"]


# ---------------------------------------------------------------------------
# Rule-based steps (no fitting, safe before the split)
# ---------------------------------------------------------------------------

def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Read the Kaggle CSV and switch to the short column names."""
    credit = pd.read_csv(path)
    raw = credit.drop(columns="Unnamed: 0").rename(columns=RENAME)
    raw[TARGET] = raw[TARGET].astype("int8")
    return raw


def correct_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based corrections. Takes the raw schema and returns the cleaned schema."""
    out = df.copy()
    income = out["MonthlyIncome"]

    # Debt must be recovered first, while the original income is still available:
    # the file stores DebtRatio = MonthlyDebt / (MonthlyIncome + 1), with missing income as 0.
    out["MonthlyDebt"] = (out["DebtRatioRaw"] * (income.fillna(0) + 1)).round()
    out = out.drop(columns="DebtRatioRaw")

    is_code = out[LATE_COLS].isin(LATE_CODES)
    out["LateCodeFlag"] = is_code.any(axis=1).astype("int8")
    out["IncomeMissing"] = (income.isna() | income.le(INCOME_FLOOR)).astype("int8")
    out["DependentsMissing"] = out["Dependents"].isna().astype("int8")

    out[LATE_COLS] = out[LATE_COLS].mask(is_code)
    out["MonthlyIncome"] = income.where(income > INCOME_FLOOR)
    out["RevUtil"] = out["RevUtil"].where(out["RevUtil"] <= UTIL_CEILING)
    out["Age"] = out["Age"].where(out["Age"] >= MIN_AGE)
    return out


def imputation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Log-scale view used to model income. Never includes the response."""
    out = pd.DataFrame(index=df.index)
    out["Age"] = df["Age"]
    out["RevUtil_log"] = np.log1p(df["RevUtil"])
    out["MonthlyDebt_log"] = np.log1p(df["MonthlyDebt"])
    out["OpenLines"] = df["OpenLines"]
    out["RealEstateLines"] = df["RealEstateLines"]
    out["LateAny"] = df[LATE_COLS].sum(axis=1).gt(0).astype(float)
    out["Dependents"] = df["Dependents"]
    out["Income_log"] = np.log(df["MonthlyIncome"])
    return out


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived variables of EDA section 9 (needs complete data)."""
    out = df.copy()
    out["DebtRatio"] = out["MonthlyDebt"] / out["MonthlyIncome"]
    out["SavingCapacity"] = out["MonthlyIncome"] - out["MonthlyDebt"]
    out["IncomePerPerson"] = out["MonthlyIncome"] / (out["Dependents"] + 1)
    out["LateTotal"] = out[LATE_COLS].sum(axis=1)
    out["SeriousLate"] = (out["Late60_89"] + out["Late90"]).gt(0).astype("int8")
    out["NonRealEstateLines"] = out["OpenLines"] - out["RealEstateLines"]
    out["OverLimit"] = out["RevUtil"].gt(1).astype("int8")
    out["UtilAtOne"] = (out["RevUtil"] - 1).abs().lt(1e-5).astype("int8")
    return out


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """log1p on right-skewed columns; signed log for SavingCapacity (can be negative)."""
    out = df.copy()
    out[LOG_COLS] = np.log1p(out[LOG_COLS])
    out[SIGNED_LOG_COLS] = np.sign(out[SIGNED_LOG_COLS]) * np.log1p(out[SIGNED_LOG_COLS].abs())
    return out


def boosted_regressor() -> HistGradientBoostingRegressor:
    """Income model chosen in EDA section 7.2 (lowest imputation error)."""
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=config.RANDOM_STATE)


# ---------------------------------------------------------------------------
# Learned steps (fitted on the training split only)
# ---------------------------------------------------------------------------

class Preprocessor:
    """Holds the statistics learned from the training data and applies them.

    Use ``Preprocessor.fit(train_clean)`` once when building the dataset and
    ``Preprocessor.load()`` everywhere else (models, explanations, app).
    """

    def __init__(self, medians: pd.Series, income_imputer: HistGradientBoostingRegressor,
                 scaler: StandardScaler, positive_class_weight: float | None = None):
        self.medians = medians
        self.income_imputer = income_imputer
        self.scaler = scaler
        self.positive_class_weight = positive_class_weight

    # -- fitting ----------------------------------------------------------
    @classmethod
    def fit(cls, train_clean: pd.DataFrame) -> "Preprocessor":
        """Learn medians, the income model and the scaler from cleaned training rows."""
        medians = train_clean[SIMPLE_MEDIAN_COLS].median()

        fit_frame = train_clean.copy()
        fit_frame[SIMPLE_MEDIAN_COLS] = fit_frame[SIMPLE_MEDIAN_COLS].fillna(medians)
        fit_frame = imputation_frame(fit_frame)
        known_income = fit_frame["Income_log"].notna()
        income_imputer = boosted_regressor().fit(fit_frame.loc[known_income, IMPUTE_INPUTS],
                                                 fit_frame.loc[known_income, "Income_log"])

        prep = cls(medians, income_imputer, scaler=StandardScaler())
        readable = derive(prep.impute(train_clean))
        prep.scaler.fit(log_transform(readable)[NUMERIC])

        y = train_clean[TARGET]
        prep.positive_class_weight = float((y == 0).sum() / (y == 1).sum())
        return prep

    # -- applying ---------------------------------------------------------
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with statistics learned from the training data."""
        out = df.copy()
        out[SIMPLE_MEDIAN_COLS] = out[SIMPLE_MEDIAN_COLS].fillna(self.medians)
        inputs = imputation_frame(out)[IMPUTE_INPUTS]
        predicted = np.exp(self.income_imputer.predict(inputs))
        out["MonthlyIncome"] = out["MonthlyIncome"].fillna(pd.Series(predicted, index=out.index))
        return out

    def to_model_scale(self, readable: pd.DataFrame) -> pd.DataFrame:
        """Readable features -> log-transformed and standardised model features."""
        out = log_transform(readable)
        out[NUMERIC] = self.scaler.transform(out[NUMERIC])
        return out

    def transform(self, clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Cleaned rows (output of ``correct_errors``) -> (readable, model) tables."""
        readable = derive(self.impute(clean))
        return readable, self.to_model_scale(readable)

    def prepare_customer(self, customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Form input in the cleaned schema (blanks allowed) -> (readable, model) tables.

        This is the run-time entry point for new applicants. Columns follow
        ``INPUT_FIELDS``; ``LateCodeFlag`` is optional and defaults to 0.
        """
        out = customers.copy()
        out["IncomeMissing"] = (out["MonthlyIncome"].isna() | out["MonthlyIncome"].le(INCOME_FLOOR)).astype("int8")
        out["DependentsMissing"] = out["Dependents"].isna().astype("int8")
        out["LateCodeFlag"] = out.get("LateCodeFlag", pd.Series(0, index=out.index)).astype("int8")
        out["MonthlyIncome"] = out["MonthlyIncome"].where(out["MonthlyIncome"] > INCOME_FLOOR)
        return self.transform(out)

    # -- persistence ------------------------------------------------------
    def save(self, path: Path = config.PREPROCESSING_FILE) -> None:
        """Save as a plain dict (same keys as the EDA notebook) so it loads without this class."""
        artifacts = {
            "medians": self.medians,
            "income_imputer": self.income_imputer,
            "scaler": self.scaler,
            "numeric": NUMERIC,
            "binary": BINARY,
            "selected_features": SELECTED,
            "sensitive_features": SENSITIVE,
            "log_columns": LOG_COLS,
            "signed_log_columns": SIGNED_LOG_COLS,
            "rules": {"income_floor": INCOME_FLOOR, "util_ceiling": UTIL_CEILING, "min_age": MIN_AGE},
            "positive_class_weight": self.positive_class_weight,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, path)

    @classmethod
    def load(cls, path: Path = config.PREPROCESSING_FILE) -> "Preprocessor":
        artifacts = joblib.load(path)
        return cls(artifacts["medians"], artifacts["income_imputer"], artifacts["scaler"],
                   artifacts.get("positive_class_weight"))


def load_split(name: str, kind: str = "readable") -> pd.DataFrame:
    """Load one prepared split. ``name`` in {train, val, test}; ``kind`` in {readable, model}."""
    return pd.read_csv(config.PROCESSED_DIR / f"{name}_{kind}.csv", index_col=0)
