"""Explanations (Person C): key factors, what-if advice, similar cases, unusual-case warning.

All four parts treat the full prediction pipeline as one function of the
fields the officer enters (``Source.data.preprocessing.INPUT_FIELDS``):

    inputs -> preprocessing -> 3 models -> combined probability of default

so factors and advice are expressed in the applicant's own terms, and derived
features (DebtRatio, SavingCapacity, ...) always stay consistent.

Entry point for the app: ``from Source.explain import explain_case`` (see case.py).
"""

__all__ = ["explain_case", "explain_cases"]


def __getattr__(name):
    # Lazy import: `import Source.explain.similar` etc. stays light (no SHAP) unless explain_case is used.
    if name in __all__:
        from importlib import import_module
        return getattr(import_module("Source.explain.case"), name)
    raise AttributeError(f"module 'Source.explain' has no attribute {name!r}")
