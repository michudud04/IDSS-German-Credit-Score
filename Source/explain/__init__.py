"""Explanations (Person C): key factors, what-if advice, similar cases, unusual-case warning.

All four parts treat the full prediction pipeline as one function of the
fields the officer enters (``Source.data.preprocessing.INPUT_FIELDS``):

    inputs -> preprocessing -> 3 models -> combined probability of default

so factors and advice are expressed in the applicant's own terms, and derived
features (DebtRatio, SavingCapacity, ...) always stay consistent.
"""
