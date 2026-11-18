"""Project-wide paths and constants shared by every module.

Keep all file locations and random seeds here so that every part of the IDSS
reads the same data split and the same saved artefacts.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data (git-ignored; the raw CSV is the Kaggle "Give Me Some Credit" cs-training.csv).
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "GiveMeSomeCredit-training.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PREPROCESSING_FILE = PROCESSED_DIR / "preprocessing.joblib"

# Trained models and fitted explanation artefacts.
MODELS_DIR = ROOT / "Models"

# Reproducibility: one seed and one split for the whole project.
RANDOM_STATE = 42
TEST_SIZE = 0.20  # share of all rows held out for the final evaluation (same as the EDA)
VAL_SIZE = 0.20   # share of the remaining training rows used for validation / tuning
