"""Build the fixed train / validation / test split and the prepared tables.

Run from the repository root:

    python -m Source.data.build_dataset

Steps
1. Load the raw CSV and apply the rule-based corrections (EDA section 5).
2. Split, stratified on the target, with one fixed seed:
     - test  = 20% of all rows. Identical to the test set of the EDA notebook,
               so nothing that was already kept aside changes.
     - val   = 20% of the remaining rows, for tuning, model combination,
               thresholds and the risk cut-off.
     - train = the rest, the only rows used to fit anything.
   Sizes: 96,000 / 24,000 / 30,000.
3. Fit the learned preprocessing (medians, income model, scaler) on train only.
4. Write readable and model-scale tables for each split, plus the fitted
   preprocessing, to ``data/processed/``.

Never tune or select anything on the test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Source import config
from Source.data.preprocessing import ALL_FEATURES, TARGET, Preprocessor, correct_errors, load_raw


def make_splits(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fixed, stratified train / val / test split of the cleaned rows."""
    rest, test = train_test_split(clean, test_size=config.TEST_SIZE, stratify=clean[TARGET],
                                  random_state=config.RANDOM_STATE)
    train, val = train_test_split(rest, test_size=config.VAL_SIZE, stratify=rest[TARGET],
                                  random_state=config.RANDOM_STATE)
    return {"train": train, "val": val, "test": test}


def main() -> None:
    clean = correct_errors(load_raw())
    splits = make_splits(clean)
    prep = Preprocessor.fit(splits["train"])

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        readable, model = prep.transform(part)

        # Checks: nothing missing, nothing infinite, prepare_customer gives the same result.
        assert np.isfinite(readable.drop(columns=TARGET).to_numpy(float)).all(), name
        assert np.isfinite(model.drop(columns=TARGET).to_numpy(float)).all(), name
        form_input = part.drop(columns=[TARGET, "IncomeMissing", "DependentsMissing"])
        replay_readable, replay_model = prep.prepare_customer(form_input)
        pd.testing.assert_frame_equal(replay_readable[ALL_FEATURES], readable[ALL_FEATURES], check_dtype=False)
        pd.testing.assert_frame_equal(replay_model[ALL_FEATURES], model[ALL_FEATURES], check_dtype=False)

        readable.to_csv(config.PROCESSED_DIR / f"{name}_readable.csv")
        model.to_csv(config.PROCESSED_DIR / f"{name}_model.csv")
        print(f"{name:5s} {len(part):7,d} rows, default rate {part[TARGET].mean():.3%}")

    assert not set(splits["train"].index) & set(splits["val"].index)
    assert not set(splits["train"].index) & set(splits["test"].index)
    assert not set(splits["val"].index) & set(splits["test"].index)

    prep.save()
    print(f"Saved prepared splits and preprocessing to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
