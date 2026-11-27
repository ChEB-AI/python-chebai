"""Generate random (non-stratified) train/validation/test splits from DataFrames."""

from __future__ import annotations

from abc import ABC

import pandas as pd
from sklearn.model_selection import train_test_split

from chebai.preprocessing.datasets.base import _DynamicDataset


class RandomSplitter(_DynamicDataset, ABC):
    def _get_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads encoded/transformed data and generates training, validation, and test splits.
        """

        filename = self.processed_file_names_dict["data"]
        data = self.load_processed_data_from_file(filename)
        df_data = pd.DataFrame(data)

        splits = create_random_splits(
            df_data,
            1 - self.validation_split - self.test_split,
            self.validation_split,
            self.test_split,
            self.dynamic_data_split_seed,
        )
        return splits["train"], splits["validation"], splits["test"]


def create_random_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int | None = 42,
) -> dict[str, pd.DataFrame]:
    """Create random (non-stratified) train/validation/test splits.

    Rows are split purely at random using ``train_test_split`` from
    scikit-learn, with no regard to label distribution or grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    label_start_col : int
        Index of the first label column (default 2). Unused by this
        function; retained for consistency with related split functions.
    train_ratio : float
        Fraction of data for training (default 0.8).
    val_ratio : float
        Fraction of data for validation (default 0.1).
    test_ratio : float
        Fraction of data for testing (default 0.1).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys ``'train'``, ``'validation'``, ``'test'``, each
        containing a DataFrame.

    Raises
    ------
    ValueError
        If the ratios do not sum to 1, any ratio is outside ``[0, 1]``, or
        *label_start_col* is out of range.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if any(r < 0 or r > 1 for r in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("All ratios must be between 0 and 1")

    df_reset = df.reset_index(drop=True)

    # ── Step 1: carve out the test set ──────────────────────────────────────
    df_trainval, df_test = train_test_split(
        df_reset, test_size=test_ratio, shuffle=True, random_state=seed
    )

    # ── Step 2: split train/val from the remaining data ─────────────────────
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_adjusted,
        shuffle=True,
        random_state=seed,
    )

    return {
        "train": df_train.reset_index(drop=True),
        "validation": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }
