"""Generate stratified train/validation/test splits from ChEBI DataFrames."""

from __future__ import annotations

from abc import ABC

import pandas as pd
from sklearn.model_selection import train_test_split

from chebai.preprocessing.datasets.base import _DynamicDataset


class GeneralSplitter(_DynamicDataset, ABC):
    def _get_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads encoded/transformed data and generates training, validation, and test splits.
        """

        filename = self.processed_file_names_dict["data"]
        data = self.load_processed_data_from_file(filename)
        df_data = pd.DataFrame(data)

        splits = create_general_splits(
            df_data,
            self._LABELS_START_IDX,
            1 - self.validation_split - self.test_split,
            self.validation_split,
            self.test_split,
            self.dynamic_data_split_seed,
        )
        return splits["train"], splits["val"], splits["test"]


def create_general_splits(
    df: pd.DataFrame,
    label_start_col: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Create stratified train/validation/test splits for multilabel DataFrames.

    Columns from index *label_start_col* onwards are treated as binary label
    columns (one boolean column per label).  The stratification strategy is
    chosen automatically based on the number of label columns:

    - More than one label column: ``MultilabelStratifiedShuffleSplit`` from
      the ``iterative-stratification`` package.
    - Single label column: ``StratifiedShuffleSplit`` from ``scikit-learn``.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  Columns ``0`` to ``label_start_col - 1`` are treated as
        feature/metadata columns; all remaining columns are boolean label
        columns.  A typical ChEBI DataFrame has columns
        ``["chebi_id", "mol", "label1", "label2", ...]``.
    label_start_col : int
        Index of the first label column (default 2).
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
        Dictionary with keys ``'train'``, ``'val'``, ``'test'``, each
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
    if label_start_col >= len(df.columns):
        raise ValueError(
            f"label_start_col={label_start_col} is out of range for a DataFrame "
            f"with {len(df.columns)} columns"
        )

    df_reset = df.reset_index(drop=True)

    # ── Step 1: carve out the test set ──────────────────────────────────────
    df_trainval, df_test = train_test_split(
        df_reset, test_size=test_ratio, shuffle=True
    )

    # ── Step 2: split train/val from the remaining data ─────────────────────
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio_adjusted,
        shuffle=True,
    )

    return {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }
