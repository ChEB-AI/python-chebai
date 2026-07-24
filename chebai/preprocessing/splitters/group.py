"""Generate stratified train/validation/test splits from ChEBI DataFrames."""

from __future__ import annotations

from abc import ABC

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from chebai.preprocessing.datasets.base import _DynamicDataset


class GroupSplitter(_DynamicDataset, ABC):
    def _get_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads encoded/transformed data and generates training, validation, and test splits.
        """

        filename = self.processed_file_names_dict["data"]
        data = self.load_processed_data_from_file(filename)
        df_data = pd.DataFrame(data)

        splits = create_group_splits(
            df_data,
            self._LABELS_START_IDX,
            1 - self.validation_split - self.test_split,
            self.validation_split,
            self.test_split,
            self.dynamic_data_split_seed,
        )
        return splits["train"], splits["validation"], splits["test"]


def create_group_splits(
    df: pd.DataFrame,
    label_start_col: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int | None = 42,
) -> dict[str, pd.DataFrame]:
    """Create group-based train/validation/test splits for DataFrames.

    Splitting is done with ``GroupShuffleSplit`` using the ``group`` column,
    so that all rows sharing the same group value are assigned to the same
    split (no group leaks across train/val/test). This is **not** a
    stratified split: label balance across splits is not guaranteed, even
    though label columns are used to build the ``y`` array passed to the
    splitter (``GroupShuffleSplit`` ignores label values and only inspects
    the ``groups`` argument).

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  Columns ``0`` to ``label_start_col - 1`` are treated as
        feature/metadata columns; all remaining columns are boolean label
        columns.  A typical ChEBI DataFrame has columns
        ``["chebi_id", "mol", "label1", "label2", ...]``. A ``group`` column
        must also be present and is used to keep related rows together.
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
        Dictionary with keys ``'train'``, ``'validation'``, ``'test'``, each
        containing a DataFrame.

    Raises
    ------
    ValueError
        If the ratios do not sum to 1, any ratio is outside ``[0, 1]``,
        *label_start_col* is out of range, the ``group`` column is missing,
        or fewer than 2 unique groups are present.
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

    if "group" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain a 'group' column for group split"
        )

    if len(df["group"].unique()) < 2:
        raise ValueError(
            "Input DataFrame must contain at least 2 unique groups for group split"
        )

    y = df.iloc[:, label_start_col:].values
    # StratifiedShuffleSplit requires a 1-D label array

    df_reset = df.reset_index(drop=True)

    # ── Step 1: carve out the test set ──────────────────────────────────────
    test_splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )

    train_val_idx, test_idx = next(test_splitter.split(y, y, groups=df_reset["group"]))

    df_test = df_reset.iloc[test_idx]
    df_trainval = df_reset.iloc[train_val_idx]

    # ── Step 2: split train/val from the remaining data ─────────────────────
    y_trainval = y[train_val_idx]
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)

    val_splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=seed
    )

    train_idx_inner, val_idx_inner = next(
        val_splitter.split(y_trainval, y_trainval, groups=df_trainval["group"])
    )

    df_train = df_trainval.iloc[train_idx_inner]
    df_val = df_trainval.iloc[val_idx_inner]

    return {
        "train": df_train.reset_index(drop=True),
        "validation": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }
