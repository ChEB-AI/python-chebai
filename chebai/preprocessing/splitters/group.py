"""Generate stratified train/validation/test splits from ChEBI DataFrames."""

from __future__ import annotations

import pandas as pd


def create_group_splits(
    df: pd.DataFrame,
    label_start_col: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int | None = 42,
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

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    from sklearn.model_selection import StratifiedShuffleSplit

    labels_matrix = df.iloc[:, label_start_col:].values
    is_multilabel = labels_matrix.shape[1] > 1
    # StratifiedShuffleSplit requires a 1-D label array
    y = labels_matrix if is_multilabel else labels_matrix[:, 0]

    df_reset = df.reset_index(drop=True)

    # ── Step 1: carve out the test set ──────────────────────────────────────
    if is_multilabel:
        test_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
    else:
        test_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
    train_val_idx, test_idx = next(test_splitter.split(y, y))

    df_test = df_reset.iloc[test_idx]
    df_trainval = df_reset.iloc[train_val_idx]

    # ── Step 2: split train/val from the remaining data ─────────────────────
    y_trainval = y[train_val_idx]
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)

    if is_multilabel:
        val_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio_adjusted, random_state=seed
        )
    else:
        val_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio_adjusted, random_state=seed
        )
    train_idx_inner, val_idx_inner = next(val_splitter.split(y_trainval, y_trainval))

    df_train = df_trainval.iloc[train_idx_inner]
    df_val = df_trainval.iloc[val_idx_inner]

    return {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }
