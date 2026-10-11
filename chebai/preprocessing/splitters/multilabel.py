"""Generate stratified train/validation/test splits from ChEBI DataFrames."""

from __future__ import annotations

from abc import ABC

import pandas as pd

from chebai.preprocessing.datasets.base import _DynamicDataset


class MultiLabelSplitter(_DynamicDataset, ABC):
    def _get_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads encoded/transformed data and generates training, validation, and test splits.
        """

        filename = self.processed_file_names_dict["data"]
        data = self.load_processed_data_from_file(filename)
        df_data = pd.DataFrame(data)

        from chebi_utils import create_multilabel_splits

        splits = create_multilabel_splits(
            df_data,
            self._LABELS_START_IDX,
            1 - self.validation_split - self.test_split,
            self.validation_split,
            self.test_split,
            self.dynamic_data_split_seed,
        )
        return splits["train"], splits["validation"], splits["test"]
