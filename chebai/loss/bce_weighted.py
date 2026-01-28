import os
from typing import Optional

import torch

from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.preprocessing.datasets.chebi import _ChEBIDataExtractor


class BCEWeighted(torch.nn.BCEWithLogitsLoss):
    """
    BCEWithLogitsLoss with weights automatically computed according to the beta parameter.

    This class computes weights based on the formula from the paper by Cui et al. (2019):
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

    Args:
        beta (float, optional): The beta parameter for weight calculation. Default is None.
        data_extractor (XYBaseDataModule, optional): The data extractor for loading the dataset. Default is None.
    """

    def __init__(
        self,
        beta: float = 0.99,
        data_extractor: Optional[XYBaseDataModule] = None,
        **kwargs,
    ):
        from chebai.preprocessing.datasets.pubchem import LabeledUnlabeledMixed

        self.beta = beta
        if isinstance(data_extractor, LabeledUnlabeledMixed):
            data_extractor = data_extractor.labeled
        self.data_extractor = data_extractor

        assert (
            isinstance(beta, float) and beta >= 0.0 and beta <= 1.0
        ), f"Beta parameter must be a float with value between 0 and 1, for loss class {self.__class__.__name__}."

        assert (
            self.data_extractor is not None
        ), f"Data extractor must be provided if this loss class ({self.__class__.__name__}) is used."

        assert all(
            os.path.exists(os.path.join(self.data_extractor.processed_dir, file_name))
            for file_name in self.data_extractor.processed_file_names
        ), "Dataset files not found. Make sure the dataset is processed before using this loss."

        assert (
            isinstance(self.data_extractor, _ChEBIDataExtractor)
            or self.data_extractor is None
        )
        super().__init__(**kwargs)
        self.pos_weight: Optional[torch.Tensor] = None

    def set_pos_weight(self, input: torch.Tensor) -> None:
        """
        Sets the positive weights for the loss function based on the input tensor.

        Args:
            input (torch.Tensor): The input tensor for which to set the positive weights.
        """
        if self.pos_weight is None:
            print(
                f"Computing loss-weights based on v{self.data_extractor.chebi_version} dataset (beta={self.beta})"
            )
            print(
                f"loading: {','.join(f for f in self.data_extractor.processed_file_names)}"
            )
            complete_labels = torch.concat(
                [
                    torch.stack(
                        [
                            torch.Tensor(row["labels"])
                            for row in self.data_extractor.load_processed_data(
                                filename=os.path.join(file_name)
                            )
                        ]
                    )
                    for file_name in self.data_extractor.processed_file_names
                ]
            )
            value_counts = complete_labels.sum(dim=0)
            weights = [
                (1 - self.beta) / (1 - pow(self.beta, value)) for value in value_counts
            ]
            mean = sum(weights) / len(weights)
            self.pos_weight = torch.tensor(
                [w / mean for w in weights], device=input.device
            )

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for the loss calculation.

        Args:
            input (torch.Tensor): The input tensor (predictions).
            target (torch.Tensor): The target tensor (labels).

        Returns:
            torch.Tensor: The computed loss.
        """
        self.set_pos_weight(input)
        return super().forward(input, target)


class UnWeightedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, input, target, **kwargs):
        # As the custom passed kwargs are not used in BCEWithLogitsLoss, we can ignore them
        return super().forward(input, target)
