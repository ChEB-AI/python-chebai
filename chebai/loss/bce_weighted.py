from typing import Optional

import torch

from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.preprocessing.datasets.pubchem import LabeledUnlabeledMixed
import pandas as pd
import os


class BCEWeighted(torch.nn.BCEWithLogitsLoss):
    """
    BCEWithLogitsLoss with weights automatically computed according to the beta parameter.

    This class computes weights based on the formula from the paper:
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

    Args:
        beta (float, optional): The beta parameter for weight calculation. Default is None.
        data_extractor (XYBaseDataModule, optional): The data extractor for loading the dataset. Default is None.
    """

    def __init__(
        self,
        beta: Optional[float] = None,
        data_extractor: Optional[XYBaseDataModule] = None,
    ):
        self.beta = beta
        if isinstance(data_extractor, LabeledUnlabeledMixed):
            data_extractor = data_extractor.labeled
        self.data_extractor = data_extractor

        super().__init__()

    def set_pos_weight(self, input: torch.Tensor) -> None:
        """
        Sets the positive weights for the loss function based on the input tensor.

        Args:
            input (torch.Tensor): The input tensor for which to set the positive weights.
        """
        if (
            self.beta is not None
            and self.data_extractor is not None
            and all(
                os.path.exists(os.path.join(self.data_extractor.raw_dir, raw_file))
                for raw_file in self.data_extractor.raw_file_names
            )
            and self.pos_weight is None
        ):
            complete_data = pd.concat(
                [
                    pd.read_pickle(
                        open(
                            os.path.join(
                                self.data_extractor.raw_dir,
                                raw_file_name,
                            ),
                            "rb",
                        )
                    )
                    for raw_file_name in self.data_extractor.raw_file_names
                ]
            )
            value_counts = []
            for c in complete_data.columns[3:]:
                value_counts.append(len([v for v in complete_data[c] if v]))
            weights = [
                (1 - self.beta) / (1 - pow(self.beta, value)) for value in value_counts
            ]
            mean = sum(weights) / len(weights)
            self.pos_weight = torch.tensor(
                [w / mean for w in weights], device=input.device
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
