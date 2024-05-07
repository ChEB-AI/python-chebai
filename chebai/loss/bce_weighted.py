import torch
from chebai.preprocessing.datasets.base import XYBaseDataModule
from chebai.preprocessing.datasets.pubchem import LabeledUnlabeledMixed
import pandas as pd
import os
import pickle


class BCEWeighted(torch.nn.BCEWithLogitsLoss):
    """BCEWithLogitsLoss with weights automatically computed according to beta parameter (formula from
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)
    """

    def __init__(
        self,
        beta: float = None,
        data_extractor: XYBaseDataModule = None,
    ):
        self.beta = beta
        if isinstance(data_extractor, LabeledUnlabeledMixed):
            data_extractor = data_extractor.labeled
        self.data_extractor = data_extractor

        super().__init__()

    def set_pos_weight(self, input):
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
                    pickle.load(
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
        self.set_pos_weight(input)
        return super().forward(input, target)
