import torch
from chebai.preprocessing.datasets.chebi import _ChEBIDataExtractor
import pandas as pd
import os
import pickle


class BCEWeighted(torch.nn.BCEWithLogitsLoss):
    """BCEWithLogitsLoss with weights automatically computed according to beta parameter (formula from
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)
    """

    def __init__(self, beta: float = None, data_extractor: _ChEBIDataExtractor = None):
        self.beta = beta
        self.data_extractor = data_extractor
        super().__init__()

    def set_pos_weight(self):
        if (
            self.beta is not None
            and self.data_extractor is not None
            and all(
                os.path.exists(os.path.join(self.data_extractor.raw_dir, raw_file))
                for raw_file in self.data_extractor.raw_file_names
            )
        ):
            complete_data = pd.concat(
                [
                    pickle.load(
                        open(
                            os.path.join(
                                self.data_extractor.raw_dir,
                                self.data_extractor.raw_file_names_dict[set],
                            ),
                            "rb",
                        )
                    )
                    for set in ["train", "validation", "test"]
                ]
            )
            value_counts = []
            for c in complete_data.columns[3:]:
                value_counts.append(len([v for v in complete_data[c] if v]))
            weights = [(1 - beta) / (1 - pow(beta, value)) for value in value_counts]
            mean = sum(weights) / len(weights)
            self.pos_weight = torch.Tensor([w / mean for w in weights])

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.set_pos_weight()
        return super().forward(input, target)
