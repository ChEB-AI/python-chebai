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
        if beta is not None and data_extractor is not None:
            complete_data = pd.concat(
                [
                    pickle.load(
                        open(
                            os.path.join(
                                data_extractor.raw_dir,
                                data_extractor.raw_file_names_dict[set],
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
            weights = torch.Tensor([w / mean for w in weights])
            super(BCEWeighted).__init__(pos_weight=weights)
        else:
            super(BCEWeighted)
