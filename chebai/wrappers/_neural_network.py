import os
from typing import Optional, Type

import torch
from rdkit import Chem

from chebai.models import ChebaiBaseNet
from chebai.preprocessing.reader import DataReader

from ._base import BaseWrapper


class NNWrapper(BaseWrapper):

    def __init__(
        self,
        model: ChebaiBaseNet,
        reader_cls: Type[DataReader],
        reader_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model: ChebaiBaseNet = model
        if reader_kwargs is None:
            reader_kwargs = dict()
        self.reader = reader_cls(**reader_kwargs)
        self.collator = reader_cls.COLLATOR()

    def _forward_pass(self, batch):
        processable_data = self.model._process_batch(
            self.collator(batch).to(self._device), 0
        )
        return self.model(processable_data, **processable_data["model_kwargs"])

    def _read_smiles(self, smiles):
        return self.reader.to_data(dict(features=smiles, labels=None))

    def _predict_from_list_of_smiles(self, smiles_list) -> list:
        token_dicts = []
        could_not_parse = []
        index_map = dict()
        for i, smiles in enumerate(smiles_list):
            try:
                # Try to parse the smiles string
                if not smiles:
                    raise ValueError()
                d = self._read_smiles(smiles)
                # This is just for sanity checks
                rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
            except Exception as e:
                # Note if it fails
                could_not_parse.append(i)
                print(f"Failing to parse {smiles} due to {e}")
            else:
                if rdmol is None:
                    could_not_parse.append(i)
                else:
                    index_map[i] = len(token_dicts)
                    token_dicts.append(d)
        print(f"Predicting {len(token_dicts), token_dicts} out of {len(smiles_list)}")
        if token_dicts:
            model_output = self._forward_pass(token_dicts)
            if not isinstance(model_output, dict) and not "logits" in model_output:
                raise ValueError()
            return model_output
        else:
            raise ValueError()

    def _predict_from_data_file(
        self, processed_dir_main: str, data_file_name="data.pt"
    ) -> list:
        data = torch.load(
            os.path.join(processed_dir_main, self.reader.name(), data_file_name),
            weights_only=False,
            map_location=self._device,
        )
        return self._forward_pass(data)
