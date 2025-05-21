import importlib
from abc import ABC, abstractmethod
from typing import overload

import torch


class BaseWrapper(ABC):
    def __init__(self, **kwargs):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @overload
    def predict(self, smiles_list: list) -> list:
        pass

    @overload
    def predict(self, data_file_path: str) -> list:
        pass

    def predict(self, x: list | str) -> list:
        if isinstance(x, list):
            return self._predict_from_list_of_smiles(x)
        elif isinstance(x, str):
            return self._predict_from_data_file(x)
        else:
            raise TypeError(f"Type {type(x)} is not supported.")

    @abstractmethod
    def _predict_from_list_of_smiles(self, smiles_list: list) -> list:
        pass

    @abstractmethod
    def _predict_from_data_file(self, data_file_path: str) -> list:
        pass

    @staticmethod
    def _load_class(class_path):
        class_name = class_path.split(".")[-1]
        module_path = ".".join(class_path.split(".")[:-1])
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
