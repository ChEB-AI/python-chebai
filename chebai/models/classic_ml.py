import os
import pickle as pkl
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from chebai.models.base import ChebaiBaseNet

LR_MODEL_PATH = os.path.join("models", "LR")


class LogisticRegression(ChebaiBaseNet):
    """
    Logistic Regression model using scikit-learn, wrapped to fit the ChebaiBaseNet interface.
    """

    def __init__(
        self,
        out_dim: int,
        input_dim: int,
        only_predict_classes: Optional[List] = None,
        n_classes=1528,
        **kwargs,
    ):
        super().__init__(out_dim=out_dim, input_dim=input_dim, **kwargs)
        self.models = [
            SklearnLogisticRegression(solver="liblinear") for _ in range(n_classes)
        ]
        # indices of classes (in the dataset used for training) where a model should be trained
        self.only_predict_classes = only_predict_classes

    def forward(self, x: Dict[str, Any], **kwargs) -> torch.Tensor:
        print(
            f"forward called with x[features].shape {x['features'].shape}, self.training {self.training}"
        )
        if self.training:
            self.fit_sklearn(x["features"], x["labels"])
        preds = []
        for model in self.models:
            try:
                p = torch.from_numpy(model.predict(x["features"])).float()
                p = p.to(x["features"].device)
                preds.append(p)
            except NotFittedError:
                preds.append(
                    torch.zeros((x["features"].shape[0]), device=(x["features"].device))
                )
            except AttributeError:
                preds.append(
                    torch.zeros((x["features"].shape[0]), device=(x["features"].device))
                )
        preds = torch.stack(preds, dim=1)
        print(f"preds shape {preds.shape}")
        return preds.squeeze(-1)

    def fit_sklearn(self, X, y):
        """
        Fit the underlying sklearn model. X and y should be numpy arrays.
        """
        for i, model in tqdm.tqdm(enumerate(self.models), desc="Fitting models"):
            import os

            if os.path.exists(os.path.join(LR_MODEL_PATH, f"LR_model_{i}.pkl")):
                print(f"Loading model {i} from file")
                self.models[i] = pkl.load(
                    open(os.path.join(LR_MODEL_PATH, f"LR_model_{i}.pkl"), "rb")
                )
            else:
                if (
                    self.only_predict_classes and i not in self.only_predict_classes
                ):  # only try these classes
                    continue
                try:
                    model.fit(X, y[:, i])
                except ValueError:
                    self.models[i] = PlaceholderModel()
                # dump
                pkl.dump(
                    model, open(os.path.join(LR_MODEL_PATH, f"LR_model_{i}.pkl"), "wb")
                )

    def configure_optimizers(self, **kwargs):
        pass


class PlaceholderModel:
    """Acts like a trained model, but isn't. Use this if training fails and you need a placeholder."""

    def __init__(self, default_prediction=1):
        self.default_prediction = default_prediction

    def predict(self, preds):
        return np.ones(preds.shape[0]) * self.default_prediction
