from typing import Any, Dict

import torch
import tqdm
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from chebai.models.base import ChebaiBaseNet


class LogisticRegression(ChebaiBaseNet):
    """
    Logistic Regression model using scikit-learn, wrapped to fit the ChebaiBaseNet interface.
    """

    def __init__(self, out_dim: int, input_dim: int, **kwargs):
        super().__init__(out_dim=out_dim, input_dim=input_dim, **kwargs)
        self.models = [SklearnLogisticRegression(solver="liblinear") for _ in range(5)]

    def forward(self, x: Dict[str, Any], **kwargs) -> torch.Tensor:
        print(
            f"forward called with x[features].shape {x['features'].shape}, self.training {self.training}"
        )
        if self.training:
            self.fit_sklearn(x["features"], x["labels"])
        try:
            preds = [
                torch.from_numpy(model.predict(x["features"]))
                .to(
                    x["features"].device
                    if isinstance(x["features"], torch.Tensor)
                    else "cpu"
                )
                .float()
                for model in self.models
            ]
        except NotFittedError:
            # Not fitted yet, return zeros
            print(
                f"returning default 0s with shape {(x['features'].shape[0], self.out_dim)}"
            )
            return torch.zeros(
                (x["features"].shape[0], self.out_dim),
                device=(
                    x["features"].device
                    if isinstance(x["features"], torch.Tensor)
                    else "cpu"
                ),
            )
        preds = torch.stack(preds, dim=1)
        print(f"preds shape {preds.shape}")
        return preds

    def fit_sklearn(self, X, y):
        """
        Fit the underlying sklearn model. X and y should be numpy arrays.
        """
        for i, model in tqdm.tqdm(enumerate(self.models), desc="Fitting models"):
            model.fit(X, y[:, i])

    def configure_optimizers(self, **kwargs):
        pass
