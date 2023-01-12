import json

from chebai.result.base import ResultProcessor


class JSONResultProcessor(ResultProcessor):
    @classmethod
    def _identifier(cls):
        return "json"

    def start(self):
        self.data = []

    def close(self):
        with open("predictions.json", "w") as fout:
            json.dump(self.data, fout)
            del self.data

    def process_prediction(self, proc_id, raw_features, labels, preds, ident, **kwargs):
        self.data.append(
            dict(
                ident=ident,
                labels=labels if labels is not None else None,
                prediction=preds.tolist(),
            )
        )
