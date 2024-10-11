import abc
import multiprocessing as mp
from typing import Iterable

import torch
import tqdm

from chebai.models.base import ChebaiBaseNet

PROCESSORS = dict()


class ResultProcessor(abc.ABC):
    @classmethod
    def _identifier(cls) -> str:
        raise NotImplementedError

    def start(self):
        pass

    def close(self):
        pass

    def __init_subclass__(cls, **kwargs):
        assert (
            cls._identifier() not in PROCESSORS
        ), f"ResultProcessor {cls.__name__} does not have a unique identifier"
        PROCESSORS[cls._identifier()] = cls

    def process_prediction(self, proc_id, features, labels, pred, ident):
        raise NotImplementedError


class ResultFactory(abc.ABC):
    def __init__(
        self, model: ChebaiBaseNet, dataset, processors: Iterable[ResultProcessor]
    ):
        self._model = model
        self._reader = dataset.reader
        self.dataset = dataset
        self._processors = processors

    def _process_row(self, row):
        return row

    def _generate_predictions(self, data_path, raw=False, **kwargs):
        self._model.eval()
        collate = self._reader.COLLATOR()
        if raw:
            data_tuples = [
                (x["features"], x["ident"], self._reader.to_data(self._process_row(x)))
                for x in self.dataset._load_dict(data_path)
            ]
        else:
            data_tuples = [
                (x.get("raw_features", x["ident"]), x["ident"], x)
                for x in torch.load(data_path, weights_only=False)
            ]

        for raw_features, ident, row in tqdm.tqdm(data_tuples):
            raw_labels = row.get("labels")

            processable_data = self._model._process_batch(collate([row]), 0)

            model_output = self._model(processable_data)
            preds, labels = self._model._get_prediction_and_labels(
                processable_data, processable_data["labels"], model_output
            )
            d = dict(
                model_output=model_output,
                preds=preds,
                raw_features=raw_features,
                ident=ident,
                threshold=self._model.thres,
            )
            if raw_labels is not None:
                d["labels"] = raw_labels
            yield d

    def call_procs(self, args):
        proc_id, proc_args = args
        for proc in self._processors:
            try:
                proc.process_prediction(proc_id, **proc_args)
            except Exception:
                print("Could not process results for", proc_args["ident"])
                raise

    def execute(self, data_path, **kwargs):
        for proc in self._processors:
            proc.start()
        try:
            with mp.Pool() as pool:
                res = map(
                    self.call_procs,
                    enumerate(self._generate_predictions(data_path, **kwargs)),
                )
            for r in res:
                pass

        except:
            raise
        finally:
            for proc in self._processors:
                proc.close()
