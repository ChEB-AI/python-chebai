import abc
from chebai.models.base import JCIBaseNet
from chebai.preprocessing.reader import DataReader
import tqdm
from typing import Iterable
import torch
import multiprocessing as mp

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
        assert cls._identifier() not in PROCESSORS, f"ResultProcessor {cls.__name__} does not have a unique identifier"
        PROCESSORS[cls._identifier()] = cls

    def process_prediction(self, proc_id, raw_features, raw_labels, features, labels, pred):
        raise NotImplementedError


class ResultFactory(abc.ABC):
    def __init__(self, model: JCIBaseNet, dataset, processors: Iterable[ResultProcessor]):
        self._model = model
        self._reader = dataset.reader
        self.dataset = dataset
        self._processors = processors

    def _process_row(self, row):
        return row

    def _generate_predictions(self, data_path):
        self._model.eval()
        for raw_features, raw_labels, features, labels in map(lambda x: (*self._reader._read_components(self._process_row(x)), *self._reader.to_data(self._process_row(x))), tqdm.tqdm(self.dataset._load_tuples(data_path))):
            yield raw_features, raw_labels, features, labels, self._model(torch.tensor(features).unsqueeze(0))

    def call_procs(self, args):
        proc_id, proc_args = args
        for proc in self._processors:
            proc.process_prediction(proc_id, *proc_args)

    def execute(self, data_path):
        for proc in self._processors:
            proc.start()
        try:
            with mp.Pool() as pool:
                res = map(self.call_procs, enumerate(self._generate_predictions(data_path)))
            for r in res:
                pass

        except:
            raise
        finally:
            for proc in self._processors:
                proc.close()

