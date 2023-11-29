from torchmetrics.classification import MultilabelF1Score

from chebai.result.base import ResultProcessor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import chebai.models.electra as electra
from chebai.loss.pretraining import ElectraPreLoss
from chebai.preprocessing.datasets.pubchem import PubChemDeepSMILES
import torch
import tqdm


def visualise_loss(logs_path):
    df = pd.read_csv(os.path.join('logs_server', 'pubchem_pretraining', 'version_6', 'metrics.csv'))
    df_loss = df.melt(id_vars='epoch', value_vars=['val_loss_epoch', 'train_loss_epoch'])
    lineplt = sns.lineplot(df_loss, x='epoch', y='value', hue='variable')
    plt.savefig(os.path.join(logs_path, 'loss'))
    plt.show()

# get predictions from model
def evaluate_model(logs_base_path, model_filename):
    data_module = PubChemDeepSMILES(chebi_version=227)
    model = electra.ElectraPre.load_from_checkpoint(os.path.join(logs_base_path, 'checkpoints', model_filename))
    assert isinstance(model, electra.ElectraPre)
    collate = data_module.reader.COLLATER()
    test_file = 'test.pt'
    data_path = os.path.join(data_module.processed_dir, test_file)
    data_list = torch.load(data_path)
    preds_list = []
    labels_list = []

    for row in tqdm.tqdm(data_list):
        processable_data = model._process_batch(collate([row]), 0)
        model_output = model(processable_data, **processable_data['model_kwargs'])
        preds, labels = model._get_prediction_and_labels(processable_data, processable_data["labels"], model_output)
        preds_list.append(preds)
        labels_list.append(labels)

    test_preds = torch.cat(preds_list)
    test_labels = torch.cat(labels_list)
    print(test_preds.shape)
    print(test_labels.shape)
    test_loss = ElectraPreLoss()
    print(f'Loss on test set: {test_loss(test_preds, test_labels)}')
    #f1_macro = MultilabelF1Score(test_preds.shape[1], average='macro')
    #f1_micro = MultilabelF1Score(test_preds.shape[1], average='micro')
    #print(f'Macro-F1 on test set with {test_preds.shape[1]} classes: {f1_macro(test_preds, test_labels):3f}')
    #print(f'Micro-F1 on test set with {test_preds.shape[1]} classes: {f1_micro(test_preds, test_labels):3f}')

class PretrainingResultProcessor(ResultProcessor):

    @classmethod
    def _identifier(cls) -> str:
        return 'PretrainingResultProcessor'

