from chebai.models import ChebiBox
from chebai.preprocessing.datasets.chebi import ChEBIOver100 #, ChEBIOver50
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from collections import defaultdict
import networkx as nx
from os import listdir
from os.path import isfile, join
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision

if __name__ == '__main__':

    model_to_eval = 'w_cui' #'w_norm'  

    if model_to_eval == 'w_cui':
        current_dir = './logs/ChebiBox/version_215/checkpoints/last/'
    
    if model_to_eval == 'w_norm':
        current_dir = './logs/ChebiBox/version_236/checkpoints/last/'
    
    model_checkpoints = [f for f in listdir(current_dir) if isfile(join(current_dir, f))]

    for chkpnt in model_checkpoints:
        print(current_dir + chkpnt)
        print('Model checkpoint: ', chkpnt.split('/')[-1])
        
        model = ChebiBox
        myModel = model.load_from_checkpoint(current_dir + chkpnt, map_location=torch.device('cpu'))
        myModel.eval()

        data_module = ChEBIOver100()

        dataset_to_eval = 'val'

        if dataset_to_eval == 'test':
            dm = data_module.test_dataloader()
        
        if dataset_to_eval == 'val':
            dm = data_module.val_dataloader()
        
        if dataset_to_eval == 'train':
            dm = data_module.train_dataloader()

        preds_list = []
        labels_list = []
        points = []

        for batch_idx, batch in enumerate(dm):
                processable_data = myModel._process_batch(batch, batch_idx)
                model_output = myModel(processable_data)
                points.append(model_output['embedded_points'].cpu().detach().numpy()[0])
                preds, labels = myModel._get_prediction_and_labels(
                                    processable_data, processable_data["labels"], model_output
                                )
                preds_list.append(preds)
                labels_list.append(labels)
                
                if batch_idx % 1000 == 0:
                    print(batch_idx)

        with open('./evaluations/' + model_to_eval + '/points/' + dataset_to_eval + '/embedded_points.pkl', 'wb') as points_file:
            pickle.dump(points, points_file)
        points_file.close()

        with open('./evaluations/' + model_to_eval + '/predictions/' + dataset_to_eval + '/predictions.pkl', 'wb') as pred_file:
            pickle.dump(preds_list, pred_file)
        pred_file.close()

        #with open('./evaluations/w_cui/predictions/train/w_cui_labels_train.pkl', 'wb') as label_file:
        #    pickle.dump(labels_list, label_file)
        #label_file.close()

        test_preds = torch.cat(preds_list)
        test_labels = torch.cat(labels_list)
        print(test_preds.shape)
        print(test_labels.shape)
        f1_macro = MultilabelF1Score(test_preds.shape[1], average="macro")
        f1_micro = MultilabelF1Score(test_preds.shape[1], average="micro")
        f1_weighted = MultilabelF1Score(test_preds.shape[1], average="weighted")

        recall_macro = MultilabelRecall(test_preds.shape[1], average="macro")
        recall_micro = MultilabelRecall(test_preds.shape[1], average="micro")
        recall_weighted = MultilabelRecall(test_preds.shape[1], average="weighted")

        precision_macro = MultilabelPrecision(test_preds.shape[1], average="macro")
        precision_micro = MultilabelPrecision(test_preds.shape[1], average="micro")
        precision_weighted = MultilabelPrecision(test_preds.shape[1], average="weighted")

        print(
            f"Macro-F1 on test set with {test_preds.shape[1]} classes: {f1_macro(test_preds, test_labels):3f}"
        )
        print(
            f"Micro-F1 on test set with {test_preds.shape[1]} classes: {f1_micro(test_preds, test_labels):3f}"
        )   
        print(
            f"Weighted-F1 on test set with {test_preds.shape[1]} classes: {f1_weighted(test_preds, test_labels):3f}"
        )

        print(
            f"Macro-Recall on test set with {test_preds.shape[1]} classes: {recall_macro(test_preds, test_labels):3f}"
        )
        print(
            f"Micro-Recall on test set with {test_preds.shape[1]} classes: {recall_micro(test_preds, test_labels):3f}"
        )
        print(
            f"Weighted-Recall on test set with {test_preds.shape[1]} classes: {recall_weighted(test_preds, test_labels):3f}"
        )

        print(
            f"Macro-Precision-F1 on test set with {test_preds.shape[1]} classes: {precision_macro(test_preds, test_labels):3f}"
        )
        print(
            f"Micro-Precision-F1 on test set with {test_preds.shape[1]} classes: {precision_micro(test_preds, test_labels):3f}"
        )
        print(
            f"Weighted-Precision-F1 on test set with {test_preds.shape[1]} classes: {precision_weighted(test_preds, test_labels):3f}"
        )
