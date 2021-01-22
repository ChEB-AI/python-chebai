import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

from molecule import Molecule
from model import ChEBIRecNN
import torch.nn as nn
import torch
import csv

NUM_EPOCHS = 10
LEARNING_RATE = 0.01

def eval_model(model, dataset, test_labels):
  raw_values = []
  predictions = []
  final_scores = []

  for idx in range(len(dataset)):
      model_outputs = model(dataset[idx])
      prediction = [1.0 if i > 0.5 else 0.0 for i in model_outputs]
      predictions.append(prediction)
      raw_values.append(model_outputs)
      final_scores.append(f1_score(prediction, test_labels[idx].tolist()))

  avg_f1 = sum(final_scores) / len(final_scores)
  return raw_values, predictions, final_scores, avg_f1


def crawl_info(DAG, sink_parents):
  topological_order = [int(i[0]) for i in DAG]
  target_nodes = [int(i[1]) for i in DAG]
  sink = target_nodes[-1]
  sources = []
  parents = {}

  for i in range(len(topological_order)):
    for j in range(len(target_nodes)):
      if topological_order[i] == target_nodes[j]:
        if topological_order[i] not in parents.keys():
          parents[topological_order[i]] = []
        parents[topological_order[i]].append(topological_order[j])

  for node in topological_order:
    if node not in parents.keys():
      sources.append(node)

  return topological_order, sources, parents, sink, sink_parents


def execute_network(model, loss_fn, optimizer, train_data, train_actual_labels, validation_data, validation_actual_labels, epochs):
    model = model.double()

    columns_name=['epoch', 'train_running_loss', 'train_running_f1', 'eval_running_loss', 'eval_running_f1']
    with open(r'loss_f1_training_validation.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns_name)

    for epoch in range(epochs):
        train_running_loss = 0
        for idx in range(len(train_data)):
            optimizer.zero_grad()
            prediction = model(train_data[idx])
            loss = loss_fn(prediction, train_actual_labels[idx])
            train_running_loss = loss.item()
            loss.backward()
            optimizer.step()
        print('train loss at epoch {} : {:.5f}'.format(epoch, train_running_loss))
        raw_values, predictions, final_scores, train_running_f1 = eval_model(model, train_data, train_actual_labels)
        print('train f1 at epoch {} : {:.5f}'.format(epoch, train_running_f1))

        eval_running_loss = 0
        for eval_idx in range(len(validation_data)):
            eval_prediction = model(validation_data[eval_idx])
            eval_loss = loss_fn(eval_prediction, validation_actual_labels[eval_idx])
            eval_running_loss = eval_loss.item()
        print('validation loss at epoch {} : {:.5f}'.format(epoch, eval_running_loss))
        raw_values, predictions, final_scores, eval_running_f1 = eval_model(model, validation_data, validation_actual_labels)
        print('validation f1 at epoch {} : {:.5f}'.format(epoch, eval_running_f1))


        fields=[epoch, train_running_loss, train_running_f1, eval_running_loss, eval_running_f1]
        with open(r'loss_f1_training_validation.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)


def prepare_data(infile):
    data = pickle.load(infile)[10:]
    infile.close()

    data_frame = pd.DataFrame.from_dict(data)
    data_frame.reset_index(drop=True, inplace=True)

    data_classes = list(data_frame.columns)
    data_classes.remove('MOLECULEID')
    data_classes.remove('SMILES')

    for col in data_classes:
        data_frame[col] = data_frame[col].astype(int)

    train_data = []
    for index, row in data_frame.iterrows():
        train_data.append([
                      data_frame.iloc[index].values[1],
                      data_frame.iloc[index].values[2:502].tolist()
                      ])

    train_df = pd.DataFrame(train_data, columns=['SMILES', 'LABELS'])
    return train_df

print('reading data from files!')
train_infile = open('./data/train.pkl','rb')
test_infile = open('./data/test.pkl','rb')
validation_infile = open('./data/validation.pkl','rb')

train_data = prepare_data(train_infile)
test_data = prepare_data(test_infile)
validation_data = prepare_data(validation_infile)


print('prepare train data!')
train_dataset = []
train_actual_labels = []

for index, row in train_data.iterrows():
    mol = Molecule(row['SMILES'], True)

    DAGs_meta_info = mol.dag_to_node
    train_dataset.append(mol)
    train_actual_labels.append(torch.tensor(row['LABELS']).float())


print('prepare validation data!')
validation_dataset = []
validation_actual_labels = []


for index, row in validation_data.iterrows():
    try:
        mol = Molecule(row['SMILES'], True)

        DAGs_meta_info = mol.dag_to_node

        validation_dataset.append(mol)
        validation_actual_labels.append(torch.tensor(row['LABELS']).float())
    except:
      pass


model = ChEBIRecNN()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

print('num of parameters of the model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

params = {
    'model':model,
    'loss_fn':loss_fn,
    'optimizer':optimizer,
    'train_data': train_dataset,
    'train_actual_labels':train_actual_labels,
    'validation_data': validation_dataset,
    'validation_actual_labels':validation_actual_labels,
    'epochs':NUM_EPOCHS
}

print('start training!')
execute_network(**params)

torch.save(model.state_dict(), 'ChEBI_RvNN_{}_epochs'.format(NUM_EPOCHS))
print('model saved!')
