# ChEBai

ChEBai is a deep learning library designed for the integration of deep learning methods with chemical ontologies, particularly ChEBI.
The library emphasizes the incorporation of the semantic qualities of the ontology into the learning process.

##  News

We now support regression tasks!

## Note for developers

If you have used ChEBai before PR #39, the file structure in which your ChEBI-data is saved has changed. This means that
datasets will be freshly generated. The data however is the same. If you want to keep the old data (including the old
splits), you can use a migration script. It copies the old data to the new location for a specific ChEBI class
(including chebi version and other parameters). The script can be called by specifying the data module from a config
```
python chebai/preprocessing/migration/chebi_data_migration.py migrate --datamodule=[path-to-data-config]
```
or by specifying the class name (e.g. `ChEBIOver50`) and arguments separately
```
python chebai/preprocessing/migration/chebi_data_migration.py migrate --class_name=[data-class] [--chebi_version=[version]]
```
The new dataset will by default generate random data splits (with a given seed).
To reuse a fixed data split, you have to provide the path of the csv file generated during the migration:
`--data.init_args.splits_file_path=[path-to-processed_data]/splits.csv`

## Installation

To install ChEBai, follow these steps:

1. Clone the repository:
```
git clone https://github.com/ChEB-AI/python-chebai.git
```

2. Install the package:

```
cd python-chebai
pip install -e .
```

Some packages are not installed by default:
```
pip install chebai[dev]
```
installs additional packages useful to people who want to contribute to the library.
```
pip install chebai[plot]
```
installs additional packages useful for plotting and visualisation.
```
pip install chebai[wandb]
```
installs the [Weights & Biases](https://wandb.ai) integration for automated logging of training runs.
```
pip install chebai[all]
```
installs all optional dependencies.

## Usage

The training and inference is abstracted using the Pytorch Lightning modules.
Here are some CLI commands for the standard functionalities of pretraining, ontology extension, fine-tuning for toxicity and prediction.
For further details, see the [wiki](https://github.com/ChEB-AI/python-chebai/wiki).
If you face any problems, please open a new [issue](https://github.com/ChEB-AI/python-chebai/issues/new).

### Pretraining
```
python -m chebai fit --data.class_path=chebai.preprocessing.datasets.pubchem.PubchemChem --model=configs/model/electra-for-pretraining.yml --trainer=configs/training/pretraining_trainer.yml
```

### Structure-based ontology extension
```
python -m chebai fit --trainer=configs/training/default_trainer.yml --model=configs/model/electra.yml  --model.pretrained_checkpoint=[path-to-pretrained-model] --model.load_prefix=generator. --data=[path-to-dataset-config] --model.out_dim=[number-of-labels]
```
A command with additional options may look like this:
```
python3 -m chebai fit --trainer=configs/training/default_trainer.yml --model=configs/model/electra.yml --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --model.pretrained_checkpoint=electra_pretrained.ckpt --model.load_prefix=generator. --data=configs/data/chebi50.yml --model.criterion=configs/loss/bce.yml --data.init_args.batch_size=10 --trainer.logger.init_args.name=chebi50_bce_unweighted --data.init_args.num_workers=9 --model.pass_loss_kwargs=false --data.init_args.chebi_version=231 --data.init_args.data_limit=1000
```

### Fine-tuning for classification tasks, e.g. Toxicity prediction
```
python -m chebai fit --config=[path-to-your-tox21-config] --trainer.callbacks=configs/training/default_callbacks.yml  --model.pretrained_checkpoint=[path-to-pretrained-model]
```

### Fine-tuning for regression tasks, e.g. solubility prediction
```
python -m chebai fit --config=[path-to-your-esol-config] --trainer.callbacks=configs/training/solCur_callbacks.yml  --model.pretrained_checkpoint=[path-to-pretrained-model]
```

### Predicting classes given SMILES strings
```
python3 -m chebai predict_from_file --model=[path-to-model-config] --checkpoint_path=[path-to-model] --input_path={path-to-file-containing-smiles] [--classes_path=[path-to-classes-file]] [--save_to=[path-to-output]]
```
The input files should contain a list of line-separated SMILES strings. This generates a CSV file  that contains the
one row for each SMILES string and one column for each class.
The `classes_path` is the path to the dataset's `raw/classes.txt` file that contains the relationship between model output and ChEBI-IDs.

## Evaluation

You can evaluate a model trained on the ontology extension task in one of two ways:

### 1. Using the Jupyter Notebook
An example notebook is provided at `tutorials/eval_model_basic.ipynb`.
- Load your finetuned model and run the evaluation cells to compute metrics on the test set.

### 2. Using the Lightning CLI
Alternatively, you can evaluate the model via the CLI:

```bash
python -m chebai test --trainer=configs/training/default_trainer.yml --trainer.devices=1 --trainer.num_nodes=1 --ckpt_path=[path-to-finetuned-model] --model=configs/model/electra.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --data=configs/data/chebi/chebi50.yml --data.init_args.batch_size=32 --data.init_args.num_workers=10 --data.init_args.chebi_version=[chebi-version] --model.pass_loss_kwargs=false --model.criterion=configs/loss/bce.yml --model.criterion.init_args.beta=0.99 --data.init_args.splits_file_path=[path-to-splits-file]
```

> **Note**: It is recommended to use `devices=1` and `num_nodes=1` during testing; multi-device settings use a `DistributedSampler`, which may replicate some samples to maintain equal batch sizes, so using a single device ensures that each sample or batch is evaluated exactly once.


## Cross-validation
You can do inner k-fold cross-validation, i.e., train models on k train-validation splits that all use the same test
set. For that, you need to specify the total_number of folds as
```
--data.init_args.inner_k_folds=K
```
and the fold to be used in the current optimisation run as
```
--data.init_args.fold_index=I
```
To train K models, you need to do K such calls, each with a different `fold_index`. On the first call with a given
`inner_k_folds`, all folds will be created and stored in the data directory

## Note for developers

If you have used ChEBai before PR #39, the file structure in which your ChEBI-data is saved has changed. This means that
datasets will be freshly generated. The data however is the same. If you want to keep the old data (including the old
splits), you can use a migration script. It copies the old data to the new location for a specific ChEBI class
(including chebi version and other parameters). The script can be called by specifying the data module from a config
```
python chebai/preprocessing/migration/chebi_data_migration.py migrate --datamodule=[path-to-data-config]
```
or by specifying the class name (e.g. `ChEBIOver50`) and arguments separately
```
python chebai/preprocessing/migration/chebi_data_migration.py migrate --class_name=[data-class] [--chebi_version=[version]]
```
The new dataset will by default generate random data splits (with a given seed).
To reuse a fixed data split, you have to provide the path of the csv file generated during the migration:
`--data.init_args.splits_file_path=[path-to-processed_data]/splits.csv`
