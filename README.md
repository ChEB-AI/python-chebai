# ChEBai

ChEBai is a deep learning library designed for the integration of deep learning methods with chemical ontologies, particularly ChEBI. The library emphasizes the incorporation of the semantic qualities of the ontology into the learning process.

## Installation

To install ChEBai, follow these steps:

1. Clone the repository:
```
git clone https://github.com/ChEB-AI/python-chebai.git
```

2. Install the package:

```
cd python-chebai
pip install .
```

## Usage

The training and inference is abstracted using the Pytorch Lightning modules. Here are some quick CLI commands on using ChEBai for pretraining, finetuning, ontology extension and prediction:

### Pretraining
```
python -m chebai fit --data.class_path=chebai.preprocessing.datasets.pubchem.SWJChem --model=configs/model/electra-for-pretraining.yml --trainer=configs/training/default_trainer.yml --trainer.callbacks=configs/training/default_callbacks.yml
```

### Finetuning for predicting classes given SMILES strings

Adapt the pretrained model for specific tasks:

```
python3 -m chebai fit --trainer=configs/training/default_trainer.yml --trainer.logger=configs/training/csv_logger.yml --model=configs/model/electra.yml --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --model.pretrained_checkpoint=electra_pretrained.ckpt --model.load_prefix=generator. --data=configs/data/chebi50.yml --model.out_dim=1446 --model.criterion=configs/loss/bce.yml --data.init_args.batch_size=10 --trainer.logger.init_args.name=chebi50_bce_unweighted --data.init_args.num_workers=9 --model.pass_loss_kwargs=false --data.init_args.chebi_version=227 --data.init_args.data_limit=1000
```
Note: Please make sure you have the pretrained model checkpoint.

### Structure-based ontology extension
```
python -m chebai fit --config=[path-to-your-electra_chebi100-config] --trainer.callbacks=configs/training/default_callbacks.yml  --model.pretrained_checkpoint=[path-to-pretrained-model] --model.load_prefix=generator.
```

### Fine-tuning for Toxicity prediction
```
python -m chebai fit --config=[path-to-your-tox21-config] --trainer.callbacks=configs/training/default_callbacks.yml  --model.pretrained_checkpoint=[path-to-pretrained-model] --model.load_prefix=generator.
```

### Predicting classes given SMILES strings
```
python3 -m chebai predict_from_file --model=[path-to-model-config] --checkpoint_path=[path-to-model] --input_path={path-to-file-containing-smiles] [--classes_path=[path-to-classes-file]] [--save_to=[path-to-output]]
```
The input files should contain a list of line-separated SMILES strings. This generates a CSV file  that contains the
one row for each SMILES string and one column for each class.

## Evaluation

The finetuned model for predicting classes using the SMILES strings can be evaluated using the following python notebook `eval_model_basic.ipynb` in the root dir. It takes in the finetuned model as input for performing the evaluation.

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