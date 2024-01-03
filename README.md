# ChEBai

ChEBai  is a deep learning library that allows the combination of deep learning methods with chemical ontologies
(especially ChEBI). Special attention is given to the integration of the semantic qualities of the ontology into the learning process. This is done in two different ways:

## Pretraining

```
python -m chebai fit --data.class_path=chebai.preprocessing.datasets.pubchem.SWJChem --model=configs/model/electra-for-pretraining.yml --trainer=configs/training/default_trainer.yml --trainer.callbacks=configs/training/default_callbacks.yml
```

## Structure-based ontology extension

```
python -m chebai fit --config=[path-to-your-electra_chebi100-config] --trainer.callbacks=configs/training/default_callbacks.yml  --model.pretrained_checkpoint=[path-to-pretrained-model] --model.load_prefix=generator.
```


## Fine-tuning for Toxicity prediction

```
python -m chebai fit --config=[path-to-your-tox21-config] --trainer.callbacks=configs/training/default_callbacks.yml  --model.pretrained_checkpoint=[path-to-pretrained-model] --model.load_prefix=generator.
```

```
python -m chebai train --config=[path-to-your-tox21-config] --trainer.callbacks=configs/training/default_callbacks.yml  --ckpt_path=[path-to-model-with-ontology-pretraining]
```

## Predicting classes given SMILES strings

```
python3 -m chebai predict_from_file --model=[path-to-model-config] --checkpoint_path=[path-to-model] --input_path={path-to-file-containing-smiles] [--classes_path=[path-to-classes-file]] [--save_to=[path-to-output]]
```
The input files should contain a list of line-separated SMILES strings. This generates a CSV file  that contains the
one row for each SMILES string and one column for each class.


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

## Chebi versions
Change the chebi version used for all sets (default: 200):
```
--data.init_args.chebi_version=VERSION
```
To change only the version of the train and validation sets independently of the test set, use
```
--data.init_args.chebi_version_train=VERSION
```

## Data folder structure
Data is stored in and retrieved from the raw and processed folders 
```
data/${dataset_name}/${chebi_version}/raw/
```
and 
``` 
data/${dataset_name}/${chebi_version}/processed/${reader_name}/
```
where `${dataset_name}` is the `_name`-attribute of the `DataModule` used,
`${chebi_version}` refers to the ChEBI version used (only for ChEBI-datasets) and
`${reader_name}` is the `name`-attribute of the `Reader` class associated with the dataset.

For cross-validation, the folds are stored as `cv_${n_folds}_fold/fold_{fold_index}_train.pkl` 
and `cv_${n_folds}_fold/fold_{fold_index}_validation.pkl` in the raw directory.
In the processed directory, `.pt` is used instead of `.pkl`.