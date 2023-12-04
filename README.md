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
Use inner cross-validation by not splitting between test and validation sets at dataset creation, 
but using k-fold cross-validation at runtime. This creates k models with separate metrics and checkpoints.
For training with `k`-fold cross-validation, use the `cv-fit` subcommand and the options
```
--data.init_args.inner_k_folds=k --n_splits=k
```
## Chebi versions
Change the chebi version used for all sets (default: 200):
```
--data.init_args.chebi_version=VERSION
```
To change only the version of the train and validation sets independently of the test set, use
```
--data.init_args.chebi_version_train=VERSION
```
