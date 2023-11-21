# ChEBai

ChEBai  is a deep learning library that allows the combination of deep learning methods with chemical ontologies
(especially ChEBI). Special attention is given to the integration of the semantic qualities of the ontology into the learning process. This is done in two different ways:

## Pretraining

```
fit --data.class_path=chebai.preprocessing.datasets.pubchem.SWJChem --model=configs/model/electra-for-pretraining.ElectraPre --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --trainer=configs/training/default_trainer.yml --trainer.callbacks=configs/training/default_callbacks.yml
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