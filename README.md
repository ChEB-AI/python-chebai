# ChEB-AI

ChEB-AI is a deep learning library designed for the integration of deep learning methods with chemical ontologies, particularly ChEBI.
The library emphasizes the incorporation of the semantic qualities of the ontology into the learning process. 
Currently, we support datasets from ChEBI, PubChem and MoleculeNet and ELECTRA / FFN models. An extension for Graph Neural Networks is available [here](https://github.com/ChEB-AI/python-chebai-graph).

- [Installation](#Installation)
- [Usage](#Usage)
- [Papers using ChEB-AI](#Papers-using-ChEB-AI)

## News
In version v1.4.0, support for ChEBI25 datasets and Asymmetric Loss got added. For the full changelog, see https://github.com/ChEB-AI/python-chebai/releases.

## Installation

To install ChEB-AI, follow these steps:

1. Clone the repository:
```
git clone https://github.com/ChEB-AI/python-chebai.git
```

2. Install the package:

```
cd python-chebai
pip install -e .
```

Some packages are not installed by default but can be added with the following extras:
```
pip install chebai[dev]
```
installs additional packages useful to people who want to contribute to the library.
This includes `pre-commit`, which runs automatic formatting before each commit.
To set up `pre-commit` for your workflow, run `pre-commit install`.
For more details, see the [`pre-commit` documentation](https://pre-commit.com).

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
python3 -m chebai fit --trainer=configs/training/default_trainer.yml --model=configs/model/electra.yml --model.train_metrics=configs/metrics/micro-macro-f1.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --model.val_metrics=configs/metrics/micro-macro-f1.yml --model.pretrained_checkpoint=electra_pretrained.ckpt --model.load_prefix=generator. --data=configs/data/chebi/chebi50.yml --model.criterion=configs/loss/bce_weighted.yml --data.init_args.batch_size=10 --trainer.logger.init_args.name=chebi50_bce_weighted --data.init_args.num_workers=9 --model.pass_loss_kwargs=false --data.init_args.chebi_version=231 --data.init_args.data_limit=1000
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
python3  chebai/result/prediction.py predict_from_file --checkpoint_path=[path-to-model] --smiles_file_path=[path-to-file-containing-smiles]  [--save_to=[path-to-output]]
```

* **`--checkpoint_path`**: Path to the Lightning checkpoint file (must end with `.ckpt`).

* **`--smiles_file_path`**: Path to a text file containing one SMILES string per line.

* **`--save_to`** *(optional)*: Predictions will be saved to the path as CSV file. The CSV will contain one row per SMILES string and one column per predicted class. Default path will be the current working directory with file name as `predictions.csv`.

> **Note**: Newly created checkpoints after PR #148 must be used for this prediction pipeline. The list of ChEBI classes (classification labels) used during training is stored in new checkpoints, which are required.

## Evaluation

You can evaluate a model trained on the ontology extension task in one of two ways:

### 1. Using the Jupyter Notebook
An example notebook is provided at `tutorials/eval_model_basic.ipynb`.
- Load your finetuned model and run the evaluation cells to compute metrics on the test set.

### 2. Using the Lightning CLI
Alternatively, you can evaluate the model via the CLI:

```bash
python -m chebai test --trainer=configs/training/default_trainer.yml --trainer.devices=1 --trainer.num_nodes=1 --ckpt_path=[path-to-finetuned-model] --model=configs/model/electra.yml --model.test_metrics=configs/metrics/micro-macro-f1.yml --data=configs/data/chebi/chebi50.yml --data.init_args.batch_size=32 --data.init_args.num_workers=10 --data.init_args.chebi_version=[chebi-version] --model.pass_loss_kwargs=false --model.criterion=configs/loss/bce_weighted.yml --model.criterion.init_args.beta=0.99 --data.init_args.splits_file_path=[path-to-splits-file]
```

> **Note**: It is recommended to use `devices=1` and `num_nodes=1` during testing; multi-device settings use a `DistributedSampler`, which may replicate some samples to maintain equal batch sizes, so using a single device ensures that each sample or batch is evaluated exactly once.

## Papers using ChEB-AI

If you are using ChEB-AI, please cite: [Glauer, Martin, et al. "Chebifier: Automating Semantic Classification in ChEBI to Accelerate Data-driven Discovery." Digital Discovery, 2024, 3, 896.](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00238a).

ChEB-AI has since been used for several publications. If you want to reproduce the experiments from a given publications, please use the specified version:

| Paper | ChEB-AI version | Comment |
| --- | --- | --- |
| [Glauer, Martin et al. "Ontology Pre-training for Poison Prediction" KI 2023 (Best Paper Award)](https://link.springer.com/chapter/10.1007/978-3-031-42608-7_4) | [v0.0.0-poison-opt](https://github.com/ChEB-AI/python-chebai/tree/v0.0.0-poison-opt)
| [Glauer, Martin, et al. "Chebifier: Automating Semantic Classification in ChEBI to Accelerate Data-driven Discovery." Digital Discovery, 2024, 3, 896.](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00238a). | [v0.1-chebifier](https://github.com/ChEB-AI/python-chebai/tree/v0.1-chebifier) | supplementary data is available on [Zenodo](https://zenodo.org/records/10247093) |
| [Flügel, Simon, et al. "A Fuzzy Loss for Ontology Classification" NeSy 2024](https://link.springer.com/chapter/10.1007/978-3-031-71167-1_6) | [v1.0.0](https://github.com/ChEB-AI/python-chebai/tree/v1.0.0) | |
| [Memariani, Adel et al. "Box embeddings for extending ontologies: a data-driven and interpretable approach" Journal of Cheminformatics, 2025](https://link.springer.com/article/10.1186/s13321-025-01086-1) | [Box4Chemi](https://github.com/adelmemariani/python-chebai/tree/Box4Chemi) | Fork of ChEB-AI, [supplementary data](https://zenodo.org/records/15286967) |
| [Flügel, Simon and Mossakowski, Till "HyDI: A hybrid Deep Learning-Inductive Logic Programming ensemble for multi-label classification" IJCLR 2026](https://arxiv.org/pdf/2609.37740) | [v1.3.0](https://github.com/ChEB-AI/python-chebai/tree/v1.3.0) | uses the GNN extension [chebai-graph, v1.1.0](https://github.com/ChEB-AI/python-chebai-graph/tree/v1.1.0) and [chebi-utils, v0.1.1](https://github.com/ChEB-AI/python-chebi-utils/tree/v0.1.1), [supplementary material](https://zenodo.org/records/20411715) | 
| [Flügel, Simon et al. "Chebifier 2: An ensemble for chemistry", preprint, 2026](https://www.researchsquare.com/article/rs-9023090/v2) | [v1.4.0](https://github.com/ChEB-AI/python-chebai/tree/v1.4.0) | uses the GNN extension [chebai-graph, v1.2.3](https://github.com/ChEB-AI/python-chebai-graph/tree/v1.2.3), [supplementary material](https://zenodo.org/records/23161410) |
| [Tumescheit, Charlotte et al. "Ontology pre-training improves machine learning-based predictions for metabolites", PLOS Computational Biology, 2026 | todo | |
