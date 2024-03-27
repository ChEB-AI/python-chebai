# ChEBai Wiki
Welcome to the ChEBai Wiki, a comprehensive guide to understanding and utilizing Chebifier, a tool designed to automate semantic classification within the ChEBI (Chemical Entities of Biological Interest) ontology. This wiki aims to provide users with detailed insights into ChEBai's functionalities, data management, model architectures, and configuration setups.

# Introduction
ChEBai is an API developed for Chebifier, a tool engineered to automate semantic classification within the ChEBI ontology using AI techniques. ChEBI serves as a vast ontology for biologically relevant chemistry, interconnecting chemical structures with meaningful biological and chemical categories. Utilizing a neuro-symbolic AI approach, ChEBai harnesses the ontology's structure to facilitate a learning system that continuously adapts and learns from expanding data, contributing significantly to data-driven chemical knowledge discovery.

For quick setup and usage instructions, please refer to the README.md file.

# Data Management

**Loading ChEBI Ontology Data**

ChEBai accesses the ChEBI ontology data from the following URL: http://purl.obolibrary.org/obo/chebi/{version}/chebi.obo.

**ChEBI versions**

Change the chebi version used for all sets (default: 200):
```
--data.init_args.chebi_version=VERSION
```
To change only the version of the train and validation sets independently of the test set, use
```
--data.init_args.chebi_version_train=VERSION
```

## Data Preprocessing

Upon loading the ontology data, ChEBai undergoes preprocessing, including hierarchy extraction and division into train, validation, and test sets. During preprocessing, a filter is applied to consider only chemical entities with a minimum number of subclasses (e.g., 50 or 100) annotated with SMILES (Simplified Molecular Input Line Entry System) strings.


## Data folder structure
Data is organized within the following directory structure:
```
data/${dataset_name}/${chebi_version}/raw/
```
 
- The raw dataset contains SMILES strings and class columns with boolean values, stored in .pkl format.
``` 
data/${dataset_name}/${chebi_version}/processed/${reader_name}/
```
- ${dataset_name} represents the _name attribute of the DataModule used.
- ${chebi_version} refers to the ChEBI version.
- ${reader_name} denotes the name attribute of the associated Reader class.

In the processed directory, `.pt` is used instead of `.pkl`.

For cross-validation, the folds are stored as `cv_${n_folds}_fold/fold_{fold_index}_train.pkl` 
and `cv_${n_folds}_fold/fold_{fold_index}_validation.pkl` in the raw directory.

# Models

ChEBai employs deep neural network models for semantic classification of chemical entities. Model classes are located within the `chebai/models/` directory.

- `chebai\models\base.py`: Contains the base class ChebaiBaseNet, inherited from PyTorch Lightning module, facilitating custom model creation.

- Example: `chebai\models\electra.py` showcases a custom model inherited from `ChebaiBaseNet`, implementing an Electra model.

# Configurations

ChEBai utilizes PyTorch Lightning for model development, training, and inference, offering structured project organization and enhanced configurability.

The configs are saved in the `configs` folder. Each component such as training, data, model etc., has their own configuration YAML file. 

Configs Folder Structure
- `configs\training\`: Basic trainer, callbacks, and logger configurations.
- `configs\data\`: Configurations for different dataset types used in training and evaluation.
- `configs\loss\`: Custom loss function configurations for training and fine-tuning.
- `configs\metrics\`: Custom configurations for evaluation metrics.
- `configs\model\`: Configurations for different models.
- `configs\weightings\`: Weight values for different datasets used in training.

For detailed information on available arguments for each module, refer to the PyTorch Lightning documentation: [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)