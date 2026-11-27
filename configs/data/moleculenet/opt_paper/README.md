# OPT paper datasets

These configs reproduce the MoleculeNet experiments of the OPT paper. They use the original
dataset classes (`chebai.preprocessing.datasets.molecule_classification`, `tox21.Tox21MolNet*`,
`molecule_regression`, `solCuration`), which split the data randomly (or by group) according to
`test_split` / `validation_split`.

For the DeepChem-based MoleculeNet datasets with predefined splits, use the configs in the
parent directory (`chebai.preprocessing.datasets.molecule_net_classification`).

Related OPT configs:
- model: `configs/model/OPT_experiments/`
- metrics: `configs/metrics/opt_paper/`
- callbacks: `configs/training/opt_paper/`

## Using the splits from the supplementary data

The original splits were not seeded. To reproduce them exactly, place the processed split files from the
supplementary data in the processed directory of the dataset, e.g. for ClinTox with the `ChemDataReader`:

```
data/ClinTox/processed/smiles_token/train.pt
data/ClinTox/processed/smiles_token/validation.pt
data/ClinTox/processed/smiles_token/test.pt
```

If these files exist, they are used as-is and no new splits are created.

ClinTox and SIDER read their raw data from `clintox_groups4.csv` and `sider_groups4.csv` (included in the
supplementary data), which have to be placed in `data/ClinTox/raw/` and `data/Sider/raw/` respectively. These
files are not downloaded automatically. For SIDER, the `group` column is used for a group-based split.
