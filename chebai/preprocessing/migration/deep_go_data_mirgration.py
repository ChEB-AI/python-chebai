from typing import List

import pandas as pd

# https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/utils.py#L18-L22
NAMESPACES = {
    "cc": "cellular_component",
    "mf": "molecular_function",
    "bp": "biological_process",
}

# https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/aminoacids.py#L11
MAXLEN = 1000


def load_data(data_dir):
    test_df = pd.DataFrame(pd.read_pickle("test_data.pkl"))
    train_df = pd.DataFrame(pd.read_pickle("train_data.pkl"))
    validation_df = pd.DataFrame(pd.read_pickle("valid_data.pkl"))

    required_columns = [
        "proteins",
        "accessions",
        "sequences",
        # https://github.com/bio-ontology-research-group/deepgo2/blob/main/gendata/uni2pandas.py#L45-L58
        "exp_annotations",  # Directly associated GO ids
        # https://github.com/bio-ontology-research-group/deepgo2/blob/main/gendata/uni2pandas.py#L60-L69
        "prop_annotations",  # Transitively associated GO ids
    ]

    new_df = pd.concat(
        [
            train_df[required_columns],
            validation_df[required_columns],
            test_df[required_columns],
        ],
        ignore_index=True,
    )
    # Generate splits.csv file to store ids of each corresponding split
    split_assignment_list: List[pd.DataFrame] = [
        pd.DataFrame({"id": train_df["proteins"], "split": "train"}),
        pd.DataFrame({"id": validation_df["proteins"], "split": "validation"}),
        pd.DataFrame({"id": test_df["proteins"], "split": "test"}),
    ]

    combined_split_assignment = pd.concat(split_assignment_list, ignore_index=True)


def save_data(data_dir, data_df):
    pass


if __name__ == "__main__":
    pass
