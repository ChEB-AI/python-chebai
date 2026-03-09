import unittest
from unittest.mock import mock_open, patch

import networkx as nx
import pandas as pd
from rdkit import Chem

from chebai.preprocessing.datasets.chebi import ChEBIOverXPartial
from tests.unit.mock_data.ontology_mock_data import ChebiMockOntology


def _build_mock_chebi_graph() -> nx.DiGraph:
    """
    Build a mock ChEBI graph with is_a relation attributes on edges,
    matching the format produced by chebi_utils.build_chebi_graph.
    Node IDs are strings. Edge direction is child -> parent (is_a).

    Edges derived from the OBO mock data:
        12345 -> 54321 (12345 is_a 54321)
        12345 -> 67890 (12345 is_a 67890)
        54321 -> 11111 (54321 is_a 11111)
        67890 -> 22222 (67890 is_a 22222)
        99999 -> 12345 (99999 is_a 12345)
        88888 -> 67890 (88888 is_a 67890)
    """
    g = nx.DiGraph()
    for node in ChebiMockOntology.get_nodes():
        g.add_node(str(node), smiles="test_smiles_placeholder")
    # child -> parent (matching build_chebi_graph convention)
    is_a_edges = [
        ("12345", "54321"),
        ("12345", "67890"),
        ("54321", "11111"),
        ("67890", "22222"),
        ("99999", "12345"),
        ("88888", "67890"),
    ]
    for src, dst in is_a_edges:
        g.add_edge(src, dst, relation="is_a")
    return g


def _build_mock_mol_df() -> pd.DataFrame:
    """
    Build a mock molecule DataFrame matching the format returned by
    chebi_utils.extract_molecules (columns: chebi_id, mol, ...).
    """
    rows = []
    for smiles, chebi_id in [
        ("C1=CC=CC=C1", "12345"),
        ("C1=CC=CC=C1O", "54321"),
        ("C1=CC=CC=C1N", "67890"),
        ("C1=CC=CC=C1F", "11111"),
        ("C1=CC=CC=C1Cl", "22222"),
        ("C1=CC=CC=C1Br", "99999"),
        ("C1=CC=CC=C1[Mg+]", "88888"),
    ]:
        rows.append({"chebi_id": chebi_id, "mol": Chem.MolFromSmiles(smiles)})
    return pd.DataFrame(rows)


class TestChEBIOverXPartial(unittest.TestCase):
    @classmethod
    @patch.multiple(ChEBIOverXPartial, __abstractmethods__=frozenset())
    @patch("os.makedirs", return_value=None)
    def setUpClass(cls, mock_makedirs) -> None:
        """
        Set up the ChEBIOverXPartial instance with a mock processed directory path and a test graph.
        """
        cls.chebi_extractor = ChEBIOverXPartial(
            top_class_id="11111", external_data_ratio=0, chebi_version=231
        )
        cls.test_graph = _build_mock_chebi_graph()

    @patch("builtins.open", new_callable=mock_open)
    @patch("chebi_utils.extract_molecules")
    def test_graph_to_raw_dataset_no_external(
        self, mock_extract_molecules, mock_open_file
    ) -> None:
        """
        Test _graph_to_raw_dataset with external_data_ratio=0.
        With child->parent edges, predecessors in the transitive closure are descendants.
        For top_class_id="11111", descendants are: 54321, 12345, 99999.
        So included IDs: {11111, 54321, 12345, 99999}.
        """
        mock_extract_molecules.return_value = _build_mock_mol_df()
        self.chebi_extractor.top_class_id = "11111"
        self.chebi_extractor.external_data_ratio = 0
        self.chebi_extractor.THRESHOLD = 1

        data_df = self.chebi_extractor._graph_to_raw_dataset(self.test_graph)

        result_ids = set(data_df["chebi_id"].tolist())
        expected_ids = {"11111", "54321", "12345", "99999"}
        self.assertEqual(
            result_ids,
            expected_ids,
            f"Expected molecule IDs {expected_ids}, got {result_ids}",
        )

        # Verify each row has a valid Mol object
        for _, row in data_df.iterrows():
            self.assertIsInstance(
                row["mol"],
                Chem.Mol,
                f"Expected Mol object for chebi_id={row['chebi_id']}",
            )

    @patch("builtins.open", new_callable=mock_open)
    @patch("chebi_utils.extract_molecules")
    def test_graph_to_raw_dataset_with_external(
        self, mock_extract_molecules, mock_open_file
    ) -> None:
        """
        Test _graph_to_raw_dataset with external_data_ratio=1 (all external nodes included).
        """
        mock_extract_molecules.return_value = _build_mock_mol_df()
        self.chebi_extractor.top_class_id = "11111"
        self.chebi_extractor.external_data_ratio = 1
        self.chebi_extractor.THRESHOLD = 1

        data_df = self.chebi_extractor._graph_to_raw_dataset(self.test_graph)

        # With external_data_ratio=1, all nodes should be included
        result_ids = set(data_df["chebi_id"].tolist())
        expected_ids = {"11111", "54321", "12345", "99999", "22222", "67890", "88888"}
        self.assertEqual(
            result_ids,
            expected_ids,
            f"Expected all molecule IDs {expected_ids}, got {result_ids}",
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("chebi_utils.extract_molecules")
    def test_graph_to_raw_dataset_leaf_class(
        self, mock_extract_molecules, mock_open_file
    ) -> None:
        """
        Test _graph_to_raw_dataset with a leaf node (no descendants).
        For top_class_id="99999", which has no children in the hierarchy,
        only the top class itself should be included.
        """
        mock_extract_molecules.return_value = _build_mock_mol_df()
        self.chebi_extractor.top_class_id = "99999"
        self.chebi_extractor.external_data_ratio = 0
        self.chebi_extractor.THRESHOLD = 1

        data_df = self.chebi_extractor._graph_to_raw_dataset(self.test_graph)

        result_ids = set(data_df["chebi_id"].tolist())
        self.assertEqual(
            result_ids,
            {"99999"},
            f"Expected only leaf node {{'99999'}}, got {result_ids}",
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("chebi_utils.extract_molecules")
    def test_graph_to_raw_dataset_has_label_columns(
        self, mock_extract_molecules, mock_open_file
    ) -> None:
        """
        Test that _graph_to_raw_dataset produces label columns from build_labeled_dataset.
        """
        mock_extract_molecules.return_value = _build_mock_mol_df()
        self.chebi_extractor.top_class_id = "11111"
        self.chebi_extractor.external_data_ratio = 0
        self.chebi_extractor.THRESHOLD = 1

        data_df = self.chebi_extractor._graph_to_raw_dataset(self.test_graph)

        # The returned DataFrame should have chebi_id, mol, and label columns
        self.assertIn("chebi_id", data_df.columns)
        self.assertIn("mol", data_df.columns)
        # Label columns are string IDs of classes that meet the threshold
        label_cols = [c for c in data_df.columns if c not in ("chebi_id", "mol")]
        self.assertGreater(
            len(label_cols), 0, "Expected at least one label column in the result"
        )
        # All label values should be boolean
        for col in label_cols:
            self.assertTrue(
                data_df[col].dtype == bool,
                f"Label column {col} should be boolean, got {data_df[col].dtype}",
            )

    @patch("pandas.DataFrame.to_csv")
    @patch.object(ChEBIOverXPartial, "_get_data_size", return_value=7.0)
    @patch("torch.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("chebi_utils.extract_molecules")
    def test_single_label_data_split(
        self,
        mock_extract_molecules,
        mock_open_file,
        mock_load,
        mock_get_data_size,
        mock_to_csv,
    ) -> None:
        """
        Test the single-label data splitting functionality.

        Mocks file operations and chebi_utils functions to ensure that data is processed
        into a raw dataset and the dynamic data splitting logic produces non-overlapping
        train, validation, and test sets.
        """
        mock_extract_molecules.return_value = _build_mock_mol_df()
        self.chebi_extractor.top_class_id = "99999"
        self.chebi_extractor.THRESHOLD = 1
        self.chebi_extractor.external_data_ratio = 1
        self.chebi_extractor.chebi_version_train = None

        data_df = self.chebi_extractor._graph_to_raw_dataset(self.test_graph)
        self.assertGreater(len(data_df), 0, "DataFrame should not be empty")
        self.assertEqual(
            type([row for _, row in data_df.iterrows()][0]["mol"]),
            Chem.Mol,
            f"No Mol objects in DataFrame: {data_df}",
        )

        # Mock _load_dict to return the expected data structure
        label_start = 2  # chebi_id=0, mol=1, labels from 2 onwards

        def mock_load_dict_generator(path):
            for _, row in data_df.iterrows():
                yield {
                    "features": row["mol"],
                    "labels": row.iloc[label_start:].to_numpy(dtype=bool),
                    "ident": row["chebi_id"],
                }

        with patch.object(
            self.chebi_extractor, "_load_dict", side_effect=mock_load_dict_generator
        ):
            data_pt = self.chebi_extractor._load_data_from_file("fake/path")

        self.assertIsInstance(
            data_pt, list, f"Data_pt should be a list, got {type(data_pt)}"
        )
        self.assertGreater(len(data_pt), 0, "Data_pt should not be empty")

        mock_load.return_value = data_pt

        # Retrieve the data splits (train, validation, and test)
        train_split = self.chebi_extractor.dynamic_split_dfs["train"]
        validation_split = self.chebi_extractor.dynamic_split_dfs["validation"]
        test_split = self.chebi_extractor.dynamic_split_dfs["test"]

        train_idents = set(train_split["ident"])
        val_idents = set(validation_split["ident"])
        test_idents = set(test_split["ident"])

        # Ensure there is no overlap between any pair of splits
        self.assertEqual(
            len(train_idents.intersection(test_idents)),
            0,
            "Train and test sets should not overlap.",
        )
        self.assertEqual(
            len(val_idents.intersection(test_idents)),
            0,
            "Validation and test sets should not overlap.",
        )
        self.assertEqual(
            len(train_idents.intersection(val_idents)),
            0,
            "Train and validation sets should not overlap.",
        )


if __name__ == "__main__":
    unittest.main()
