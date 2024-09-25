import unittest
from typing import Any, Dict

import fastobo
from fastobo.term import TermFrame

from chebai.preprocessing.datasets.chebi import term_callback
from tests.unit.mock_data.ontology_mock_data import ChebiMockOntology


class TestChebiTermCallback(unittest.TestCase):
    """
    Unit tests for the `term_callback` function used in processing ChEBI ontology terms.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test class by loading ChEBI term data and storing it in a dictionary
        where keys are the term IDs and values are TermFrame instances.
        """
        cls.callback_input_data: Dict[int, TermFrame] = {
            int(term_doc.id.local): term_doc
            for term_doc in fastobo.loads(ChebiMockOntology.get_raw_data())
            if term_doc and ":" in str(term_doc.id)
        }

    def test_process_valid_terms(self) -> None:
        """
        Test that `term_callback` correctly processes valid ChEBI terms.
        """

        expected_result: Dict[str, Any] = {
            "id": 12345,
            "parents": [54321, 67890],
            "has_part": set(),
            "name": "Compound A",
            "smiles": "C1=CC=CC=C1",
        }

        actual_dict: Dict[str, Any] = term_callback(
            self.callback_input_data.get(expected_result["id"])
        )
        self.assertEqual(
            expected_result,
            actual_dict,
            msg="term_callback should correctly extract information from valid ChEBI terms.",
        )

    def test_skip_obsolete_terms(self) -> None:
        """
        Test that `term_callback` correctly skips obsolete ChEBI terms.
        """
        term_callback_output = []
        for ident in ChebiMockOntology.get_obsolete_nodes_ids():
            raw_term = self.callback_input_data.get(ident)
            term_dict = term_callback(raw_term)
            if term_dict:
                term_callback_output.append(term_dict)

        self.assertEqual(
            term_callback_output,
            [],
            msg="The term_callback function should skip obsolete terms and return an empty list.",
        )


if __name__ == "__main__":
    unittest.main()
