import unittest

from rdkit import Chem

from chebai.preprocessing.reader import StaticSMILESReader
from chebai.preprocessing.smiles_tokenizer import UNKNOWN_TOKEN_IDX


class TestStaticSMILESReader(unittest.TestCase):
    """
    Unit tests for the StaticSMILESReader class.

    Focuses on two core properties:
    - Determinism: the same SMILES always produces the same token sequence.
    - Decode correctness: token sequences decode back to the canonical SMILES.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.reader = StaticSMILESReader()

    @staticmethod
    def _canonical(smiles: str) -> str:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    # --- Determinism tests ---

    def test_same_input_produces_same_tokens(self) -> None:
        """Calling _read_data with two instances on the same SMILES gives identical token sequences."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        result1 = self.reader._read_data(smiles)
        new_reader = (
            StaticSMILESReader()
        )  # Create a new reader instance to ensure no shared state
        result2 = new_reader._read_data(smiles)
        self.assertIsNotNone(result1)
        self.assertEqual(result1, result2)

    def test_non_canonical_input_matches_canonical(self) -> None:
        """Different SMILES representations of the same molecule produce identical tokens."""
        canonical = "CC(=O)Oc1ccccc1C(=O)O"
        non_canonical = "OC(=O)c1ccccc1OC(C)=O"
        result_canonical = self.reader._read_data(canonical)
        result_non_canonical = self.reader._read_data(non_canonical)
        self.assertIsNotNone(result_non_canonical)
        self.assertEqual(
            result_canonical,
            result_non_canonical,
            "Non-canonical and canonical SMILES of the same molecule produced different token sequences.",
        )

    # --- Static vocabulary tests ---

    def test_vocabulary_does_not_grow(self) -> None:
        """Encoding multiple novel SMILES strings never increases the vocabulary size."""
        initial_vocab_size = len(self.reader.tokenizer.vocab)
        smiles_list = [
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
            "C[C@H](N)C(=O)O",  # L-alanine
            "[13CH3]CO",  # isotope
            "c1ccc2c(c1)[nH]cn2",  # benzimidazole
            "[NH4+]",  # charged atom
        ]
        for smiles in smiles_list:
            self.reader._read_data(smiles)
        self.assertEqual(
            len(self.reader.tokenizer.vocab),
            initial_vocab_size,
            "Vocabulary grew after encoding SMILES strings — StaticSMILESReader must not add new tokens.",
        )

    def test_unknown_tokens_use_unknown_idx(self) -> None:
        """Tokens outside the static vocabulary are mapped to UNKNOWN_TOKEN_IDX, not added."""
        # [123] has no element symbol, so _parse_bracket_atom produces element_None
        # which is not in the vocabulary.
        token_ids = self.reader.tokenizer.encode("[123]")
        self.assertIn(
            UNKNOWN_TOKEN_IDX,
            token_ids,
            f"Expected UNKNOWN_TOKEN_IDX ({UNKNOWN_TOKEN_IDX}) in encoded output for out-of-vocabulary token.",
        )

    # --- Decode roundtrip tests ---

    def test_decode_roundtrip_simple_organic(self) -> None:
        """Encoding then decoding recovers the canonical SMILES for a simple organic molecule."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        canonical = self._canonical(smiles)
        token_ids = self.reader._read_data(smiles)
        decoded = self.reader.tokenizer.decode(token_ids)
        self.assertEqual(decoded, canonical)

    def test_decode_roundtrip_stereo(self) -> None:
        """Encoding then decoding recovers canonical SMILES for a molecule with stereochemistry."""
        smiles = "C[C@H](N)C(=O)O"  # L-alanine
        canonical = self._canonical(smiles)
        token_ids = self.reader._read_data(smiles)
        decoded = self.reader.tokenizer.decode(token_ids)
        self.assertEqual(decoded, canonical)

    def test_decode_roundtrip_isotope(self) -> None:
        """Encoding then decoding recovers canonical SMILES for a molecule with an isotope label."""
        smiles = "[13CH3]CO"
        canonical = self._canonical(smiles)
        token_ids = self.reader._read_data(smiles)
        decoded = self.reader.tokenizer.decode(token_ids)
        self.assertEqual(decoded, canonical)

    def test_decode_roundtrip_charged_atom(self) -> None:
        """Encoding then decoding recovers canonical SMILES for a molecule with a charged atom."""
        smiles = "[NH4+]"
        canonical = self._canonical(smiles)
        token_ids = self.reader._read_data(smiles)
        decoded = self.reader.tokenizer.decode(token_ids)
        self.assertEqual(decoded, canonical)

    def test_decode_roundtrip_bracket_aromatic(self) -> None:
        """Encoding then decoding recovers canonical SMILES for an aromatic ring with a bracketed atom."""
        smiles = "c1ccc2c(c1)[nH]cn2"  # benzimidazole
        canonical = self._canonical(smiles)
        token_ids = self.reader._read_data(smiles)
        decoded = self.reader.tokenizer.decode(token_ids)
        self.assertEqual(decoded, canonical)

    # --- Invalid input tests ---

    def test_invalid_smiles_returns_none(self) -> None:
        """Invalid SMILES strings cause _read_data to return None."""
        invalid_smiles = ["%INVALID%", "ADADAD", "ADASDAD"]
        for smiles in invalid_smiles:
            result = self.reader._read_data(smiles)
            self.assertIsNone(
                result,
                f"Expected None for invalid SMILES '{smiles}', got {result!r}.",
            )


if __name__ == "__main__":
    unittest.main()
