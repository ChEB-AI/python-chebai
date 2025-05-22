import os
import sys
from typing import Optional

sys.path.append(os.path.join("..", "..", "..", "PycharmProjects", "chemlog2"))
import chemlog
from chebi_utils import CHEBI_FRAGMENT, get_transitive_predictions
from chemlog.classification.charge_classifier import (
    ChargeCategories,
    get_charge_category,
)
from chemlog.classification.peptide_size_classifier import get_n_amino_acid_residues
from chemlog.classification.proteinogenics_classifier import (
    get_proteinogenic_amino_acids,
)
from chemlog.classification.substructure_classifier import (
    is_diketopiperazine,
    is_emericellamide,
)
from chemlog.cli import resolve_chebi_classes
from prediction_models.base import PredictionModel
from rdkit import Chem


class ChemLog(PredictionModel):

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[
            str
        ] = "A rule-based model for predicting peptides and peptide-like molecules.",
    ):
        super().__init__(name, description)

    def get_chemlog_results(self, smiles_list) -> list:
        all_preds = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None or not smiles:
                all_preds.append(None)
                continue
            mol.UpdatePropertyCache()
            charge_category = get_charge_category(mol)
            n_amino_acid_residues, add_output = get_n_amino_acid_residues(mol)
            r = {
                "charge_category": charge_category.name,
                "n_amino_acid_residues": n_amino_acid_residues,
            }
            if n_amino_acid_residues == 5:
                r["emericellamide"] = is_emericellamide(mol)[0]
            if n_amino_acid_residues == 2:
                r["2,5-diketopiperazines"] = is_diketopiperazine(mol)[0]

            chebi_classes = [f"CHEBI:{c}" for c in resolve_chebi_classes(r)]

            all_preds.append(chebi_classes)
        return all_preds

    def get_chemlog_result_info(self, smiles):
        """Get classification for single molecule with additional information."""
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None or not smiles:
            return {"error": "Failed to parse SMILES"}
        mol.UpdatePropertyCache()
        try:
            Chem.Kekulize(mol)
        except Chem.KekulizeException as e:
            pass

        charge_category = get_charge_category(mol)
        n_amino_acid_residues, add_output = get_n_amino_acid_residues(mol)
        if n_amino_acid_residues > 1:
            proteinogenics, proteinogenics_locations, _ = get_proteinogenic_amino_acids(
                mol, add_output["amino_residue"], add_output["carboxy_residue"]
            )
        else:
            proteinogenics, proteinogenics_locations, _ = [], [], []
        results = {
            "charge_category": charge_category.name,
            "n_amino_acid_residues": n_amino_acid_residues,
            "proteinogenics": proteinogenics,
            "proteinogenics_locations": proteinogenics_locations,
        }

        if n_amino_acid_residues == 5:
            emericellamide = is_emericellamide(mol)
            results["emericellamide"] = emericellamide[0]
            if emericellamide[0]:
                results["emericellamide_atoms"] = emericellamide[1]
        if n_amino_acid_residues == 2:
            diketopiperazine = is_diketopiperazine(mol)
            results["2,5-diketopiperazines"] = diketopiperazine[0]
            if diketopiperazine[0]:
                results["2,5-diketopiperazines_atoms"] = diketopiperazine[1]

        return {**results, **add_output}

    def predict(self, smiles_list):
        return [
            get_transitive_predictions([positive_i])
            for positive_i in self.get_chemlog_results(smiles_list)
        ]
