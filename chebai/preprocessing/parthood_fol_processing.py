import logging
import os

import networkx as nx
import pandas as pd
import tqdm
from chemlog2.fol_classification.model_checking import ModelChecker, ModelCheckerOutcome
from chemlog2.fol_classification.substruct_verifier import SubstructVerifier
from chemlog2.preprocessing.mol_to_fol import mol_to_fol_atoms
from gavel.logic import logic, logic_utils


def load_data():
    res_smiles = pd.read_pickle(os.path.join("parthood", "res_smiles.pkl"))
    groups = pd.read_pickle(os.path.join("parthood", "groups.pkl"))
    return res_smiles, groups


def build_group_hierarchy(groups, processed_target_formulas):
    # get graph of groups, missing SMILES

    groups["group_parents"] = [
        [p for p in row["parents"] if p in groups.index]
        for idx, row in groups.iterrows()
    ]
    group_hierarchy = nx.DiGraph()
    group_hierarchy.add_nodes_from(groups.index)
    for idx, row in groups.iterrows():
        for p in row["group_parents"]:
            group_hierarchy.add_edge(idx, p)
    # remove nodes from hierarchy if no FOL is available
    for n in list(group_hierarchy.nodes):
        if n not in processed_target_formulas:
            for d in nx.descendants(group_hierarchy, n):
                for a in nx.ancestors(group_hierarchy, n):
                    group_hierarchy.add_edge(a, d)
            group_hierarchy.remove_node(n)
    return group_hierarchy


def process_formulas():
    verifier = SubstructVerifier()
    processed_target_formulas = {
        int(k.split("_")[1]): logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL,
            logic_utils.get_vars_in_formula(f.right),
            f.right,
        )
        for k, f in verifier.substruct_defs.items()
    }
    print(f"Processed {len(processed_target_formulas)} target formulas")
    return verifier, processed_target_formulas


def classify(
    chebi_smiles, processed_target_formulas, verifier, group_hierarchy, groups
):
    all_matches = []
    for idx, row in tqdm.tqdm(chebi_smiles.iterrows()):
        if "*" in row["smiles"]:
            continue
        universe, extensions = mol_to_fol_atoms(row["mol"])
        model_checker = ModelChecker(
            universe,
            extensions,
            all_different=True,
            predicate_definitions={
                pred: (formula.left.arguments, formula.right)
                for pred, formula in verifier.substruct_helpers.items()
            },
        )
        matches = []
        tried_targets = []
        n_proofs_made = 0
        while len(tried_targets) < len(processed_target_formulas):
            top_targets = [
                n
                for n in group_hierarchy.nodes
                if n not in tried_targets
                and all(d in tried_targets for d in nx.descendants(group_hierarchy, n))
            ]
            print(
                f"This layer: try {len(top_targets)} target cls (status: {len(tried_targets)} / {len(processed_target_formulas)})"
            )
            for target_cls in top_targets:
                try:
                    result = model_checker.find_model(
                        processed_target_formulas[target_cls]
                    )[0]
                    n_proofs_made += 1
                    if target_cls in [23004]:
                        logging.info(f"Proof for {idx} has part {target_cls}: {result}")
                    if result in [
                        ModelCheckerOutcome.MODEL_FOUND,
                        ModelCheckerOutcome.MODEL_FOUND_INFERRED,
                    ]:
                        logging.info(
                            f"Found match: {idx} has part {target_cls} ({groups.loc[target_cls, 'name']})"
                        )
                        matches.append(target_cls)
                    else:
                        # if supercls has been disproven, mark all subclasses as tried
                        for a in nx.ancestors(group_hierarchy, target_cls):
                            tried_targets.append(a)
                except Exception as e:
                    print(f"Error while classifying {idx} has part {target_cls}: {e}")
                    # if supercls has been disproven, mark all subclasses as tried
                    for a in nx.ancestors(group_hierarchy, target_cls):
                        tried_targets.append(a)
                tried_targets.append(target_cls)
        print(f"Made {n_proofs_made} proofs for {idx}")

        all_matches.append(matches)
    chebi_smiles["matches_fol"] = all_matches
    chebi_smiles["matches_n_fol"] = chebi_smiles["matches_fol"].apply(len)
    return chebi_smiles


if __name__ == "__main__":
    chebi_smiles, groups = load_data()
    verifier, processed_target_formulas = process_formulas()
    # group_hierarchy = build_group_hierarchy(groups, processed_target_formulas)
    # chebi_smiles_test = chebi_smiles[[i in [48604, 29256] for i in chebi_smiles.index]]

    # logging.basicConfig(
    #    handlers=[
    #        logging.FileHandler("quick_logs.log", encoding="utf-8"),
    #        logging.StreamHandler(sys.stdout),
    #    ],
    #    level=logging.INFO,
    # )
    # chebi_smiles = classify(
    #    chebi_smiles_test, processed_target_formulas, verifier, group_hierarchy, groups
    # )
    # chebi_smiles.to_pickle(os.path.join("../../parthood", "chebi_smiles_fol.pkl"))
