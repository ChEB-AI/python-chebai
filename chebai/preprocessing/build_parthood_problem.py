import csv
import logging
import os
import re
import sys
import time
from enum import Enum

import networkx as nx
import pandas as pd
from chemlog2.fol_classification.substruct_verifier import SubstructVerifier
from gavel.dialects.tptp.compiler import TPTPCompiler
from gavel.logic import logic, status
from gavel.prover.vampire.interface import VampireInterface
from rdkit import Chem


class VampireOutcome(Enum):
    TIMEOUT = 0
    COUNTER = 1
    PROOF = 2
    NO_REFUTATION = 3
    ERROR = 4
    COUNTER_INFERRED = 5
    PROOF_INFERRED = 6


BACKGROUND = [
    f"fof(bond_symmetry_{bond_type}, axiom, ![X,Y]: (b{bond_type}(X,Y) <=> b{bond_type}(Y,X)))."
    for bond_type in Chem.rdchem.BondType.names
]


def build_parthood_problem(verifier, subcls, supercls):
    sub_formula = verifier.substruct_defs[f"chebi_{subcls}"]
    super_formula = verifier.substruct_defs[f"chebi_{supercls}"]
    tptp_compiler = TPTPCompiler()

    conjecture = logic.BinaryFormula(
        logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL, sub_formula.left.arguments, sub_formula.left
        ),
        logic.BinaryConnective.IMPLICATION,
        logic.QuantifiedFormula(
            logic.Quantifier.EXISTENTIAL,
            super_formula.left.arguments,
            super_formula.left,
        ),
    )

    sub_formula = logic.QuantifiedFormula(
        logic.Quantifier.UNIVERSAL, sub_formula.left.arguments, sub_formula
    )
    super_formula = logic.QuantifiedFormula(
        logic.Quantifier.UNIVERSAL, super_formula.left.arguments, super_formula
    )
    problem = BACKGROUND + [
        f"fof({name}, {'conjecture' if 'conj' in name else 'axiom'}, {tptp_compiler.visit(f)}).\n".replace(
            "'", ""
        )
        for name, f in [
            (f"chebi_{subcls}", sub_formula),
            (f"chebi_{supercls}", super_formula),
            (f"conj_chebi_{subcls}_implies_chebi_{supercls}", conjecture),
        ]
    ]
    return "\n".join(problem)


def prove(problem: str, timeout=10) -> VampireOutcome:
    vampire = VampireInterface(
        flags=["-p tptp", "--input_syntax tptp", f"-t {timeout}"]
    )
    try:
        orf_out = vampire._submit_problem(problem)
    except RuntimeError as e:
        orf_out = e
        if "Time limit reached" in str(e):
            return VampireOutcome.TIMEOUT
        elif "Refutation not found" in str(e):
            return VampireOutcome.NO_REFUTATION
        else:
            print(e)
            return VampireOutcome.ERROR
    else:
        # prf = vampire._build_proof(vampire._post_process_proof(orf_out), None)
        szs_status = re.search(
            r"SZS status (\w+)",
            vampire._post_process_proof(orf_out),
        )
        if szs_status:
            szs_status = status.get_status(szs_status.groups()[0])()
        else:
            szs_status = status.StatusUnknown()
        # print(szs_status._name)
        if isinstance(szs_status, status.StatusTheorem):
            return VampireOutcome.PROOF
        else:
            return VampireOutcome.COUNTER


def build_group_hierarchy(verifier, groups: list, results_file):
    group_hierarchy = nx.DiGraph()
    existing_proofs = {}
    with open(results_file, "r") as f:
        r = csv.reader(f)
        for row in r:
            if row[0] == "subcls":
                continue
            existing_proofs[(int(row[0]), int(row[1]))] = row[2]
    logging.info(f"Loaded {len(existing_proofs)} existing proofs")

    for (subcls, supercls), outcome in existing_proofs.items():
        if outcome == "PROOF":
            group_hierarchy.add_edge(subcls, supercls)

    # iteration pattern: given that all proofs for pairwise combinations of n classes have been made, add a new class
    # and prove all combinations with the new class
    for idx_a, cls_a in enumerate(groups):
        for idx_b in range(idx_a):
            for subcls, supercls in [(cls_a, groups[idx_b]), (groups[idx_b], cls_a)]:
                if (subcls, supercls) in existing_proofs:
                    continue
                logging.info(f"Proving {subcls} -> {supercls}")
                start_time = time.perf_counter()
                problem = build_parthood_problem(verifier, subcls, supercls)
                outcome = prove(problem)
                existing_proofs[(subcls, supercls)] = outcome.name
                with open(results_file, "a") as f:
                    f.write(
                        f"{subcls},{supercls},{outcome.name},{time.perf_counter() - start_time}\n"
                    )
                if outcome == VampireOutcome.PROOF:
                    group_hierarchy.add_edge(subcls, supercls)
                    for ancestor in nx.ancestors(group_hierarchy, subcls):
                        existing_proofs[(ancestor, supercls)] = "PROOF_INFERRED"
                        with open(results_file, "a") as f:
                            f.write(f"{subcls},{supercls},PROOF_INFERRED,0\n")
                    for descendant in nx.descendants(group_hierarchy, supercls):
                        existing_proofs[(subcls, descendant)] = "PROOF_INFERRED"
                        with open(results_file, "a") as f:
                            f.write(f"{subcls},{supercls},PROOF_INFERRED,0\n")

    return group_hierarchy


def get_groups():
    # load groups and sort by size
    groups = pd.read_pickle(os.path.join("parthood", "groups.pkl"))
    groups["n_direct_members"] = [
        sum(idx in p for p in groups["group_parents"]) for idx in groups.index
    ]
    return (
        groups[[f"chebi_{idx}" in verifier.substruct_defs for idx in groups.index]]
        .sort_values("n_members", ascending=False)
        .index
    )


def get_group_is_a_relations():
    # assume that if A is_a B, then likely for all M: `M has_part some A` is_a `M has_part some B`
    groups = pd.read_pickle(os.path.join("parthood", "groups.pkl"))
    pairs = []
    for idx, row in groups.iterrows():
        for p in row["group_parents"]:
            pairs.append((idx, p))
    return pairs


if __name__ == "__main__":
    logging.basicConfig(
        handlers=[
            logging.FileHandler("quick_logs.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        level=logging.INFO,
    )
    logging.info("Initialising formulas")
    verifier = SubstructVerifier()
    with open(os.path.join("parthood", "fol_groups_by_size.txt"), "r") as f:
        groups = [int(r) for r in f.read().splitlines()]
    print(f"Loaded {len(groups)} groups: {groups[:10]}")
    logging.info("Building hierarchy")
    group_hierarchy = build_group_hierarchy(
        verifier, groups, os.path.join("parthood", "group_hierarchy_proofs.csv")
    )
    print(group_hierarchy)
    print(group_hierarchy.edges)
