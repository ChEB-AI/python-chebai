import fastobo
import pyhornedowl
import tqdm
from lnn import Implies, Model, Not, Predicate, Variable, World
from owlready2 import get_ontology


def get_name(iri: str):
    return iri.split("/")[-1]


if __name__ == "__main__":
    formulae = []

    # Load disjointness axioms
    # onto_dis = pyhornedowl.open_ontology("/data/ontologies/chebi-disjoints.owl")
    # print("Process disjointness releation")
    # formulae += [Implies(predicates[get_name(c)](x), Not(predicates[get_name(d)](x))) for _, c,d in (ax for ax in onto_dis.get_axioms() if ax[0] == "AxiomKind::SubClassOf" and isinstance(ax[-1], str))]

    model = Model()
    x = Variable("x")
    y = Variable("y")

    onto = pyhornedowl.open_ontology("/data/ontologies/chebi.owl")

    print("Process classes")
    predicates = {get_name(c): Predicate(get_name(c)) for c in onto.get_classes()}

    print("Process subsumption releation")
    formulae += [
        Implies(predicates[get_name(c)](x), predicates[get_name(d)](x))
        for _, c, d in (
            ax
            for ax in onto.get_axioms()
            if ax[0] == "AxiomKind::SubClassOf" and isinstance(ax[-1], str)
        )
    ]

    model.add_knowledge(*formulae, world=World.AXIOM)
    model.print()
