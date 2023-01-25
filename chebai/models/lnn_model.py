from owlready2 import get_ontology
from lnn import Model, Predicate, Variable, World, Implies, Not, Or
from chebai.models.base import JCIBaseNet
import pyhornedowl
import itertools
import tqdm
import networkx as nx
def get_name(iri: str):
    return iri.split("/")[-1]

def _collect_subclasses(onto, iri):
    for sub in onto.get_subclasses(iri):
        yield sub
        for ssub in _collect_subclasses(onto, sub):
            yield ssub
class LNN(JCIBaseNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lnn = Model()
        x = Variable("x")
        y = Variable("y")


        # Load disjointness axioms
        onto_dis = pyhornedowl.open_ontology("/data/ontologies/chebi-disjoints.owl")


        onto = pyhornedowl.open_ontology("/data/ontologies/chebi.owl")
        print("Process classes")
        subsumptions = [
            (get_name(sub), get_name(sup)) for _, sub, sup in (
                ax
                for ax in onto.get_axioms()
                if ax[0] == "AxiomKind::SubClassOf" and isinstance(ax[-1], str)
            )]
        graph = nx.DiGraph()
        graph.add_edges_from(subsumptions)
        graph = nx.transitive_closure_dag(graph)

        classes = [f"CHEBI_{f}" for f in kwargs["class_labels"]]
        graph = graph.subgraph(classes)
        big_graph = graph


        self.predicates = predicates = {c: Predicate(c) for c in classes}


        disjoint_pairs = {(a, b)  for _, c, d in (
                    ax for ax in onto_dis.get_axioms()
                if ax[0] == "AxiomKind::DisjointClasses" and isinstance(ax[-1], str)) if c in big_graph.nodes and d in big_graph.nodes
                    for a in big_graph.predecesspr(c) for b in big_graph.predecesspr(d) if a in classes and b in classes}

        print("Process disjointness releation")
        formulae = [Or(Not(predicates[get_name(c)](x)), Not(predicates[get_name(d)](x))) for c,d in disjoint_pairs]


        graph = nx.transitive_reduction(graph)

        print("Process subsumption releation")
        formulae += [
                Implies(predicates[sub](x), predicates[sup](x))
                for (sub, sup) in graph.edges
            ]


        self.classes = [self.predicates[f"CHEBI_{h}"] for h in kwargs["class_labels"]]

        self.lnn.add_knowledge(*tqdm.tqdm(formulae), world=World.AXIOM)

    def _certainty_to_boundaries(self, certainty):
        if certainty < 0.5:
            return 0.0, 2*certainty
        else:
            return 2*certainty-1, 1.0
    def forward(self, data):
        batch = data["features"]
        self.lnn.add_data({
            c: {
                str(i): self._certainty_to_boundaries(certainty)
                    for i, certainty in enumerate(batch[:,cid])
            } for cid, c in enumerate(self.classes)
        })
        t = self.lnn.forward()
        return t

