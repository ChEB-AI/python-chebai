from chem.models import graph, graph_k2, lstm
from chem.data import JCIGraphData, JCIExtendedGraphData, JCIData, JCIExtendedData


def main():
    exps = [
        (graph.JCIGraphNet,[100, 100, 500], (JCIGraphData, JCIExtendedGraphData)),
        (lstm.ChemLSTM,  [100, 500, 500], (JCIGraphData, JCIExtendedGraphData)),
        (graph_k2.JCIGraphK2Net, [100, 100, 500], (JCIGraphData, JCIExtendedGraphData))
    ]
    for net_cls, model_args, datasets in exps:
        for dataset in datasets:
            for weighted in [True, False]:
                net_cls.run(dataset, model_args=model_args, weighted=weighted)

if __name__ == "__main__":
    main()