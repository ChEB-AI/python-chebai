from chem.models import graph, recursive, chemyk, electra
from chem import data
import sys

def main(batch_size):
    exps = [
        #(models.lstm.ChemLSTM, [100, 500, 500], (data.JCIExtendedData, data.JCIData)),
        #(models.graph.JCIGraphNet, [100, 100, 500], (data.JCIGraphData, data.JCIExtendedGraphData)),
        #(graph.JCIGraphAttentionNet, [100, 100, 500], (data.JCIGraphData, data.JCIExtendedGraphData)),
        (electra.ElectraPre, dict(config=dict(vocab_size=1400,
            max_position_embeddings=1800,
            num_attention_heads=8,
            num_hidden_layers=6,
            type_vocab_size=1,)), (data.PubChemToxicToken,)),
        #(models.graph_k2.JCIGraphK2Net, [100, 100, 500], (data.JCIGraphTwoData, data.JCIExtendedGraphTwoData))
    ]
    for net_cls, model_kwargs, datasets in exps:
        for dataset in datasets:
            for weighted in [False]:
                net_cls.run(dataset(batch_size), net_cls.NAME, model_kwargs=model_kwargs, weighted=weighted)

if __name__ == "__main__":
    main(int(sys.argv[1]))