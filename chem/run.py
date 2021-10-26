import sys

from chem.models import chemyk, electra, graph, recursive
from chem.preprocessing import datasets as ds


def main(batch_size):
    exps = [
        (
            electra.ElectraPre,
            dict(
                lr=1e-4,
                config=dict(
                    vocab_size=1400,
                    max_position_embeddings=1800,
                    num_attention_heads=8,
                    num_hidden_layers=6,
                    type_vocab_size=1,
                ),
                epochs=2,
            ),
            (ds.PubchemUnlabelled,),
            dict(k=100),
        ),
        (
            electra.Electra,
            dict(
                lr=1e-4,
                pretrained_checkpoint="<put-your-path-here>",
                config=dict(
                    vocab_size=1400,
                    max_position_embeddings=1800,
                    num_attention_heads=8,
                    num_hidden_layers=6,
                    type_vocab_size=1,
                ),
                epochs=20,
            ),
            (ds.JCITokenData,),
            dict(),
        ),
    ]
    for net_cls, model_kwargs, datasets, data_kwargs in exps:
        for dataset in datasets:
            for weighted in [False]:
                net_cls.run(
                    dataset(batch_size, **data_kwargs),
                    net_cls.NAME,
                    model_kwargs=model_kwargs,
                    weighted=weighted,
                )


if __name__ == "__main__":
    main(int(sys.argv[1]))
