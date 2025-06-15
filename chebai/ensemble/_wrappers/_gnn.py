import chebai_graph.preprocessing.properties as p
import torch
from chebai_graph.preprocessing.property_encoder import IndexEncoder, OneHotEncoder
from torch_geometric.data.data import Data as GeomData

from ._neural_network import NNWrapper


class GNNWrapper(NNWrapper):
    def _read_smiles(self, smiles):
        d = self._data_cls_instance.reader.to_data(dict(features=smiles, labels=None))
        geom_data = d["features"]
        assert isinstance(geom_data, GeomData), "Must be an instance of GeoData"
        edge_attr = geom_data.edge_attr
        x = geom_data.x
        molecule_attr = torch.empty((1, 0))
        for property in self._data_cls_instance.properties:
            property_values = self._data_cls_instance.reader.read_property(
                smiles, property
            )
            encoded_values = []
            for value in property_values:
                # cant use standard encode for index encoder because model has been trained on a certain range of values
                # use default value if we meet an unseen value
                if isinstance(property.encoder, IndexEncoder):
                    if str(value) in property.encoder.cache:
                        index = (
                            property.encoder.cache.index(str(value))
                            + property.encoder.offset
                        )
                    else:
                        index = 0
                        print(
                            f"Unknown property value {value} for property {property} at smiles {smiles}"
                        )
                    if isinstance(property.encoder, OneHotEncoder):
                        encoded_values.append(
                            torch.nn.functional.one_hot(  # pylint: disable=E1102
                                torch.tensor(index),
                                num_classes=property.encoder.get_encoding_length(),
                            )
                        )
                    else:
                        encoded_values.append(torch.tensor([index]))

                else:
                    encoded_values.append(property.encoder.encode(value))
            if len(encoded_values) > 0:
                encoded_values = torch.stack(encoded_values)

            if isinstance(encoded_values, torch.Tensor):
                if len(encoded_values.size()) == 0:
                    encoded_values = encoded_values.unsqueeze(0)
                if len(encoded_values.size()) == 1:
                    encoded_values = encoded_values.unsqueeze(1)
            else:
                encoded_values = torch.zeros(
                    (0, property.encoder.get_encoding_length())
                )
            if isinstance(property, p.AtomProperty):
                x = torch.cat([x, encoded_values], dim=1)
            elif isinstance(property, p.BondProperty):
                edge_attr = torch.cat([edge_attr, encoded_values], dim=1)
            else:
                molecule_attr = torch.cat([molecule_attr, encoded_values[0]], dim=1)

        d["features"] = GeomData(
            x=x,
            edge_index=geom_data.edge_index,
            edge_attr=edge_attr,
            molecule_attr=molecule_attr,
        )
        return d

    def _evaluate_from_data_file(self, **kwargs) -> list:
        model_logits = super()._evaluate_from_data_file(**kwargs)
        # Currently gnn in forward method, logits are returned instead of dict containing logits
        return {"logits": model_logits}
