from typing import Tuple

import torch
import torch_geometric.data

import helper as hp


class ACOPFInput:

    def __init__(self, data: torch_geometric.data.HeteroData, net, index_mappers: Tuple[dict]):
        x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, scalers_dict = hp.extract_unsupervised_inputs(
            data)
        self.x_dict = x_dict
        self.constraint_dict = constraint_dict
        self.edge_idx_dict = edge_idx_dict
        self.edge_attr_dict = edge_attr_dict
        self.bus_idx_neighbors_dict = bus_idx_neighbors_dict
        self.net = net
        self.scaler_dict = scalers_dict
        self.index_mappers = index_mappers


class ACOPFOutput:
    def __init__(self, out_dict: dict, scaler_dict: dict, net, index_mappers: Tuple[dict]):

        self.index_mappers = index_mappers
        self.output = self.process_output(out_dict, scaler_dict)
        self.net = net

    def process_output(self, out_dict: dict, scaler_dict: dict):
        num_samples = sum([len(out_dict[node_type]) for node_type in out_dict])
        output = hp.np.zeros([num_samples, 4])

        for node_type in out_dict:
            # Inverse Transform the Output Core Features
            scaled_core_features = out_dict[node_type].detach().numpy()
            scaler = scaler_dict[node_type]
            unscaled_core_features = hp.custom_minmax_inverse_transform(scaler, scaled_core_features)

            # Place the unscaled core features to their corresponding real indices/places
            indices = self.inverse_map_index(node_type)
            output[indices] = unscaled_core_features

        return output

    def inverse_map_index(self, node_type: str) -> list:
        indices = []
        idx_mapper = self.index_mappers[0]
        node_type_idx_mapper = self.index_mappers[1]

        for i in range(len(node_type_idx_mapper[node_type])):
            idx = idx_mapper[node_type_idx_mapper[node_type][i]]
            indices.append(idx)

        return indices
