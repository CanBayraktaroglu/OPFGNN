from typing import Tuple

import torch
import torch_geometric.data
from sklearn.preprocessing import MinMaxScaler

import helper as hp


class ACOPFInput:

    def __init__(self, data: torch_geometric.data.HeteroData, net, index_mappers: Tuple[dict]):
        x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, scaler, angle_params, res_bus = hp.extract_unsupervised_inputs(
            data, net, index_mappers)
        self.x_dict = x_dict
        self.constraint_dict = constraint_dict
        self.edge_idx_dict = edge_idx_dict
        self.edge_attr_dict = edge_attr_dict
        self.bus_idx_neighbors_dict = bus_idx_neighbors_dict
        self.net = net
        self.scaler = scaler
        self.index_mappers = index_mappers
        self.angle_params = angle_params
        self.res_bus = res_bus



class ACOPFOutput:
    def __init__(self, out_dict: dict, scaler, net, index_mappers: Tuple[dict], angle_params, res_bus):

        self.index_mappers = index_mappers
        self.net = net
        self.res_bus = res_bus
        self.scaler = scaler
        self.out_dict = out_dict
        #self.scaler_res_bus = scaler_res_bus
        self.output, self.target = self.process_output(out_dict, scaler, angle_params)

    def process_output(self, out_dict: dict, scaler: MinMaxScaler, angle_params):
        num_samples = sum([len(out_dict[node_type]) for node_type in out_dict])
        output = hp.np.zeros([num_samples, 4])
        target = hp.np.zeros([num_samples, 4])
        min_angle, max_angle = angle_params

        for node_type in out_dict:
            # Inverse Transform the Output Core Features
            scaled_targets = self.res_bus[node_type].detach().numpy()
            scaled_core_features = out_dict[node_type].detach().numpy()
            unscaled_core_features = scaler.inverse_transform(scaled_core_features)
            unscaled_targets = scaler.inverse_transform(scaled_targets)
            #unscaled_core_features = hp.custom_minmax_res_bus_transform(scaler, scaled_core_features)

            # Place the unscaled core features to their corresponding real indices/places
            indices = self.inverse_map_index(node_type)
            output[indices] = unscaled_core_features
            target[indices] = unscaled_targets

        sb_idx = self.inverse_map_index("SB")[0]
        output[:, 1] -= output[sb_idx, 1]

        init_volt_mags = self.net.bus.vn_kv.values
        output[:, 0] /= init_volt_mags
        target[:, 0] /= init_volt_mags

        return output, target

    def inverse_map_index(self, node_type: str) -> list:
        indices = []
        idx_mapper = self.index_mappers[0]
        node_type_idx_mapper = self.index_mappers[1]

        for i in range(len(node_type_idx_mapper[node_type])):
            idx = idx_mapper[node_type_idx_mapper[node_type][i]]
            indices.append(idx)

        return indices
