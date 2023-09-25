import torch
import torch_geometric.data

import helper as hp
class ACOPFData:
    def __init__(self, data: torch_geometric.data.HeteroData, network_load_P, network_load_Q):
        x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, scalers_dict = hp.extract_unsupervised_inputs(data)
        self.x_dict = x_dict
        self.constraint_dict = constraint_dict
        self.edge_idx_dict = edge_idx_dict
        self.edge_attr_dict = edge_attr_dict
        self.bus_idx_neighbors_dict = bus_idx_neighbors_dict
        self.network_loads = torch.tensor([network_load_P, network_load_Q], dtype=torch.float32, requires_grad=False)
        self.scalers_dict = scalers_dict