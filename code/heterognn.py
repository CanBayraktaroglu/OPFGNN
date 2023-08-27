import torch_geometric.nn.norm.layer_norm
from gnn import *
import warnings
from collections import defaultdict
from typing import Dict, Optional
from torch import Tensor
from torch_geometric.nn.conv.hgt_conv import group
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import Adj, EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops

# Define the GNN layer
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, act_fn: str
                 , norm: Union[str, Callable, None] = None):
        """
        Edge Types:
            SB - PV, SB - PQ, SB - NB
            PV - PQ, PV - NB, PV - PV
            PQ - NB; PQ - PQ
            NB - NB
        Args:
            hidden_channels:
            out_channels:
            num_layers:
        """
        super().__init__()

        self.in_channels = 11
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = activation_resolver(act_fn, **({}))
        self.act_first = False
        self.jk_mode = "last"
        in_channels = hidden_channels
        self.lin = Linear(in_channels, self.out_channels, bias=False)
        self.norm = norm if isinstance(norm, str) else None
        self.bias = nn.Parameter(torch.tensor(self.out_channels, dtype=torch.float32))

        self.edge_types = [
            ('PV', "isConnected", 'SB'),
            ('SB', "isConnected", 'PQ'),
            ('SB', "isConnected", 'NB'),
            ('PV', "isConnected", 'PQ'),
            ('NB', "isConnected", 'PQ'),
            ('PQ', "isConnected", 'NB'),

            ('SB', "isConnected", 'PV'),
            ('PQ', "isConnected", 'SB'),
            ('NB', "isConnected", 'SB'),
            ('PQ', "isConnected", 'PV'),
            ('PQ', "isConnected", 'NB'),
            ('NB', "isConnected", 'PQ'),

            ('PV', "isConnected", 'PV'),
            ('PQ', "isConnected", 'PQ'),
            ('NB', "isConnected", 'NB'),

        ]

        self.convs = ModuleList()

        for _ in range(self.num_layers):
            conv = HeteroConv({
                ('SB', "isConnected", 'PV'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('SB', "isConnected", 'PQ'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('SB', "isConnected", 'NB'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PV', "isConnected", 'PQ'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PV', "isConnected", 'NB'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PQ', "isConnected", 'NB'): TransformerConv(-1, hidden_channels, edge_dim=2),

                ('PV', "isConnected", 'SB'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PQ', "isConnected", 'SB'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('NB', "isConnected", 'SB'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PQ', "isConnected", 'PV'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PQ', "isConnected", 'NB'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('NB', "isConnected", 'PQ'): TransformerConv(-1, hidden_channels, edge_dim=2),

                ('PV', "isConnected", 'PV'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PQ', "isConnected", 'PQ'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('NB', "isConnected", 'NB'): TransformerConv(-1, hidden_channels, edge_dim=2),
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict):

        # All calls of the forward function must support edge_attr & edge_idx for this model
        for conv in self.convs.children():
            x_dict = conv(x_dict, edge_idx_dict, edge_attr_dict)

            # Apply activation function for the output features of each type and update the dict
            for node_type in x_dict:
                x_dict[node_type] = self.act(x_dict[node_type])

        for node_type in x_dict:
            x_dict[node_type] = self.lin(x_dict[node_type])

        # ACOPF Forward Pass for P and Q
        for from_bus in bus_idx_neighbors_dict:
            for bus_idx in bus_idx_neighbors_dict[from_bus]:
                V_i = abs(x_dict[from_bus][bus_idx][0])
                volt_angle_i = x_dict[from_bus][bus_idx][1]
                P_i = torch.tensor(0.0, dtype=torch.float32)
                Q_i = torch.tensor(0.0, dtype=torch.float32)

                for pair in bus_idx_neighbors_dict[from_bus][bus_idx]:
                    # For each neighbor of the iterated bus
                    to_bus, to_bus_idx, edge_attr = pair

                    V_j = abs(x_dict[to_bus][to_bus_idx][0])

                    volt_angle_j = x_dict[to_bus][to_bus_idx][1]
                    delta_ij = volt_angle_j - volt_angle_i

                    G_ij = edge_attr[0] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)
                    B_ij = -edge_attr[1] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)

                    # ACOPF Equation for P_i
                    P_ij = V_i * V_j * (G_ij * torch.cos(delta_ij) + B_ij * torch.sin(delta_ij))

                    P_i += P_ij

                    # ACOPF Equation for Q_i
                    Q_ij = V_i * V_j * (G_ij * torch.sin(delta_ij) - B_ij * torch.cos(delta_ij))

                    Q_i += Q_ij

                x_dict[from_bus][bus_idx][2] = P_i
                x_dict[from_bus][bus_idx][3] = Q_i

        return x_dict
