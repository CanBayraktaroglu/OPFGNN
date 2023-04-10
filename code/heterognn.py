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
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, act_fn: str, norm: torch_geometric.nn.norm.HeteroLayerNorm):
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

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = activation_resolver(act_fn, **({}))
        self.act_first = False
        self.jk_mode = "last"
        self.norm = norm if isinstance(norm, str) else None
        self.edge_types = [['SB', '-', 'PV'], ['SB', '-', 'PQ'], ['SB', '-', 'NB'], ['PV', '-', 'PQ'], ['PV', '-', 'NB']
                           , ['PV', '-', 'PV'], ['PQ', '-', 'NB'], ['PQ', '-', 'PQ'], ['NB', '-', 'NB']]

        self.convs = ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv({
                ('SB', '-', 'PV'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('SB', '-', 'PQ'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('SB', '-', 'NB'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PV', '-', 'PQ'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PV', '-', 'NB'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PV', '-', 'PV'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PQ', '-', 'NB'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('PQ', '-', 'PQ'): TransformerConv(-1, hidden_channels, edge_dim=2),
                ('NB', '-', 'NB'): TransformerConv(-1, hidden_channels, edge_dim=2),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict):
        edge_idx_dict = dict()
        edge_attr_dict = dict()

        for key in self.edge_types:
            edge_idx_dict[key] = x_dict[key]["edge_index"]
            edge_attr_dict[key] = x_dict[key]["edge_attr"]

        for conv in self.convs:
            x_dict = conv(x_dict, edge_idx_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['author'])
