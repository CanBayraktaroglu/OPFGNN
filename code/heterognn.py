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

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = activation_resolver(act_fn, **({}))
        self.act_first = False
        self.jk_mode = "last"
        self.norm = norm if isinstance(norm, str) else None

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

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_idx_dict, edge_attr_dict):
        # All calls of the forward function must support edge_attr & edge_idx for this model
        for conv in self.convs.children():
            x_dict = conv(x_dict, edge_idx_dict, edge_attr_dict)

        return x_dict

