from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import GraphConv, GCNConv, TransformerConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import Linear, ModuleList
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (activation_resolver, normalization_resolver)
from torch_sparse import SparseTensor
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
import copy

"""
class GCNConv(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def forward(self, x, edge_weight, edge_index):

        # Step 1: Multiply edge weights by source node features
        edge_weight = F.relu(self.lin(edge_weight))

        # Step 2: Compute normalization.
        edge_weight = edge_weight.view(-1)

        # Step 3: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=edge_weight)

        # Step 4: Apply a final bias vector.
        out += self.bias

        return out

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def message(self, x_j, norm):
        # Step 4: Normalize messages by edge weights
        return norm.view(-1, 1) * x_j


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, edge_weight):
        super(GCNNet, self).__init__()
        self.conv1 = GraphConv(in_channels, out_channels)
        self.conv2 = GraphConv(out_channels, in_channels)
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def forward(self, x):
        # Step 1: Pass the input through the first GCN layer.
        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)

        # Step 2: Pass the output through the second GCN layer.
        x = self.conv2(x, self.edge_index, self.edge_weight)

        return x
"""


class GNN(BasicGNN):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
        """


    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: Optional[int] = None,
                 dropout: float = 0.0, norm: Union[str, Callable, None] = None, jk: Optional[str] = None, layer_type="GCN", **kwargs):
        super(BasicGNN, self).__init__()
        self.in_channels = in_channels
        self.layer_type=layer_type
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = activation_resolver("tanh", **({}))
        self.act_first = False
        self.jk_mode = jk
        self.norm = norm if isinstance(norm, str) else None

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels

        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels

        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                hidden_channels,
                **({}),
            )
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm_layer))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels, bias=False, dtype=torch.float32)

        self.bias = nn.Parameter(torch.tensor(self.out_channels, dtype=torch.float32))

        if layer_type == "GCN" or layer_type == "GraphConv":
            self.supports_edge_weight = True
            self.supports_edge_attr = False
        elif layer_type == "TransConv":
            self.supports_edge_weight = False
            self.supports_edge_attr = True
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()

        for norm in self.norms or []:
            norm.reset_parameters()

        if hasattr(self, 'jk'):
            self.jk.reset_parameters()

        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

        if hasattr(self, "bias"):
            self.bias.data.zero_()

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        if self.layer_type == "TransConv":
            result = TransformerConv(in_channels, out_channels, **kwargs)#GCNConv(in_channels, out_channels, **kwargs)
        elif self.layer_type == "GCN":
            result = GCNConv(in_channels, out_channels, **kwargs)
        else:
            result = GraphConv(in_channels, out_channels, **kwargs)
        return result

    def forward(self, x, edge_index, edge_weight: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None):
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
        """

        xs: List[torch.Tensor] = []
        for i in range(self.num_layers):
            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_weight)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x

    @torch.no_grad()
    def inference(self, loader: NeighborLoader,
                  device: Optional[torch.device] = None,
                  progress_bar: bool = False) -> torch.Tensor:
        r"""Performs layer-wise inference on large-graphs using a
        :class:`~torch_geometric.loader.NeighborLoader`, where
        :class:`~torch_geometric.loader.NeighborLoader` should sample the
        full neighborhood for only one layer.
        This is an efficient way to compute the output embeddings for all
        nodes in the graph.
        Only applicable in case :obj:`jk=None` or `jk='last'`.
        """
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training
        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description('Inference')

        x_all = loader.data.x.cpu()
        loader.data.n_id = torch.arange(x_all.size(0), dtype=torch.float32)

        for i in range(self.num_layers):
            xs: List[torch.Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                if hasattr(batch, 'adj_t'):
                    edge_index = batch.adj_t.to(device)
                else:
                    edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index)[:batch.batch_size]
                if i == self.num_layers - 1 and self.jk_mode is None:
                    xs.append(x.cpu())
                    if progress_bar:
                        pbar.update(1)
                    continue
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.norms is not None:
                    x = self.norms[i](x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                if i == self.num_layers - 1 and hasattr(self, 'lin'):
                    x = self.lin(x)
                xs.append(x.cpu())
                if progress_bar:
                    pbar.update(1)
            x_all = torch.cat(xs, dim=0)
        if progress_bar:
            pbar.close()
        del loader.data.n_id

        return x_all

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
