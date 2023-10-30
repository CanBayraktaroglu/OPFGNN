from collections import defaultdict

import torch
import torch_geometric.nn.aggr
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch_geometric.nn import SplineConv, CGConv, GENConv, ResGatedGraphConv, GATConv, GATv2Conv, GeneralConv
from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.utils.hetero import check_add_self_loops
from torch_geometric.nn.norm import BatchNorm
from gnn import *

import helper as hp


# Define the GNN layer


def coth(x):
    # To ensure numerical stability, we'll avoid calculating coth for values close to zero.
    # We'll set a threshold below which the value will be clamped.
    epsilon = 1e-7
    x = torch.clamp(x, min=epsilon)
    return 1.0 / torch.tanh(x)


def custom_sigmoid(x: Tensor, lower_bound, upper_bound, slope=torch.tensor(0.1)):
    # Scale and shift the sigmoid function
    scaled_sigmoid = 1 / (1 + torch.exp(-slope * (x - (lower_bound + upper_bound) / 2)))

    # Scale and shift the result to fit within the specified lower_bound and upper_bound
    scaled_sigmoid = (scaled_sigmoid - 0.5) * (upper_bound - lower_bound) + (lower_bound + upper_bound) / 2

    return scaled_sigmoid


def custom_tanh(x: torch.Tensor, lower_bound, upper_bound) -> torch.Tensor:
    width = upper_bound - lower_bound
    return 0.5 * width * torch.tanh(x) + 0.5 * (upper_bound + lower_bound)


def soft_abs(x):
    return torch.log(1 + torch.exp(x))


def group(xs: List[Tensor], hidden_channels, aggr: Optional[str]) -> Optional[Tensor]:

    out = torch.cat(xs, dim=0)
    half = int(hidden_channels)

    P_ij = out[:, :half]
    Q_ij = out[:, half:]


    P_i = getattr(torch, aggr)(P_ij, dim=1)
    Q_i = getattr(torch, aggr)(Q_ij, dim=1)

    if len(xs) > 1:
        out = torch.cat([P_i.view(-1,1), Q_i.view(-1,1)], dim=1)

    else:
        out = torch.stack([P_i, Q_i])
    return out


def group_mean(xs: List[Tensor]) -> Optional[Tensor]:
    out = getattr(torch, "mean")(xs, dim=0)
    return out

def flatten_and_reshape_tensors(tensor_list):
    flat_list = []

    for tensor in tensor_list:
        # Check if the current element is a tensor
        if isinstance(tensor, torch.Tensor):
            # Check for scalar tensor (0-dimensional tensor)
            if tensor.dim() == 0:
                flat_list.append(tensor)  # append scalar tensor directly to the list
                continue

            # If it's a nested tensor, extract inner tensors
            if tensor.dim() > 0 and isinstance(tensor[0], torch.Tensor):
                flat_list.extend(flatten_and_reshape_tensors(tensor))
            else:
                last_dim = tensor.shape[-1]
                reshaped_tensor = tensor.view(-1, last_dim)  # Reshape the tensor
                flat_list.append(reshaped_tensor)
        # If the current element is a list, recurse into it
        elif isinstance(tensor, list):
            flat_list.extend(flatten_and_reshape_tensors(tensor))

    return flat_list


class ACOPFEmbedder_Bus(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, act_fn: str = "elu"
                 , norm: Union[str, Callable, None] = None, init_data=None):
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

        self.norms = None
        self.meta_data = None
        self.aggr = "sum"
        #self.aggr_P = torch_geometric.nn.aggr.PowerMeanAggregation(learn=True)
        #self.aggr_Q = torch_geometric.nn.aggr.PowerMeanAggregation(learn=True)
        self.heads = 1
        self.in_channels = 2
        self.hidden_channels = hidden_channels
        self.out_channels = 2
        self.num_layers = num_layers
        self.dropout = dropout
        self.act_fn = act_fn
        self.act = activation_resolver(act_fn, **({}))
        self.act_first = False
        self.jk_mode = "last"
        in_channels = hidden_channels * self.heads
        self.node_types = ["SB", "PQ", "PV", "NB"]


        self.convs = ModuleList()
        # self.splines = ModuleList()
        #GATv2Conv(-1, self.hidden_channels, edge_dim=2)
        for _ in range(self.num_layers):
            conv = ModuleDict({
                'SB': TransformerConv(-1, self.hidden_channels, edge_dim=2,heads=self.heads),

                'PQ': TransformerConv(-1, self.hidden_channels, edge_dim=2,heads=self.heads),

                'PV': TransformerConv(-1, self.hidden_channels, edge_dim=2,heads=self.heads),
                'NB': TransformerConv(-1, self.hidden_channels, edge_dim=2,heads=self.heads),
            })

            self.convs.append(conv)

        self.node_types_dict = {key: init_data[key].x.size(0) for key in init_data.node_types}

        fc = ModuleDict()
        for node_type in self.node_types_dict:
            size = self.node_types_dict[node_type]
            fc[node_type] = Linear(-1, int(size * 2), bias=True)

        self.fcs = fc

    def lazy_init(self, lazy_input, net, index_mappers):
        x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, scaler, _, res_bus_dict = hp.extract_unsupervised_inputs(
            lazy_input, net, index_mappers)

        #initial_voltage_dict = dict()
        #for node_type in x_dict:
            #V = x_dict[node_type][:, 0].view(-1, 1)
            #initial_voltage_dict[node_type] = V
        #self.init_layer_norm(x_dict)

        _x_dict = dict()
        for node_type in x_dict:
            V = x_dict[node_type][:, 0].view(-1,1)
            delta = x_dict[node_type][:, 1].view(-1,1)
            powers = torch.cat([V,delta], dim=1)
            _x_dict[node_type] = powers

        output = self.forward(_x_dict, bus_idx_neighbors_dict, edge_idx_dict, edge_attr_dict, check_nans=True)
        print(output)

    def set_meta_data(self, x_dict):
        meta_data = dict()

        for node_type in x_dict:
            meta_data[node_type] = len(x_dict[node_type])

        self.meta_data = meta_data

    def init_layer_norm(self, x_dict):
        self.set_meta_data(x_dict)
        self.norms = ModuleList()
        norm_layer = HeteroMinMaxNorm()

        # for _ in range(self.num_layers):
        # self.norms.append(copy.deepcopy(norm_layer))
        self.norms.append(norm_layer)

    def group(self, xs: List[Tensor]) -> Optional[Tensor]:

        out = torch.cat(xs, dim=0)
        half = int(self.hidden_channels/2)

        P_ij = out[:, :half]
        Q_ij = out[:, half:]

        if self.aggr is not None:
            P_i = getattr(torch, self.aggr)(P_ij, dim=1)
            Q_i = getattr(torch, self.aggr)(Q_ij, dim=1)

        else:
            P_i = self.aggr_P(P_ij, dim=1)
            print(f"P_i : {P_i}")
            Q_i = self.aggr_Q(Q_ij, dim=1)
            print(f"Q_i : {Q_i}")

        if len(xs) > 1:
            out = torch.cat([P_i.view(-1, 1), Q_i.view(-1, 1)], dim=1)

        else:
            out = torch.stack([P_i, Q_i])
        return out

    def group_wired(self, xs: List[Tensor]) -> Optional[Tensor]:

        out = torch.cat(xs, dim=0)
        half = int(self.hidden_channels/4)

        P_ij = out[:, :half]
        Q_ij = out[:, half:]

        if self.aggr is not None:
            P_i = getattr(torch, self.aggr)(P_ij, dim=1)
            Q_i = getattr(torch, self.aggr)(Q_ij, dim=1)

        else:
            P_i = self.aggr_P(P_ij, dim=1)
            print(f"P_i : {P_i}")
            Q_i = self.aggr_Q(Q_ij, dim=1)
            print(f"Q_i : {Q_i}")

        if len(xs) > 1:
            out = torch.cat([P_i.view(-1, 1), Q_i.view(-1, 1)], dim=1)

        else:
            out = torch.stack([P_i, Q_i])
        return out

    def forward(self, x_dict, bus_idx_neighbors_dict, edge_idx_dict, edge_attr_dict,
                check_nans: bool = False):


        if check_nans:
            for node_type in x_dict:
                x = x_dict[node_type]
                if torch.isnan(x).any():
                    print(f"nan found in node type {node_type} of x_dict")

        # All calls of the forward function must support edge_attr & edge_idx for this model

        out_dict = defaultdict(list)

        for i in range(self.num_layers):

            for from_bus in bus_idx_neighbors_dict:
                conv = self.convs[i][from_bus]
                to_xs = []
                edge_attr = []
                edge_idx = [[], []]
                #print(f"from_bus : {from_bus}")

                for bus_idx in bus_idx_neighbors_dict[from_bus]:

                    for pair in bus_idx_neighbors_dict[from_bus][bus_idx]:
                        # For each neighbor of the iterated bus
                        to_bus, to_bus_idx, edge_attribute = pair
                        to_x = x_dict[to_bus][to_bus_idx]
                        to_xs.append(to_x.view(1, self.in_channels))
                        #print(f"bus_idx: {bus_idx}, to_idx: {len(edge_idx[0])}")
                        edge_idx[1].append(bus_idx)
                        edge_idx[0].append(len(edge_idx[0]))
                        edge_attr.append(edge_attribute.view(1,2))

                #print(f"edge_idx: {edge_idx}")
                edge_attrs = torch.cat(edge_attr,dim=0)
                #print(f"edge_attr: {edge_attr.shape}")
                x_dst = torch.cat(to_xs, dim=0)
                #print(f"x_dst shape: {x_dst.shape}")
                #print(f"x_src shape: {x_dict[from_bus].shape}")

                out = conv((x_dst, x_dict[from_bus]), edge_index=torch.tensor(edge_idx), edge_attr=edge_attrs)

                out_dict[from_bus].append(out)
                #print(f"shape: {out.shape}")


        # Calculate the Aggregations
        for key, value in out_dict.items():
            #print(f"key: {key}")
            #shape = [x.shape for x in value]
            #print(f"shape before Aggr {shape}")

            val = self.group(value)

            #print(f"shape after Aggr {val.shape}")
            fc = self.fcs[key]

            # Flatten the tensor
            _out_ = val.view(-1)
            #print(f"shape after flattened {_out_.shape}")

            out = fc(_out_)

            out_dict[key] = out.view(self.node_types_dict[key], self.out_channels)

            #print(f"{key}: {len(out_dict[key])}")

        return out_dict


class ACOPFGeneral(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, act_fn: str = "elu"
                 , norm: Union[str, Callable, None] = None, init_data=None):
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

        self.norms = None
        self.meta_data = None
        self.aggr = "sum"
        #self.aggr_P = torch_geometric.nn.aggr.PowerMeanAggregation(learn=True)
        #self.aggr_Q = torch_geometric.nn.aggr.PowerMeanAggregation(learn=True)
        self.heads = 1
        self.in_channels = 4
        self.hidden_channels = hidden_channels
        self.out_channels = 4
        self.num_layers = num_layers
        self.dropout = dropout
        self.act_fn = act_fn
        self.act = activation_resolver(act_fn, **({}))
        self.act_first = False
        self.jk_mode = "last"
        in_channels = hidden_channels * self.heads
        self.node_types = ["SB", "PQ", "PV", "NB"]


        self.convs = ModuleList()
        # self.splines = ModuleList()
        #GATv2Conv(-1, self.hidden_channels, edge_dim=2)
        for _ in range(self.num_layers):
            conv = ModuleDict({
                'SB': GeneralConv(-1,self.hidden_channels,2,attention=True,l2_normalize=True),#TransformerConv(-1, self.hidden_channels, edge_dim=2,heads=self.heads),

                'PQ': GeneralConv(-1,self.hidden_channels,2,attention=True,l2_normalize=True),#TransformerConv(-1, self.hidden_channels, edge_dim=2,heads=self.heads),

                'PV': GeneralConv(-1,self.hidden_channels,2,attention=True,l2_normalize=True),#TransformerConv(-1, self.hidden_channels, edge_dim=2,heads=self.heads),
                'NB': GeneralConv(-1,self.hidden_channels,2,attention=True,l2_normalize=True),#TransformerConv(-1, self.hidden_channels, edge_dim=2,heads=self.heads)
            })

            self.convs.append(conv)

        self.node_types_dict = {key: init_data[key].x.size(0) for key in init_data.node_types}

        fc = ModuleDict()
        for node_type in self.node_types_dict:
            size = self.node_types_dict[node_type]
            fc[node_type] = Linear(-1, int(size * self.out_channels), bias=True)

        self.fcs = fc

    def lazy_init(self, lazy_input, net, index_mappers):
        x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, scaler, _, res_bus_dict = hp.extract_unsupervised_inputs(
            lazy_input, net, index_mappers)

        #self.init_layer_norm(x_dict)
        output = self.forward(x_dict, bus_idx_neighbors_dict, edge_idx_dict, edge_attr_dict, check_nans=True)
        print(output)

    def set_meta_data(self, x_dict):
        meta_data = dict()

        for node_type in x_dict:
            meta_data[node_type] = len(x_dict[node_type])

        self.meta_data = meta_data

    def init_layer_norm(self, x_dict):
        self.set_meta_data(x_dict)
        self.norms = ModuleList()
        norm_layer = HeteroMinMaxNorm()

        # for _ in range(self.num_layers):
        # self.norms.append(copy.deepcopy(norm_layer))
        self.norms.append(norm_layer)

    def group(self, xs: List[Tensor]) -> Optional[Tensor]:

        out = torch.cat(xs, dim=0)
        half = int(self.hidden_channels/2)

        P_ij = out[:, :half]
        Q_ij = out[:, half:]

        if self.aggr is not None:
            P_i = getattr(torch, self.aggr)(P_ij, dim=1)
            Q_i = getattr(torch, self.aggr)(Q_ij, dim=1)

        else:
            P_i = self.aggr_P(P_ij, dim=1)
            print(f"P_i : {P_i}")
            Q_i = self.aggr_Q(Q_ij, dim=1)
            print(f"Q_i : {Q_i}")

        if len(xs) > 1:
            out = torch.cat([P_i.view(-1, 1), Q_i.view(-1, 1)], dim=1)

        else:
            out = torch.stack([P_i, Q_i])
        return out

    def group_wired(self, xs: List[Tensor]) -> Optional[Tensor]:

        out = torch.cat(xs, dim=0)
        half = int(self.hidden_channels/2)
        quarter = int(self.hidden_channels/4)

        V = out[:, :quarter]
        delta = out[:, quarter:half]
        P = out[:, half:half+quarter]
        Q = out[:, half+quarter:]


        if self.aggr is not None:
            V_i = getattr(torch, self.aggr)(V, dim=1)
            delta_i = getattr(torch, self.aggr)(delta, dim=1)
            P_i = getattr(torch, self.aggr)(P, dim=1)
            Q_i = getattr(torch, self.aggr)(Q, dim=1)

        else:
            P_i = self.aggr_P(P, dim=1)
            print(f"P_i : {P_i}")
            Q_i = self.aggr_Q(Q, dim=1)
            print(f"Q_i : {Q_i}")

        if len(xs) > 1:
            out = torch.cat([V_i.view(-1, 1),delta_i.view(-1, 1),P_i.view(-1, 1), Q_i.view(-1, 1)], dim=1)

        else:
            out = torch.stack([V_i,delta_i,P_i, Q_i])
        return out

    def forward(self, x_dict, bus_idx_neighbors_dict, edge_idx_dict, edge_attr_dict,
                check_nans: bool = False):


        if check_nans:
            for node_type in x_dict:
                x = x_dict[node_type]
                if torch.isnan(x).any():
                    print(f"nan found in node type {node_type} of x_dict")

        # All calls of the forward function must support edge_attr & edge_idx for this model

        out_dict = defaultdict(list)

        for i in range(self.num_layers):

            for from_bus in bus_idx_neighbors_dict:
                conv = self.convs[i][from_bus]
                to_xs = []
                edge_attr = []
                edge_idx = [[], []]
                #print(f"from_bus : {from_bus}")

                for bus_idx in bus_idx_neighbors_dict[from_bus]:

                    for pair in bus_idx_neighbors_dict[from_bus][bus_idx]:
                        # For each neighbor of the iterated bus
                        to_bus, to_bus_idx, edge_attribute = pair
                        to_x = x_dict[to_bus][to_bus_idx]
                        to_xs.append(to_x.view(1, self.in_channels))
                        #print(f"bus_idx: {bus_idx}, to_idx: {len(edge_idx[0])}")
                        edge_idx[1].append(bus_idx)
                        edge_idx[0].append(len(edge_idx[0]))
                        edge_attr.append(edge_attribute.view(1,2))

                #print(f"edge_idx: {edge_idx}")
                edge_attr = torch.cat(edge_attr,dim=0)
                #print(f"edge_attr: {edge_attr.shape}")
                x_dst = torch.cat(to_xs, dim=0)
                #print(f"x_dst shape: {x_dst.shape}")
                #print(f"x_src shape: {x_dict[from_bus].shape}")

                out = conv((x_dst, x_dict[from_bus]), edge_index=torch.tensor(edge_idx), edge_attr=edge_attr)

                out_dict[from_bus].append(out)
                #print(f"shape: {out.shape}")


        # Calculate the Aggregations
        for key, value in out_dict.items():
            #print(f"key: {key}")
            #shape = [x.shape for x in value]
            #print(f"shape before Aggr {shape}")

            val = self.group_wired(value)

            #print(f"shape after Aggr {val.shape}")
            fc = self.fcs[key]

            # Flatten the tensor
            _out_ = val.view(-1)
            #print(f"shape after flattened {_out_.shape}")

            out = fc(_out_)

            out_dict[key] = out.view(self.node_types_dict[key], self.out_channels)

            #print(f"{key}: {len(out_dict[key])}")




        return out_dict

class ACOPFGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, act_fn: str = "elu"
                 , norm: Union[str, Callable, None] = None, init_data=None):
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

        self.norms = None
        self.meta_data = None
        self.aggr = "sum"
        self.heads = 1
        self.in_channels = 4
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act_fn = act_fn
        self.act = activation_resolver(act_fn, **({}))
        self.act_first = False
        self.jk_mode = "last"
        in_channels = hidden_channels * self.heads
        self.lin = Linear(-1, self.out_channels, bias=True)
        self.bias = nn.Parameter(torch.tensor(self.out_channels, dtype=torch.float32))
        self.lin_2 = Linear(-1, self.out_channels, bias=True)
        self.node_types = ["SB", "PQ", "PV", "NB"]

        # if norm is not None:
        # norm_layer = HeteroLayerNorm(

        # """normalization_resolver(
        #   norm,
        #   hidden_channels,
        #   **({}),
        # )"""

        # self.norms = ModuleList()

        # for _ in range(num_layers):
        # for _ in self.edge_types:
        # self.norms.append(copy.deepcopy(norm_layer))

        self.edge_types = [
            'PV__isConnected__SB',
            'SB__isConnected__PQ',
            'SB__isConnected__NB',

            'PV__isConnected__PQ',
            'NB__isConnected__PQ',
            'PQ__isConnected__NB',

            'SB__isConnected__PV',
            'PQ__isConnected__SB',
            'NB__isConnected__SB',

            'PQ__isConnected__PV',
            'PV__isConnected__NB',
            'NB__isConnected__PV',

            'PV__isConnected__PV',
            'PQ__isConnected__PQ',
            'NB__isConnected__NB',

        ]
        self.convs = ModuleList()
        self.fcs = ModuleList()

        for _ in range(self.num_layers):
            conv = ModuleDict({
                'SB__isConnected__PQ': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),

                'PV__isConnected__PQ': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),
                'NB__isConnected__PQ': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),
                'PQ__isConnected__NB': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),

                'PQ__isConnected__SB': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),

                'PQ__isConnected__PV': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),
                'PV__isConnected__NB': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),
                'NB__isConnected__PV': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),

                'PV__isConnected__PV': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),
                'PQ__isConnected__PQ': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads),
                'NB__isConnected__NB': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                # TransformerConv(-1, hidden_channels, edge_dim=2, heads=self.heads)
            })

            check_add_self_loops(conv, self.edge_types)

            self.convs.append(conv)


        fc = ModuleDict()
        for node_type in self.node_types:
            fc[node_type] = Linear(-1, self.out_channels, bias=True)
        self.fcs = fc

    def lazy_init(self, lazy_input, net, index_mappers):
        x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, scaler, _, _ = hp.extract_unsupervised_inputs(
            lazy_input, net, index_mappers)
        self.node_types = list(set(self.node_types).intersection(set(x_dict.keys())))
        self.init_layer_norm(x_dict)
        output = self.forward(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, check_nans=True)
        print(output)

    def set_meta_data(self, x_dict):
        meta_data = dict()

        for node_type in x_dict:
            meta_data[node_type] = len(x_dict[node_type])

        self.meta_data = meta_data

    def init_layer_norm(self, x_dict):
        self.set_meta_data(x_dict)
        self.norms = ModuleList()
        norm_layer = HeteroMinMaxNorm()

        # for _ in range(self.num_layers):
        # self.norms.append(copy.deepcopy(norm_layer))
        self.norms.append(norm_layer)

    # def _Forward_(self,x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict,
    #              check_nans: bool = False):
    #     torch.autograd.set_detect_anomaly(True)
    #     if check_nans:
    #         for node_type in x_dict:
    #             x = x_dict[node_type]
    #             if torch.isnan(x).any():
    #                 print(f"nan found in node type {node_type} of x_dict")
    #
    #     # All calls of the forward function must support edge_attr & edge_idx for this model
    #
    #     out_dict = defaultdict(list)
    #
    #     for i in range(self.num_layers):
    #
    #         for edge_type, edge_index in edge_idx_dict.items():
    #             src, rel, dst = edge_type
    #             edge_attr = edge_attr_dict[edge_type]
    #
    #             str_edge_type = '__'.join(edge_type)
    #             if str_edge_type not in self.convs[i]:
    #                 self.convs[i][str_edge_type] = copy.deepcopy(self.base_model)
    #
    #             # Convolute the corresponding node features at the respective TransformerConv layer of the edge type
    #             conv = self.convs[i][str_edge_type]
    #
    #             if src == dst:
    #                 out = conv(x_dict[src], edge_index=edge_index, edge_attr=edge_attr)
    #             else:
    #                 out = conv((x_dict[src], x_dict[dst]), edge_index=edge_index, edge_attr=edge_attr)
    #
    #             if check_nans and torch.isnan(out).any():
    #                 print(f"NAN FOUND in out")
    #
    #             out_dict[dst].append(out)
    #
    #     # Calculate the Aggregations
    #     for key, value in out_dict.items():
    #         out_dict[key] = group(list(value), self.aggr)
    #         if check_nans and torch.isnan(out_dict[key]).any():
    #             print(f"NAN FOUND in AGGREGATION")
    #
    #     # Convert the node features to the desired shape being Nx4 with N=Number of Nodes(Busses)
    #     for node_type in out_dict:
    #         out_dict[node_type] = self.act(self.lin(out_dict[node_type]))
    #         if check_nans and torch.isnan(out_dict[node_type]).any():
    #             print(f"NAN FOUND in LINEARIZING")
    #         # print(f"Nan Value present in out_dict after linear operations?: {torch.isnan(out_dict).any()}")
    #
    #     acopf_out_dict = out_dict.copy()
    #
    #     # ACOPF Forward Pass for P and Q
    #     for from_bus in bus_idx_neighbors_dict:
    #         for bus_idx in bus_idx_neighbors_dict[from_bus]:
    #             # Store the constraints
    #             volt_mag_lower_bound = constraint_dict[from_bus][bus_idx][0]
    #             volt_mag_upper_bound = constraint_dict[from_bus][bus_idx][1]
    #             max_apparent_pow = constraint_dict[from_bus][bus_idx][2]
    #             active_pow_lower_bound = constraint_dict[from_bus][bus_idx][3]
    #             active_pow_upper_bound = constraint_dict[from_bus][bus_idx][4]
    #             reactive_pow_lower_bound = constraint_dict[from_bus][bus_idx][5]
    #             reactive_pow_upper_bound = constraint_dict[from_bus][bus_idx][6]
    #
    #             V_i = out_dict[from_bus][bus_idx][0].clone().detach()
    #             volt_angle_i = out_dict[from_bus][bus_idx][1].clone().detach()
    #             P_i = out_dict[from_bus][bus_idx][2].clone().detach()
    #             Q_i = out_dict[from_bus][bus_idx][3].clone().detach()
    #
    #             P = []
    #             Q = []
    #             # V = []
    #             # delta = []
    #
    #             for pair in bus_idx_neighbors_dict[from_bus][bus_idx]:
    #                 # For each neighbor of the iterated bus
    #                 to_bus, to_bus_idx, edge_attr = pair
    #
    #                 V_j = acopf_out_dict[to_bus][to_bus_idx][0].clone().detach()
    #
    #                 if to_bus_idx >= len(out_dict[to_bus]):
    #                     print("INDEX PROBLEM")
    #
    #                 volt_angle_j = acopf_out_dict[to_bus][to_bus_idx][1].clone().detach()
    #                 delta_ij = volt_angle_j - volt_angle_i
    #
    #                 G_ij = edge_attr[0] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)
    #                 if check_nans and torch.isnan(G_ij).any():
    #                     print("Nan Value present in G_ij")
    #
    #                 B_ij = -edge_attr[1] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)
    #                 if check_nans and torch.isnan(B_ij).any():
    #                     print(f"Nan Value present in B_ij")
    #
    #                 # ACOPF Equation for P_i
    #                 P_ij = V_i * V_j * (
    #                         G_ij.item() * torch.cos(delta_ij) + B_ij.item() * torch.sin(
    #                     delta_ij)) if node_type != "NB" else torch.tensor(0.0, requires_grad=True)
    #                 if check_nans and torch.isnan(P_ij).any():
    #                     print(f"from_bus: {from_bus}, to_bus: {to_bus}, bus_idx: {bus_idx}, to_bus_idx: {to_bus_idx}")
    #                     print(f"Nan Value present in P_ij")
    #                     print(f"V_j_positive: {V_j}"
    #                           f" volt_angle_j: {volt_angle_j}")
    #
    #                 P.append(P_ij)
    #
    #                 # ACOPF Equation for Q_i
    #                 Q_ij = V_i * V_j * (
    #                         G_ij.item() * torch.sin(delta_ij) - B_ij.item() * torch.cos(
    #                     delta_ij)) if node_type != "NB" else torch.tensor(0.0, requires_grad=True)
    #                 if check_nans and torch.isnan(Q_ij).any():
    #                     print(f"Nan Value present in Q_ij")
    #                     print(
    #                         f"V_j_positive: {V_j}"
    #                         f" volt_angle_j: {volt_angle_j}")
    #
    #                 Q.append(Q_ij)
    #                 """
    #                 # Calculate V_i alternatingly
    #                 V_new = self.act(0.5 * (P_ij / (((
    #                         G_ij.item() * self.act(delta_ij) + B_ij.item() * self.act(
    #                     delta_ij))) * V_j)
    #                                         + Q_ij / (((
    #                                     G_ij.item() * self.act(delta_ij) - B_ij.item() * self.act(
    #                                 delta_ij))) * V_j)))
    #                 V.append(V_new)
    #                 """
    #                 """
    #                 # Calculate delta_ij alternatingly
    #                 # Calculate Voltage Angle from the mean
    #                 volt_angle_ij = 0.5 * hp.arccos_approx(torch.sqrt(torch.square(P_ij.clone().detach()/max_apparent_pow.item()))) + hp.arcsin_approx(torch.sqrt(torch.square(Q_ij.clone().detach()/max_apparent_pow.item())))
    #                 delta.append(volt_angle_ij)
    #                 """
    #             # Sums of all P_ij and Q_ij equal to P_i and Q_i respectively
    #             P_ = torch.stack(P)
    #             P_new = self.act(self.P_alpha * torch.sum(P_) + P_i)
    #             Q_ = torch.stack(Q)
    #             Q_new = self.act(self.Q_alpha * torch.sum(Q_) + Q_i)
    #             # V_ = torch.stack(V)
    #             # V_mean = self.act(self.V_alpha * torch.mean(V_) + V_i)
    #             # V_mean = self.act(V_i.clone().detach())
    #             # delta_ = torch.stack(delta)
    #             # delta_mean = self.act(self.delta_alpha * torch.sum(delta_) + volt_angle_i)
    #             # delta_mean = self.act(volt_angle_i.clone().detach())
    #
    #             # Enforce the constraints
    #
    #             # Active Power boundary constraint
    #             # P_i_enforced = custom_tanh(P_new, min(active_pow_lower_bound.item(), active_pow_upper_bound.item()), max(active_pow_lower_bound.item(), active_pow_upper_bound.item())) if node_type != "NB" else torch.tensor(0.0,requires_grad=False) * P_new
    #             P_i_enforced = torch.min(torch.max(active_pow_lower_bound, P_new),
    #                                      active_pow_upper_bound) if node_type != "NB" else torch.tensor(0.0,
    #                                                                                                     requires_grad=False)
    #             if check_nans and torch.isnan(P_new).any():
    #                 print(f"Nan Value present in P_i after S enforcement")
    #                 # print(f" P Lower bound {active_pow_lower_bound}, P upper bound {active_pow_upper_bound}")
    #
    #             # Reactive Power boundary constraint
    #             # Q_i_enforced = custom_tanh(Q_new, min(reactive_pow_lower_bound.item(), reactive_pow_upper_bound.item()), max(reactive_pow_lower_bound.item(), reactive_pow_upper_bound.item())) if node_type != "NB" else torch.tensor(0.0, requires_grad=False) * Q_new
    #             Q_i_enforced = torch.min(torch.max(reactive_pow_lower_bound, Q_new),
    #                                      reactive_pow_upper_bound) if node_type != "NB" else torch.tensor(0.0,
    #                                                                                                       requires_grad=False)
    #             if check_nans and torch.isnan(Q_new).any():
    #                 print(f"Nan Value present in Q_i after S enforcement")
    #                 # print(f" Q Lower bound {reactive_pow_lower_bound}, Q upper bound {reactive_pow_upper_bound}")
    #
    #             # Enforce the constraints
    #             # Voltage Magnitude boundary constraint
    #             # V_i_enforced = custom_tanh(V_mean, min(volt_mag_lower_bound.item(),volt_mag_upper_bound.item()), max(volt_mag_lower_bound.item(),volt_mag_upper_bound.item()))  # /x_dict[from_bus][bus_idx][0]
    #
    #             V_i_enforced = torch.min(torch.max(volt_mag_lower_bound, V_i), volt_mag_upper_bound)
    #
    #             # Voltage Angle activation
    #             delta_i_double_enforced = self.act(volt_angle_i) + 0.5 * (hp.arccos_approx(torch.sqrt(
    #                 torch.square(P_i_enforced.clone().detach() / max_apparent_pow.item()))) + hp.arcsin_approx(
    #                 torch.sqrt(torch.square(
    #                     Q_i_enforced.clone().detach() / max_apparent_pow.item())))) if node_type != "NB" else self.act(
    #                 volt_angle_i)
    #             delta_i_double_enforced = 0.5 * (torch.arccos(P_i_enforced / max_apparent_pow) + torch.arcsin(
    #                 Q_i_enforced / max_apparent_pow))
    #
    #             acopf_out_dict[from_bus][bus_idx] = torch.stack(
    #                 [V_i_enforced, delta_i_double_enforced, P_i_enforced, Q_i_enforced])
    #
    #     return acopf_out_dict

    def forward(self, _x_dict, constraint_dict, edge_idx_dict, edge_attr_dict,
                check_nans: bool = False):
        x_dict = dict()

        for node_type in _x_dict:
            x = _x_dict[node_type]
            c = constraint_dict[node_type]
            x_dict[node_type] = torch.cat((x, c), dim=1)

        torch.autograd.set_detect_anomaly(True)
        if check_nans:
            for node_type in x_dict:
                x = x_dict[node_type]
                if torch.isnan(x).any():
                    print(f"nan found in node type {node_type} of x_dict")

        # All calls of the forward function must support edge_attr & edge_idx for this model

        out_dict = defaultdict(list)

        for i in range(self.num_layers):

            for edge_type, edge_index in edge_idx_dict.items():
                src, rel, dst = edge_type
                edge_attr = edge_attr_dict[edge_type]

                str_edge_type = '__'.join(edge_type)
                if str_edge_type not in self.convs[i]:
                    self.convs[i][str_edge_type] = TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                                   heads=self.heads)

                # Convolute the corresponding node features at the respective TransformerConv layer of the edge type
                conv = self.convs[i][str_edge_type]

                if src == dst:
                    out = conv(x_dict[src], edge_index=edge_index, edge_attr=edge_attr)
                else:
                    out = conv((x_dict[src], x_dict[dst]), edge_index=edge_index, edge_attr=edge_attr)

                if check_nans and torch.isnan(out).any():
                    print(f"NAN FOUND in out")

                out_dict[dst].append(out)

        # Calculate the Aggregations
        for key, value in out_dict.items():
            val = group(list(value), self.aggr)
            out_dict[key] = self.act(self.lin(val))
            if check_nans and torch.isnan(out_dict[key]).any():
                print(f"NAN FOUND in AGGREGATION")

        # Normalization
        norm_layer = self.norms[0]
        # for node_type in out_dict:
        # out_dict[node_type] = torch.stack(flatten_and_reshape_tensors(out_dict[node_type]))
        # print(f"shape of {node_type} is {out_dict[node_type].size()}")
        out_dict = norm_layer(out_dict, self.node_types)

        node_dict = defaultdict(list)

        for i in range(self.num_layers):
            for key, value in out_dict.items():
                fc = self.fcs[i][key]
                out = self.act(fc(value))
                node_dict[key].append(out)

        # Calculate the Aggregations
        for key, value in node_dict.items():
            val = group(list(value), self.aggr)
            node_dict[key] = self.act(self.lin_2(val))

            # if key == "NB":
            #   features = self.act(self.lin_2(val))
            #  V_node = features[:, 0].view(-1,1)
            # delta_node = features[:, 1].view(-1,1)
            # P_node = torch.tensor([0.0]*features.size(0)).view(-1,1)
            # Q_node = torch.tensor([0.0]*features.size(0)).view(-1,1)
            # enforced_features = torch.cat([V_node, delta_node, P_node, Q_node], dim=1)
            # node_dict[key] = enforced_features

            if check_nans and torch.isnan(node_dict[key]).any():
                print(f"NAN FOUND in AGGREGATION")

        return node_dict


class ACOPFEmbedder(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, act_fn: str = "elu"
                 , norm: Union[str, Callable, None] = None, init_data=None):
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

        self.norms = None
        self.meta_data = None
        self.aggr = "sum"
        self.heads = 1
        self.in_channels = 2
        self.hidden_channels = hidden_channels
        self.out_channels = 2
        self.num_layers = num_layers
        self.dropout = dropout
        self.act_fn = act_fn
        self.act = activation_resolver(act_fn, **({}))
        self.act_first = False
        self.jk_mode = "last"
        in_channels = hidden_channels * self.heads
        #self.lin = Linear(-1, self.out_channels, bias=True)
        #self.lin_aggr = Linear(-1, self.out_channels, bias=True)
        #self.bias = nn.Parameter(torch.tensor(self.out_channels, dtype=torch.float32))
        self.node_types = ["SB", "PQ", "PV", "NB"]

        # if norm is not None:
        # norm_layer = HeteroLayerNorm(

        # """normalization_resolver(
        #   norm,
        #   hidden_channels,
        #   **({}),
        # )"""

        # self.norms = ModuleList()

        # for _ in range(num_layers):
        # for _ in self.edge_types:
        # self.norms.append(copy.deepcopy(norm_layer))

        self.edge_types = [
            'PV__isConnected__SB',
            'SB__isConnected__PQ',
            'SB__isConnected__NB',

            'PV__isConnected__PQ',
            'NB__isConnected__PQ',
            'PQ__isConnected__NB',

            'SB__isConnected__PV',
            'PQ__isConnected__SB',
            'NB__isConnected__SB',

            'PQ__isConnected__PV',
            'PV__isConnected__NB',
            'NB__isConnected__PV',

            'PV__isConnected__PV',
            'PQ__isConnected__PQ',
            'NB__isConnected__NB',

        ]
        self.convs = ModuleList()
        #self.splines = ModuleList()

        for _ in range(self.num_layers):
            conv = ModuleDict({
                'SB__isConnected__PQ': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),

                'PV__isConnected__PQ': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                'NB__isConnected__PQ': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                'PQ__isConnected__NB': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),

                'PQ__isConnected__SB': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),

                'PQ__isConnected__PV': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                'PV__isConnected__NB': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                'NB__isConnected__PV': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),

                'PV__isConnected__PV': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                'PQ__isConnected__PQ': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
                'NB__isConnected__NB': TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                       heads=self.heads),
            })
            """
            spline = ModuleDict({
                'SB__isConnected__PQ': GATConv(-1, 2, edge_dim=2),

                'PV__isConnected__PQ': GATConv(-1, 2, edge_dim=2),
                'NB__isConnected__PQ': GATConv(-1, 2, edge_dim=2),
                'PQ__isConnected__NB': GATConv(-1, 2, edge_dim=2),

                'PQ__isConnected__SB': GATConv(-1, 2, edge_dim=2),

                'PQ__isConnected__PV': GATConv(-1, 2, edge_dim=2),
                'PV__isConnected__NB': GATConv(-1, 2, edge_dim=2),
                'NB__isConnected__PV': GATConv(-1, 2, edge_dim=2),

                'PV__isConnected__PV': GATConv(-1, 2, edge_dim=2),
                'PQ__isConnected__PQ': GATConv(-1, 2, edge_dim=2),
                'NB__isConnected__NB': GATConv(-1, 2, edge_dim=2)
            })
            """
            self.convs.append(conv)
            #self.splines.append(spline)

        self.node_types_dict = {key: init_data[key].x.size(0) for key in init_data.node_types}

        fc = ModuleDict()
        for node_type in self.node_types_dict:
            size = self.node_types_dict[node_type]
            fc[node_type] = Linear(-1, int(size*2), bias=True)

        self.fcs = fc

    def lazy_init(self, lazy_input, net, index_mappers):
        x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, scaler, _, _ = hp.extract_unsupervised_inputs(
            lazy_input, net, index_mappers)

        self.init_layer_norm(x_dict)
        output = self.forward(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, check_nans=True)
        print(output)

    def set_meta_data(self, x_dict):
        meta_data = dict()

        for node_type in x_dict:
            meta_data[node_type] = len(x_dict[node_type])

        self.meta_data = meta_data

    def init_layer_norm(self, x_dict):
        self.set_meta_data(x_dict)
        self.norms = ModuleList()
        norm_layer = HeteroMinMaxNorm()

        # for _ in range(self.num_layers):
        # self.norms.append(copy.deepcopy(norm_layer))
        self.norms.append(norm_layer)

    # def _Forward_(self,x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict,
    #              check_nans: bool = False):
    #     torch.autograd.set_detect_anomaly(True)
    #     if check_nans:
    #         for node_type in x_dict:
    #             x = x_dict[node_type]
    #             if torch.isnan(x).any():
    #                 print(f"nan found in node type {node_type} of x_dict")
    #
    #     # All calls of the forward function must support edge_attr & edge_idx for this model
    #
    #     out_dict = defaultdict(list)
    #
    #     for i in range(self.num_layers):
    #
    #         for edge_type, edge_index in edge_idx_dict.items():
    #             src, rel, dst = edge_type
    #             edge_attr = edge_attr_dict[edge_type]
    #
    #             str_edge_type = '__'.join(edge_type)
    #             if str_edge_type not in self.convs[i]:
    #                 self.convs[i][str_edge_type] = copy.deepcopy(self.base_model)
    #
    #             # Convolute the corresponding node features at the respective TransformerConv layer of the edge type
    #             conv = self.convs[i][str_edge_type]
    #
    #             if src == dst:
    #                 out = conv(x_dict[src], edge_index=edge_index, edge_attr=edge_attr)
    #             else:
    #                 out = conv((x_dict[src], x_dict[dst]), edge_index=edge_index, edge_attr=edge_attr)
    #
    #             if check_nans and torch.isnan(out).any():
    #                 print(f"NAN FOUND in out")
    #
    #             out_dict[dst].append(out)
    #
    #     # Calculate the Aggregations
    #     for key, value in out_dict.items():
    #         out_dict[key] = group(list(value), self.aggr)
    #         if check_nans and torch.isnan(out_dict[key]).any():
    #             print(f"NAN FOUND in AGGREGATION")
    #
    #     # Convert the node features to the desired shape being Nx4 with N=Number of Nodes(Busses)
    #     for node_type in out_dict:
    #         out_dict[node_type] = self.act(self.lin(out_dict[node_type]))
    #         if check_nans and torch.isnan(out_dict[node_type]).any():
    #             print(f"NAN FOUND in LINEARIZING")
    #         # print(f"Nan Value present in out_dict after linear operations?: {torch.isnan(out_dict).any()}")
    #
    #     acopf_out_dict = out_dict.copy()
    #
    #     # ACOPF Forward Pass for P and Q
    #     for from_bus in bus_idx_neighbors_dict:
    #         for bus_idx in bus_idx_neighbors_dict[from_bus]:
    #             # Store the constraints
    #             volt_mag_lower_bound = constraint_dict[from_bus][bus_idx][0]
    #             volt_mag_upper_bound = constraint_dict[from_bus][bus_idx][1]
    #             max_apparent_pow = constraint_dict[from_bus][bus_idx][2]
    #             active_pow_lower_bound = constraint_dict[from_bus][bus_idx][3]
    #             active_pow_upper_bound = constraint_dict[from_bus][bus_idx][4]
    #             reactive_pow_lower_bound = constraint_dict[from_bus][bus_idx][5]
    #             reactive_pow_upper_bound = constraint_dict[from_bus][bus_idx][6]
    #
    #             V_i = out_dict[from_bus][bus_idx][0].clone().detach()
    #             volt_angle_i = out_dict[from_bus][bus_idx][1].clone().detach()
    #             P_i = out_dict[from_bus][bus_idx][2].clone().detach()
    #             Q_i = out_dict[from_bus][bus_idx][3].clone().detach()
    #
    #             P = []
    #             Q = []
    #             # V = []
    #             # delta = []
    #
    #             for pair in bus_idx_neighbors_dict[from_bus][bus_idx]:
    #                 # For each neighbor of the iterated bus
    #                 to_bus, to_bus_idx, edge_attr = pair
    #
    #                 V_j = acopf_out_dict[to_bus][to_bus_idx][0].clone().detach()
    #
    #                 if to_bus_idx >= len(out_dict[to_bus]):
    #                     print("INDEX PROBLEM")
    #
    #                 volt_angle_j = acopf_out_dict[to_bus][to_bus_idx][1].clone().detach()
    #                 delta_ij = volt_angle_j - volt_angle_i
    #
    #                 G_ij = edge_attr[0] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)
    #                 if check_nans and torch.isnan(G_ij).any():
    #                     print("Nan Value present in G_ij")
    #
    #                 B_ij = -edge_attr[1] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)
    #                 if check_nans and torch.isnan(B_ij).any():
    #                     print(f"Nan Value present in B_ij")
    #
    #                 # ACOPF Equation for P_i
    #                 P_ij = V_i * V_j * (
    #                         G_ij.item() * torch.cos(delta_ij) + B_ij.item() * torch.sin(
    #                     delta_ij)) if node_type != "NB" else torch.tensor(0.0, requires_grad=True)
    #                 if check_nans and torch.isnan(P_ij).any():
    #                     print(f"from_bus: {from_bus}, to_bus: {to_bus}, bus_idx: {bus_idx}, to_bus_idx: {to_bus_idx}")
    #                     print(f"Nan Value present in P_ij")
    #                     print(f"V_j_positive: {V_j}"
    #                           f" volt_angle_j: {volt_angle_j}")
    #
    #                 P.append(P_ij)
    #
    #                 # ACOPF Equation for Q_i
    #                 Q_ij = V_i * V_j * (
    #                         G_ij.item() * torch.sin(delta_ij) - B_ij.item() * torch.cos(
    #                     delta_ij)) if node_type != "NB" else torch.tensor(0.0, requires_grad=True)
    #                 if check_nans and torch.isnan(Q_ij).any():
    #                     print(f"Nan Value present in Q_ij")
    #                     print(
    #                         f"V_j_positive: {V_j}"
    #                         f" volt_angle_j: {volt_angle_j}")
    #
    #                 Q.append(Q_ij)
    #                 """
    #                 # Calculate V_i alternatingly
    #                 V_new = self.act(0.5 * (P_ij / (((
    #                         G_ij.item() * self.act(delta_ij) + B_ij.item() * self.act(
    #                     delta_ij))) * V_j)
    #                                         + Q_ij / (((
    #                                     G_ij.item() * self.act(delta_ij) - B_ij.item() * self.act(
    #                                 delta_ij))) * V_j)))
    #                 V.append(V_new)
    #                 """
    #                 """
    #                 # Calculate delta_ij alternatingly
    #                 # Calculate Voltage Angle from the mean
    #                 volt_angle_ij = 0.5 * hp.arccos_approx(torch.sqrt(torch.square(P_ij.clone().detach()/max_apparent_pow.item()))) + hp.arcsin_approx(torch.sqrt(torch.square(Q_ij.clone().detach()/max_apparent_pow.item())))
    #                 delta.append(volt_angle_ij)
    #                 """
    #             # Sums of all P_ij and Q_ij equal to P_i and Q_i respectively
    #             P_ = torch.stack(P)
    #             P_new = self.act(self.P_alpha * torch.sum(P_) + P_i)
    #             Q_ = torch.stack(Q)
    #             Q_new = self.act(self.Q_alpha * torch.sum(Q_) + Q_i)
    #             # V_ = torch.stack(V)
    #             # V_mean = self.act(self.V_alpha * torch.mean(V_) + V_i)
    #             # V_mean = self.act(V_i.clone().detach())
    #             # delta_ = torch.stack(delta)
    #             # delta_mean = self.act(self.delta_alpha * torch.sum(delta_) + volt_angle_i)
    #             # delta_mean = self.act(volt_angle_i.clone().detach())
    #
    #             # Enforce the constraints
    #
    #             # Active Power boundary constraint
    #             # P_i_enforced = custom_tanh(P_new, min(active_pow_lower_bound.item(), active_pow_upper_bound.item()), max(active_pow_lower_bound.item(), active_pow_upper_bound.item())) if node_type != "NB" else torch.tensor(0.0,requires_grad=False) * P_new
    #             P_i_enforced = torch.min(torch.max(active_pow_lower_bound, P_new),
    #                                      active_pow_upper_bound) if node_type != "NB" else torch.tensor(0.0,
    #                                                                                                     requires_grad=False)
    #             if check_nans and torch.isnan(P_new).any():
    #                 print(f"Nan Value present in P_i after S enforcement")
    #                 # print(f" P Lower bound {active_pow_lower_bound}, P upper bound {active_pow_upper_bound}")
    #
    #             # Reactive Power boundary constraint
    #             # Q_i_enforced = custom_tanh(Q_new, min(reactive_pow_lower_bound.item(), reactive_pow_upper_bound.item()), max(reactive_pow_lower_bound.item(), reactive_pow_upper_bound.item())) if node_type != "NB" else torch.tensor(0.0, requires_grad=False) * Q_new
    #             Q_i_enforced = torch.min(torch.max(reactive_pow_lower_bound, Q_new),
    #                                      reactive_pow_upper_bound) if node_type != "NB" else torch.tensor(0.0,
    #                                                                                                       requires_grad=False)
    #             if check_nans and torch.isnan(Q_new).any():
    #                 print(f"Nan Value present in Q_i after S enforcement")
    #                 # print(f" Q Lower bound {reactive_pow_lower_bound}, Q upper bound {reactive_pow_upper_bound}")
    #
    #             # Enforce the constraints
    #             # Voltage Magnitude boundary constraint
    #             # V_i_enforced = custom_tanh(V_mean, min(volt_mag_lower_bound.item(),volt_mag_upper_bound.item()), max(volt_mag_lower_bound.item(),volt_mag_upper_bound.item()))  # /x_dict[from_bus][bus_idx][0]
    #
    #             V_i_enforced = torch.min(torch.max(volt_mag_lower_bound, V_i), volt_mag_upper_bound)
    #
    #             # Voltage Angle activation
    #             delta_i_double_enforced = self.act(volt_angle_i) + 0.5 * (hp.arccos_approx(torch.sqrt(
    #                 torch.square(P_i_enforced.clone().detach() / max_apparent_pow.item()))) + hp.arcsin_approx(
    #                 torch.sqrt(torch.square(
    #                     Q_i_enforced.clone().detach() / max_apparent_pow.item())))) if node_type != "NB" else self.act(
    #                 volt_angle_i)
    #             delta_i_double_enforced = 0.5 * (torch.arccos(P_i_enforced / max_apparent_pow) + torch.arcsin(
    #                 Q_i_enforced / max_apparent_pow))
    #
    #             acopf_out_dict[from_bus][bus_idx] = torch.stack(
    #                 [V_i_enforced, delta_i_double_enforced, P_i_enforced, Q_i_enforced])
    #
    #     return acopf_out_dict

    def forward(self, _x_dict, constraint_dict, edge_idx_dict, edge_attr_dict,
                check_nans: bool = False):

        x_dict = dict()
        for node_type in _x_dict:
            P = _x_dict[node_type][:,2].view(-1,1)
            Q = _x_dict[node_type][:, 3].view(-1, 1)
            powers = torch.cat([P,Q],dim=1)
            x_dict[node_type] = powers

        if check_nans:
            for node_type in x_dict:
                x = x_dict[node_type]
                if torch.isnan(x).any():
                    print(f"nan found in node type {node_type} of x_dict")

        # All calls of the forward function must support edge_attr & edge_idx for this model

        out_dict = defaultdict(list)

        for i in range(self.num_layers):

            for edge_type, edge_index in edge_idx_dict.items():
                src, rel, dst = edge_type
                edge_attr = edge_attr_dict[edge_type]

                mapper_from = dict()
                mapper_to = dict()

                for j in range(len(edge_index[0])):
                    from_idx = edge_index[0][j].item()

                    if from_idx not in mapper_from:
                        mapper_from[from_idx] = j

                for k in range(len(edge_index[1])):
                    to_idx = edge_index[1][k].item()

                    if to_idx not in mapper_to:
                        mapper_to[to_idx] = k

                from_idx = [mapper_from[key.item()] for key in edge_index[0]]
                to_idx = [mapper_to[key.item()] for key in edge_index[1]]

                node_idx_edge = torch.tensor([from_idx, to_idx])

                str_edge_type = '__'.join(edge_type)

                if str_edge_type not in self.convs[i]:
                    self.convs[i][str_edge_type] = TransformerConv(-1, self.hidden_channels, edge_dim=2,
                                                                   heads=self.heads)

                    #self.splines[i][str_edge_type] = SplineConv(-1, self.out_channels, 2, 4, degree=3)

                # Convolute the corresponding node features at the respective TransformerConv layer of the edge type
                conv = self.convs[i][str_edge_type]
                #spline = self.splines[i][str_edge_type]

                if src == dst:
                    out = conv((x_dict[src][edge_index[0]], x_dict[dst][edge_index[1]]), edge_index=node_idx_edge, edge_attr=edge_attr)
                    #out = spline(out ,edge_index=edge_index, edge_attr=edge_attr)

                else:
                    out = conv((x_dict[src][edge_index[0]], x_dict[dst][edge_index[1]]), edge_index=node_idx_edge, edge_attr=edge_attr)
                    #out = spline(out, edge_index=edge_index,edge_attr=edge_attr)

                #out_dict[dst].append(self.act(out))
                #print(f"str edge type: {str_edge_type}, shape: {out.shape}")

                out_dict[dst].append(out)


        # Calculate the Aggregations
        for key, value in out_dict.items():
            #print(f"key: {key}")
            #shape = [x.shape for x in value]
            #print(f"shape before Aggr {shape}")

            val = group(value, self.hidden_channels, self.aggr)
            #print(f"shape after Aggr {val.shape}")
            fc = self.fcs[key]

            # Flatten the tensor
            _out_ = val.view(-1)
            #print(f"shape after flattened {_out_.shape}")

            out = fc(_out_)

            out_dict[key] = out.view(self.node_types_dict[key], self.out_channels)

            #print(f"{key}: {len(out_dict[key])}")
            if check_nans and torch.isnan(out_dict[key]).any():
                print(f"NAN FOUND in AGGREGATION")

        """
        node_dict = defaultdict(list)
        
        for i in range(self.num_layers):
            for key, value in out_dict.items():
                fc = self.fcs[i][key]
                out = self.act(fc(value))
                node_dict[key].append(out)

        for key, value in node_dict.items():
            val = group_mean(value)
            node_dict[key] = self.act(self.lin(val))


        return node_dict
        """
        return out_dict


class ACOPFEnforcer(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, act_fn: str = "elu"
                 , norm: Union[str, Callable, None] = None, init_data=None):
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

        self.norms = None
        self.meta_data = None
        self.aggr = "sum"
        self.heads = 1
        self.in_channels = 4
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act_fn = act_fn
        self.act = activation_resolver(act_fn, **({}))
        self.act_first = False
        self.jk_mode = "last"
        in_channels = hidden_channels * self.heads
        self.lin_2 = Linear(-1, self.out_channels, bias=True)
        self.node_types = ["SB", "PQ", "PV", "NB"]

        # if norm is not None:
        # norm_layer = HeteroLayerNorm(

        # """normalization_resolver(
        #   norm,
        #   hidden_channels,
        #   **({}),
        # )"""

        # self.norms = ModuleList()

        # for _ in range(num_layers):
        # for _ in self.edge_types:
        # self.norms.append(copy.deepcopy(norm_layer))


        self.fcs = ModuleList()

        for _ in range(self.num_layers):
            fc = ModuleDict()
            for node_type in self.node_types:
                fc[node_type] = Linear(-1, self.out_channels, bias=True)
            self.fcs.append(fc)

    def lazy_init(self, lazy_input, net, index_mappers):
        x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, scaler, _, _ = hp.extract_unsupervised_inputs(
            lazy_input, net, index_mappers)
        self.node_types = list(set(self.node_types).intersection(set(x_dict.keys())))
        self.init_layer_norm(x_dict)
        output = self.forward(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, scaler, check_nans=True)
        print(output)

    def set_meta_data(self, x_dict):
        meta_data = dict()

        for node_type in x_dict:
            meta_data[node_type] = len(x_dict[node_type])

        self.meta_data = meta_data

    def init_layer_norm(self, x_dict):
        self.set_meta_data(x_dict)
        self.norms = ModuleList()
        norm_layer = HeteroMinMaxNorm()

        # for _ in range(self.num_layers):
        # self.norms.append(copy.deepcopy(norm_layer))
        self.norms.append(norm_layer)

    def forward(self, _x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, scaler: StandardScaler,
                check_nans: bool = False):

        x_dict = dict()

        for node_type in _x_dict:
            x = _x_dict[node_type]
            c = constraint_dict[node_type]
            x_dict[node_type] = torch.cat((x, c), dim=1)

        torch.autograd.set_detect_anomaly(True)
        if check_nans:
            for node_type in x_dict:
                x = x_dict[node_type]
                if torch.isnan(x).any():
                    print(f"nan found in node type {node_type} of x_dict")

        # All calls of the forward function must support edge_attr & edge_idx for this model

        node_dict = defaultdict(list)

        for i in range(self.num_layers):
            for key, value in x_dict.items():
                fc = self.fcs[i][key]
                out = self.act(fc(value))
                node_dict[key].append(out)

        # Calculate the Aggregations
        for key, value in node_dict.items():
            val = group(list(value), self.aggr)
            node_dict[key] = self.act(self.lin_2(val))

            if check_nans and torch.isnan(node_dict[key]).any():
                print(f"NAN FOUND in AGGREGATION")

        """
        final_dict = defaultdict(Tensor)

        # Enforce the constraints
        for node_type, features in node_dict.items():
            # Store the constraints
            V_min = constraint_dict[node_type][:, 0].view(-1, 1)
            V_max = constraint_dict[node_type][:, 1].view(-1, 1)
            P_min = constraint_dict[node_type][:, 3].view(-1, 1)
            P_max = constraint_dict[node_type][:, 4].view(-1, 1)
            Q_min = constraint_dict[node_type][:, 5].view(-1, 1)
            Q_max = constraint_dict[node_type][:, 6].view(-1, 1)

            # Store the outputs
            V_node = features[:, 0].view(-1, 1)
            delta_node = features[:, 1].view(-1, 1)
            P_node = features[:, 2].view(-1, 1)
            Q_node = features[:, 3].view(-1, 1)
            #delta_mean = scaler.mean_[1]
            #delta_std = scaler.scale_[1]
            #delta_sum_target = -delta_mean / delta_std

            lower_V = V_min + torch.nn.functional.relu(V_node - V_min)
            constrained_V = V_max - torch.nn.functional.relu(V_max - lower_V)
            #constrained_delta = torch.nn.functional.relu(delta_node)
            #constrained_V = custom_tanh(V_node, V_min, V_max)

            if node_type != "NB":
                lower_P = P_min + torch.nn.functional.relu(P_node - P_min)
                constrained_P = P_max - torch.nn.functional.relu(P_max - lower_P)

            #constrained_P = custom_tanh(P_node, P_min, P_max)

                lower_Q = Q_min + torch.nn.functional.relu(Q_node - Q_min)
                constrained_Q = Q_max - torch.nn.functional.relu(Q_max - lower_Q)
            #constrained_Q = custom_tanh(Q_node, Q_min, Q_max)
            else:
                P_mean = scaler.mean_[2]
                P_std = scaler.scale_[2]
                P_sum_target = -P_mean / P_std
                Q_mean = scaler.mean_[3]
                Q_std = scaler.scale_[3]
                Q_sum_target = -Q_mean / Q_std
                constrained_P = torch.tensor([P_sum_target]*P_node.size(0), dtype=torch.float32).view(-1,1)
                constrained_Q = torch.tensor([Q_sum_target]*Q_node.size(0),dtype=torch.float32).view(-1,1)

            enforced_features = torch.cat([constrained_V, delta_node, constrained_P, constrained_Q], dim=1)
            final_dict[node_type] = enforced_features

        return final_dict
        """
        return node_dict


class HeteroMinMaxNorm(nn.Module):
    def __init__(self, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # We need to keep track of the running min and max across all node types
        self.register_buffer('running_min', torch.tensor(0.0))
        self.register_buffer('running_max', torch.tensor(1.0))

    def forward(self, x_dict, node_types):
        # Extract node feature tensors and concatenate them
        x_concat = torch.cat([x_dict[key] for key in node_types], dim=0)

        # Compute the global min and max values for the current batch
        batch_min = x_concat.min()
        batch_max = x_concat.max()

        # Update the running min and max (detach to avoid backward pass issues)
        with torch.no_grad():
            self.running_min = (1.0 - self.momentum) * self.running_min + self.momentum * batch_min
            self.running_max = (1.0 - self.momentum) * self.running_max + self.momentum * batch_max

        # Scale the concatenated tensor values to [-1, 1]
        x_scaled = (x_concat - self.running_min) / (self.running_max - self.running_min + self.eps) * 2 - 1

        # Split the scaled tensor back into a dictionary format
        split_sizes = [x_dict[key].size(0) for key in node_types]
        x_split = torch.split(x_scaled, split_sizes, dim=0)

        out_dict = {key: x_split[idx] for idx, key in enumerate(node_types)}
        return out_dict


class HeteroBatchNorm(torch.nn.Module):
    r"""Applies batch normalization over a batch of heterogeneous features as
    described in the `"Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper.
    Compared to :class:`BatchNorm`, :class:`HeteroBatchNorm` applies
    normalization individually for each node or edge type.

    Args:
        in_channels (int): Size of each input sample.
        num_types (int): The number of types.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
    """

    def __init__(
            self,
            in_channels: int,
            num_types: int,
            eps: float = 1e-5,
            momentum: Optional[float] = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_types, in_channels))
            self.bias = torch.nn.Parameter(torch.empty(num_types, in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean',
                                 torch.empty(num_types, in_channels))
            self.register_buffer('running_var',
                                 torch.empty(num_types, in_channels))
            self.register_buffer('num_batches_tracked', torch.tensor(0))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.mean_var = FusedAggregation(['mean', 'var'])

        self.reset_parameters()

    def reset_running_stats(self):
        r"""Resets all running statistics of the module."""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward_(self, x: Tensor, type_vec: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input features.
            type_vec (torch.Tensor): A vector that maps each entry to a type.
        """
        if not self.training and self.track_running_stats:
            mean, var = self.running_mean, self.running_var
        else:
            with torch.no_grad():
                mean, var = self.mean_var(x, type_vec, dim_size=self.num_types)

        if self.training and self.track_running_stats:
            if self.momentum is None:
                self.num_batches_tracked.add_(1)
                exp_avg_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exp_avg_factor = self.momentum

            with torch.no_grad():  # Update running mean and variance:
                type_index = torch.unique(type_vec)

                self.running_mean[type_index] = (
                        (1.0 - exp_avg_factor) * self.running_mean[type_index] +
                        exp_avg_factor * mean[type_index])
                self.running_var[type_index] = (
                        (1.0 - exp_avg_factor) * self.running_var[type_index] +
                        exp_avg_factor * var[type_index])

        out = (x - mean[type_vec]) / var.clamp(self.eps).sqrt()[type_vec]

        if self.affine:
            out = out * self.weight[type_vec] + self.bias[type_vec]

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'num_types={self.num_types})')

    def forward(self, x_dict: dict) -> Tensor:
        # Extract node feature tensors and concatenate them
        keys = ["SB", "PQ", "PV", "NB"]

        x_concat = torch.cat([x_dict[key] for key in keys], dim=0)

        # Create type_vec
        type_vec = torch.cat([
            torch.full((x_dict["SB"].size(0),), 0, dtype=torch.long),
            torch.full((x_dict["PQ"].size(0),), 1, dtype=torch.long),
            torch.full((x_dict["PV"].size(0),), 2, dtype=torch.long),
            torch.full((x_dict["NB"].size(0),), 3, dtype=torch.long)
        ])

        # Apply the original forward logic using x_concat and type_vec
        out = self.forward_(x_concat, type_vec)

        # Split the output tensor back into a dictionary format
        split_sizes = [x_dict[key].size(0) for key in keys]
        out_list = torch.split(out, split_sizes, dim=0)
        out_dict = {"SB": out_list[0], "PQ": out_list[1], "PV": out_list[2], "NB": out_list[3]}

        return out_dict

