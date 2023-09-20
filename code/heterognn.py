import torch_geometric.nn.norm.layer_norm
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
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


def custom_sigmoid(x, lower_bound, upper_bound, slope=0.1):
    # Scale and shift the sigmoid function
    scaled_sigmoid = 1 / (1 + torch.exp(-slope * (x - (lower_bound + upper_bound) / 2)))

    # Scale and shift the result to fit within the specified lower_bound and upper_bound
    scaled_sigmoid = (scaled_sigmoid - 0.5) * (upper_bound - lower_bound) + (lower_bound + upper_bound) / 2

    return scaled_sigmoid


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

        self.in_channels = 4
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

    def forward(self, x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict):
        out_dict = x_dict.copy()
        # All calls of the forward function must support edge_attr & edge_idx for this model
        for conv in self.convs.children():
            out_dict = conv(out_dict, edge_idx_dict, edge_attr_dict)

            # Apply activation function for the output features of each type and update the dict
            for node_type in out_dict:
                out_dict[node_type] = self.act(out_dict[node_type])

        for node_type in out_dict:
            out_dict[node_type] = self.lin(out_dict[node_type])

        # ACOPF Forward Pass for P and Q
        for from_bus in bus_idx_neighbors_dict:
            for bus_idx in bus_idx_neighbors_dict[from_bus]:
                # Store the constraints
                volt_mag_lower_bound = constraint_dict[from_bus][bus_idx][0]
                volt_mag_upper_bound = constraint_dict[from_bus][bus_idx][1]
                max_apparent_pow = constraint_dict[from_bus][bus_idx][2]
                active_pow_lower_bound= constraint_dict[from_bus][bus_idx][3]
                active_pow_upper_bound = constraint_dict[from_bus][bus_idx][4]
                reactive_pow_lower_bound = constraint_dict[from_bus][bus_idx][5]
                reactive_pow_upper_bound = constraint_dict[from_bus][bus_idx][6]

                V_i = abs(out_dict[from_bus][bus_idx][0])
                volt_angle_i = out_dict[from_bus][bus_idx][1]
                P_i = torch.tensor(0.0, dtype=torch.float32)
                Q_i = torch.tensor(0.0, dtype=torch.float32)

                for pair in bus_idx_neighbors_dict[from_bus][bus_idx]:
                    # For each neighbor of the iterated bus
                    to_bus, to_bus_idx, edge_attr = pair

                    V_j = abs(out_dict[to_bus][to_bus_idx][0])

                    volt_angle_j = out_dict[to_bus][to_bus_idx][1]
                    delta_ij = volt_angle_j - volt_angle_i

                    G_ij = edge_attr[0] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)
                    B_ij = -edge_attr[1] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)

                    # ACOPF Equation for P_i
                    P_ij = V_i * V_j * (G_ij * torch.cos(delta_ij) + B_ij * torch.sin(delta_ij))

                    P_i += P_ij

                    # ACOPF Equation for Q_i
                    Q_ij = V_i * V_j * (G_ij * torch.sin(delta_ij) - B_ij * torch.cos(delta_ij))

                    Q_i += Q_ij

                # Enforce the constraints
                out_dict[from_bus][bus_idx][0] = custom_sigmoid(V_i, volt_mag_lower_bound, volt_mag_upper_bound)/x_dict[from_bus][bus_idx][0]
                out_dict[from_bus][bus_idx][2] = custom_sigmoid(P_i, active_pow_lower_bound, active_pow_upper_bound)
                out_dict[from_bus][bus_idx][3] = custom_sigmoid(Q_i, reactive_pow_lower_bound, reactive_pow_upper_bound)

        return out_dict


"""
# ACOPFMODEL WITH CONVEX OPTIMIZATION LAYER
class ACOPFModel(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, act_fn: str
                 , norm: Union[str, Callable, None] = None):
        super(ACOPFModel, self).__init__()
        self.gnn = HeteroGNN(self, hidden_channels, out_channels, num_layers, dropout, act_fn, norm)
        self.cvxpy_layer = CvxpyLayer(
            self.build_acopf_problem,  # Pass the problem builder function
            input_vars=[x],  # Input variables
            output_vars=[objective, *constraints]  # Output variables
        )

    def forward(self, x):
        # Perform GNN propagation on input data 'x' to compute node_features
        node_features = self.gnn(x)

        # Call the CvxpyLayer to solve the ACOPF problem
        objective, constraints = self.cvxpy_layer(node_features)

        return objective, constraints

    def build_acopf_problem(self, x_dict: dict):
        num_nodes = 0
        min_voltage_magnitudes = []
        max_voltage_magnitudes = []
        max_apparent_power = []
        min_active_powers = []
        max_active_powers = []
        min_reactive_powers = []
        max_reactive_powers = []

        for key in x_dict:
            x = x_dict[key]
            num_nodes += x.shape[0]

            min_voltage_magnitudes_node = list(x[:, 4])
            max_voltage_magnitudes_node = list(x[:, 5])
            max_apparent_power_node = list(x[:, 6])
            min_active_powers_node = list(x[:, 7])
            max_active_powers_node = list(x[:, 8])
            min_reactive_powers_node = list(x[:, 9])
            max_reactive_powers_node = list(x[:, 10])

            min_voltage_magnitudes.extend(min_voltage_magnitudes_node)
            max_voltage_magnitudes.extend(max_voltage_magnitudes_node)
            max_apparent_power.extend(max_apparent_power_node)
            min_active_powers.extend(min_active_powers_node)
            max_active_powers.extend(max_active_powers_node)
            min_reactive_powers.extend(min_reactive_powers_node)
            max_reactive_powers.extend(max_reactive_powers_node)


        # Define optimization variables
        voltage_magnitudes = cp.Variable(num_nodes)
        active_powers = cp.Variable(num_nodes)
        reactive_powers = cp.Variable(num_nodes)

        # Define objective function (example, you can modify this)
        objective = cp.Minimize(cp.sum_squares(active_powers) + cp.sum_squares(reactive_powers))

        # Define constraint expressions
        constraints = [
            voltage_magnitudes >= min_voltage_magnitudes,
            voltage_magnitudes <= max_voltage_magnitudes,
            # Add other constraints here...
            active_powers >= min_active_powers,
            active_powers <= max_active_powers,
            reactive_powers >= min_reactive_powers,
            reactive_powers <= max_reactive_powers,
            # Add other constraints here...
        ]
        return objective, constraints

    def custom_sigmoid(x, lower_bound, upper_bound, slope=0.1):
        # Scale and shift the sigmoid function
        scaled_sigmoid = 1 / (1 + torch.exp(-slope * (x - (lower_bound + upper_bound) / 2)))

        # Scale and shift the result to fit within the specified lower_bound and upper_bound
        scaled_sigmoid = (scaled_sigmoid - 0.5) * (upper_bound - lower_bound) + (lower_bound + upper_bound) / 2

        return scaled_sigmoid
        
"""