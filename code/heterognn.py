from torch_geometric.nn.norm import BatchNorm
from gnn import *


# Define the GNN layer
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv


def custom_sigmoid(x, lower_bound, upper_bound, slope=0.1):
    # Scale and shift the sigmoid function
    scaled_sigmoid = 1 / (1 + torch.exp(-slope * (x - (lower_bound + upper_bound) / 2)))

    # Scale and shift the result to fit within the specified lower_bound and upper_bound
    scaled_sigmoid = (scaled_sigmoid - 0.5) * (upper_bound - lower_bound) + (lower_bound + upper_bound) / 2

    return scaled_sigmoid


class ACOPFGNN(torch.nn.Module):
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
        self.act_fn = act_fn
        self.act = activation_resolver(act_fn, **({}))
        self.act_first = False
        self.jk_mode = "last"
        in_channels = hidden_channels
        self.lin = Linear(in_channels, self.out_channels, bias=False)
        self.bias = nn.Parameter(torch.tensor(self.out_channels, dtype=torch.float32))

        self.norms = None
        if norm is not None:
            norm_layer = BatchNorm(in_channels=self.hidden_channels)

            """normalization_resolver(
                norm,
                hidden_channels,
                **({}),
            )"""

            self.norms = ModuleList()

            for _ in range(num_layers):
                for _ in ["SB", "PV", "PQ", "NB"]:
                    self.norms.append(copy.deepcopy(norm_layer))


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


    def forward(self, x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, check_nans:bool = False):
        out_dict = x_dict.copy()

        # All calls of the forward function must support edge_attr & edge_idx for this model
        for conv in self.convs.children():
            out_dict = conv(out_dict, edge_idx_dict, edge_attr_dict)
            #print(f"Nan Value present in out_dict?: {torch.isnan(out_dict).any()}")

            # Apply activation function for the output features of each type and update the dict
            for i, node_type in enumerate(out_dict.keys()):
                if self.norms is not None:
                    out_dict[node_type] = self.norms[i](out_dict[node_type])
                out_dict[node_type] = self.act(out_dict[node_type])
                #print(f"Nan Value present in out_dict after activation fn applied?: {torch.isnan(out_dict).any()}")

        for node_type in out_dict:
            out_dict[node_type] = self.lin(out_dict[node_type])
            #print(f"Nan Value present in out_dict after linear operations?: {torch.isnan(out_dict).any()}")

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

                if from_bus != "NB":

                    for pair in bus_idx_neighbors_dict[from_bus][bus_idx]:
                        # For each neighbor of the iterated bus
                        to_bus, to_bus_idx, edge_attr = pair

                        V_j = abs(out_dict[to_bus][to_bus_idx][0])

                        volt_angle_j = out_dict[to_bus][to_bus_idx][1]
                        delta_ij = volt_angle_j - volt_angle_i

                        G_ij = edge_attr[0] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)
                        if check_nans and torch.isnan(G_ij).any():
                            print("Nan Value present in G_ij")

                        B_ij = -edge_attr[1] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)
                        if check_nans and torch.isnan(B_ij).any():
                            print(f"Nan Value present in B_ij")

                        # ACOPF Equation for P_i
                        P_ij = V_i * V_j * (G_ij * torch.cos(delta_ij) + B_ij * torch.sin(delta_ij))
                        if check_nans and torch.isnan(P_ij).any():
                            print(f"Nan Value present in P_ij?")
                        P_i += P_ij

                        # ACOPF Equation for Q_i
                        Q_ij = V_i * V_j * (G_ij * torch.sin(delta_ij) - B_ij * torch.cos(delta_ij))
                        if check_nans and torch.isnan(Q_ij).any():
                            print(f"Nan Value present in Q_ij")
                        Q_i += Q_ij

                    # Enforce the constraints
                    # Active Power boundary constraint
                    P_i = custom_sigmoid(P_i, active_pow_lower_bound, active_pow_upper_bound)
                    if check_nans and torch.isnan(P_i).any():
                        print(f"Nan Value present in P_i after sigmoid")
                    # Reactive Power boundary constraint
                    Q_i = custom_sigmoid(Q_i, reactive_pow_lower_bound, reactive_pow_upper_bound)

                    if check_nans and torch.isnan(Q_i).any():
                        print(f"Nan Value present in Q_i after sigmoid")

                    # Maximum Apparent Power S constraint
                    constrained_P_i_upper_bound = torch.sqrt(torch.square(max_apparent_pow) - torch.square(Q_i))
                    learnable_P_i_constraint = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
                    P_i = custom_sigmoid(P_i, min(active_pow_lower_bound, constrained_P_i_upper_bound), max(active_pow_lower_bound, constrained_P_i_upper_bound),slope=learnable_P_i_constraint)

                    if check_nans and torch.isnan(P_i).any():
                        print(f"Nan Value present in P_i after S enforcement")

                    constrained_Q_i_upper_bound = torch.sqrt(torch.square(max_apparent_pow) - torch.square(P_i))
                    learnable_Q_i_constraint = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
                    Q_i = custom_sigmoid(Q_i, min(reactive_pow_lower_bound, constrained_Q_i_upper_bound), max(reactive_pow_lower_bound, constrained_Q_i_upper_bound),slope=learnable_Q_i_constraint)
                    if check_nans and torch.isnan(Q_i).any():
                        print(f"Nan Value present in Q_i after S enforcement")

                # Enforce the constraints
                # Voltage Magnitude boundary constraint
                learnable_V_i_constraint = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
                out_dict[from_bus][bus_idx][0] = custom_sigmoid(V_i, volt_mag_lower_bound, volt_mag_upper_bound,slope=learnable_V_i_constraint)#/x_dict[from_bus][bus_idx][0]

                # Voltage Angle constraint
                out_dict[from_bus][bus_idx][1] = torch.relu(volt_angle_i) if from_bus != "SB" else torch.relu(volt_angle_i) - out_dict["SB"][0][1] # voltage angle >= 0

                out_dict[from_bus][bus_idx][2] = P_i
                out_dict[from_bus][bus_idx][3] = Q_i

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

    def forward(self, x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict):
        # Perform GNN propagation on input data 'x' to compute node_features
        out_dict = self.gnn(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict)

        return out_dict

    def build_acopf_problem(self, out_dict):
        return NotImplementedError"""

        
