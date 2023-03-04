from gnn import *

# Define the GNN layer
class HeteroGNN(nn.Module):
    def __init__(self, input_dim, output_dim, node_types, num_nodes, adjacency_matrix, num_layers=1,
                 activation=nn.ReLU()):
        super().__init__()
        self.node_types = node_types
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList([nn.GCNConv(input_dim, output_dim) for _ in range(num_layers)])
        self.activation = activation
        self.adjacency_matrix = adjacency_matrix

    def compute_node_embeddings(self, x):
        # Create a list to store the node embeddings
        node_embeddings = []
        adjacency_matrix = self.adjacency_matrix

        # Iterate over the node types
        for node_type in self.node_types:
            # Get the number of nodes of the current type
            n = self.num_nodes[node_type]
            # Compute the node embeddings for the current node type
            node_embeddings.append(torch.sum(x[:n] * adjacency_matrix[:n, :n], dim=1))
            # Remove the nodes of the current type from the node features and adjacency matrix
            x = x[n:]
            adjacency_matrix = adjacency_matrix[n:, n:]

        # Concatenate the node embeddings of all node types
        node_embeddings = torch.cat(node_embeddings, dim=0)

        return node_embeddings

    def forward(self, x):
        # Compute the node embeddings using the adjacency matrix and node features
        node_embeddings = self.compute_node_embeddings(self, x)

        # Apply the GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            node_embeddings = gnn_layer(node_embeddings, self.adjacency_matrix)

            if i < self.num_layers - 1:
                node_embeddings = self.activation(node_embeddings)

        return node_embeddings

class GNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, node_types, num_nodes, adjacency_matrix, num_layers=1, activation=nn.ReLU()):
        super().__init__()
        self.gnn = HeteroGNN(input_dim, output_dim, node_types, num_nodes, adjacency_matrix, num_layers, activation)
        self.fc = nn.Linear(output_dim * len(node_types), output_dim * len(node_types))

    def forward(self, x):
        x = self.gnn(x, self.adjacency_matrix)
        x = self.fc(x)

        # Enforce the constraint that the voltage angle of the slack node should be 0.0
        x[0, 1] = 0.0


        # Enforce the constraint that the active power of generator nodes should be negative
        x[1:num_nodes['Generator Node']+1, 2] = -x[1:num_nodes['Generator Node']+1, 2]

        # Enforce the constraint that the active power of load nodes should be positive
        x[num_nodes['Generator Node']+1:, 2] = x[num_nodes['Generator Node']+1:, 2]

        return x