OVERWIEW:

    FILES:
        -gnn.ipynb: main file for all operations
        -gnn.py: File including the homogeneous GNN Class
        -ACOPFData.py: Helper Class for organizing heterogeneous Data
        -graphdata.py: Helper Class for organizing homogeneous Data
        -helper.py: Helper library file including all methods used for processing, generation and modification
        -heterognn.py: File including all heterogeneous GNNs and other classes
        -main.ipynb: File used for evaluating and displaying models` results (Main Focus for Advisor/Supervisor)

    Folders:
        -data: Folder containing all homogeneous Datasets as CSV and heterogeneous Test Datasets as pickle (.pkl) files
            In terms of the heterogeneous datasets, only the test datasets are included in order to comply with the maximum allowed data size limit for submission
        -Models: Folder containing all Models that have been subject to the training and evaluation within the scope of this research.
            * hetero_model_bus_constrained.pt: Node Type-Based Hetero GNN with embedded Constraint Enforcement
            * hetero_model_bus_final.pt: Base Node Type-Based Hetero GNN
            * hetero_model_edge_without_constraints.pt: Edge Type-Based Hetero GNN without constraints
            * hetero_model_edge.pt: Edge Type-Based Hetero GNN with constraints
            * supervisedmodel.pt: Homogeneous GNN