# OPFGNN
Master Thesis: Optimal Power in Power Systems using Graph Neural Networks

## Abstract
This thesis investigates the Optimal Power Flow (OPF) problem in electrical networks,
specifically focusing on the Alternating Current Optimal Power Flow (ACOPF) scenario.
It explores the application of both Homogeneous and Heterogeneous Graph Neural
Networks (GNNs) to optimize power generation and distribution, thereby enhancing
network efficiency with a focus on network bus features. Motivated by the imperative
of efficient energy management, the performance of these models is evaluated and
compared against traditional numerical solvers. The research demonstrates the efficacy
of GNNs in solving real-world power system challenges, offering valuable insights into
their application in the energy optimization landscape. This work significantly contributes to the discourse on energy optimization techniques, emphasizing the feasibility
and efficiency of GNNs in addressing intricate power system problems.

## Best Performing Models
  - Homogeneous GNN (HGNN)
  - Node-Type Based Heterogeneous GNN (NHGNN)

## Test Results
### HGNN: 
- RMSE: 0.002404558245325461
- MAE: 0.0015261667082086205
- MRE: 0.020806053886190057
### NHGNN: 
- RMSE: 0.006351334974169731
- MAE: 0.0020139727275818586
- MRE: 0.04707365483045578
