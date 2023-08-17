import pandapower.topology as pptop
import simbench
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torchvision
from torch_geometric.data import Data, HeteroData
import sklearn.metrics as metrics
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import pandapower as pp
import networkx as nx
import wandb
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, MaxAbsScaler, QuantileTransformer
import simbench as sb
import os
import random
from graphdata import GraphData
from typing import Tuple
import torch_geometric.transforms as T
import json

def sample_uniform_from_df(load):
    # Create Datasets from sampling reference active power and reactive power demands of loads uniformly
    # load - Nx17 Dataframe with N = number of loads in the network (net.load)

    p_mw = []
    q_mvar = []

    for i in range(len(load)):
        p_ref = load.iloc[i][2]
        q_ref = load.iloc[i][3]
        p_L = random.uniform(0.9 * p_ref, 1.1 * p_ref)  # uniformly sample p_ref
        q_L = random.uniform(0.9 * q_ref, 1.1 * q_ref)  # uniformly sample q_ref
        p_mw.append(p_L)
        q_mvar.append(q_L)

    load["p_mw"] = p_mw
    load["q_mvar"] = q_mvar
    return load


def create_dataset_from_dcopf_and_acopf(net):
    # Creates Dataset from the combination of initial bus voltages and loads
    # with DCOPP Solution of voltage angles and active powers#

    net.load = sample_uniform_from_df(net.load)  # Sample reference loads uniformly

    # OPERATIONAL CONSTRAINTS

    # Set upper and lower limits of active-reactive powers of loads
    min_p_mw_val, max_p_mw_val, min_q_mvar_val, max_q_mvar_val = [], [], [], []
    p_mw = list(net.load.p_mw.values)
    q_mvar = list(net.load.q_mvar.values)

    for i in range(len(p_mw)):
        min_p_mw_val.append(p_mw[i])
        max_p_mw_val.append(p_mw[i])
        min_q_mvar_val.append(q_mvar[i])
        max_q_mvar_val.append(q_mvar[i])

    net.load.min_p_mw = min_p_mw_val
    net.load.max_p_mw = max_p_mw_val
    net.load.min_q_mvar = min_q_mvar_val
    net.load.max_q_mvar = max_q_mvar_val

    # Replace all ext_grids but the first one with generators and set the generators to slack= false
    ext_grids = [i for i in range(1, len(net.ext_grid.name.values))]
    pp.replace_ext_grid_by_gen(net, ext_grids=ext_grids, slack=False)

    # TODO: reactive power limits for gens?

    # TODO: reactive power limits for sgens?

    # NETWORK CONSTRAINTS

    # Maximize the branch limits

    # max_i_ka = list(net.line.max_i_ka.values)

    # for i in range(len(max_i_ka)):
    # max_i_ka[i] = max(max_i_ka)

    # Maximize line loading percents
    max_loading_percent = list(net.line.max_loading_percent.values)
    for i in range(len(max_loading_percent)):
        max_loading_percent[i] = 100.0
    net.line.max_loading_percent = max_loading_percent

    # Maximize trafo loading percent
    max_loading_percent = list(net.trafo.max_loading_percent.values)
    for i in range(len(max_loading_percent)):
        max_loading_percent[i] = 100.0
    net.trafo.max_loading_percent = max_loading_percent

    # Maximize trafo3w loading percent
    max_loading_percent = list(net.trafo3w.max_loading_percent.values)
    for i in range(len(max_loading_percent)):
        max_loading_percent[i] = 100.0
    net.trafo3w.max_loading_percent = max_loading_percent

    # Cost assignment
    pp.create_pwl_costs(net, [i for i in range(len(net.gen.name.values))], et="gen",
                        points=[[[0, 20, 1], [20, 30, 2]] for _ in range(len(net.gen.name.values))])
    pp.create_pwl_costs(net, [i for i in range(len(net.sgen.name.values))], et="sgen",
                        points=[[[0, 20, 0.25], [20, 30, 0.5]] for _ in range(len(net.sgen.name.values))])
    pp.create_pwl_costs(net, [i for i in range(len(net.ext_grid.name.values))], et="ext_grid",
                        points=[[[0, 20, 2], [20, 30, 5]] for _ in range(len(net.ext_grid.name.values))])

    try:
        pp.runpm_dc_opf(net) # Run DCOPP-Julia
    except pp.OPFNotConverged:
        #print("DC OPTIMAL POWERFLOW COMPUTATION DID NOT CONVERGE. SKIPPING THIS DATASET.")
        return None

    df = pd.DataFrame(net.res_bus)  # Store the resulting busses in a dataframe

    df = df.drop(labels=['lam_p', 'lam_q'], axis=1) # Drop columns

    for i in range(len(df)):
        df.iloc[i][0] = 1.0  # net.bus.iloc[i][1]  # Set the initial bus voltages

    for i in range(len(net.load)):
        idx_bus = net.load.iloc[i][1]  # Get the bus from the net.load dataframe

        for j in range(len(df.index)):  # Find the index of df with linear search where index = idx_bus
            if df.index[j] == idx_bus:
                idx_bus = j
                break

        df.iloc[idx_bus][3] = net.load.iloc[i][3]  # Set the q_mvar at idx_bus to that given in net.load

    df['q_mvar'] = df['q_mvar'].fillna(0)  # Drop all nAn and replace them with 0

    for init in ["pf", "flat", "results"]:
        try:
            pp.runopp(net, init=init)  # Calculate ACOPF with IPFOPT
        except pp.OPFNotConverged:
            #print("OPTIMAL POWERFLOW COMPUTATION DID NOT CONVERGE FOR START VECTOR CONFIG " + init)
            if init == "results":
                #print("SKIPPING THIS DATASET.")
                return None
            continue
        break

    net.res_bus = net.res_bus.drop(labels=['lam_p', 'lam_q'], axis=1)  # Drop columns

    # Concatenate augmented results from DCOPF (X - N x 4) with the augmented results from ACOPF (Y - N x 4)
    df = pd.concat([df, net.res_bus], axis=0)

    return df  # Resulting Dataset is 2N x 4 with 0-N x 4 being X and N+1-2N x 4 being Y

def create_dataset_from_dcopf(net):
    # Creates Dataset from the combination of initial bus voltages and loads
    # with DCOPP Solution of voltage angles and active powers#

    net.load = sample_uniform_from_df(net.load)  # Sample reference loads uniformly

    # OPERATIONAL CONSTRAINTS

    # Set upper and lower limits of active-reactive powers of loads
    min_p_mw_val, max_p_mw_val, min_q_mvar_val, max_q_mvar_val = [], [], [], []
    p_mw = list(net.load.p_mw.values)
    q_mvar = list(net.load.q_mvar.values)

    for i in range(len(p_mw)):
        min_p_mw_val.append(p_mw[i])
        max_p_mw_val.append(p_mw[i])
        min_q_mvar_val.append(q_mvar[i])
        max_q_mvar_val.append(q_mvar[i])

    net.load.min_p_mw = min_p_mw_val
    net.load.max_p_mw = max_p_mw_val
    net.load.min_q_mvar = min_q_mvar_val
    net.load.max_q_mvar = max_q_mvar_val

    # Replace all ext_grids but the first one with generators and set the generators to slack= false
    ext_grids = [i for i in range(1, len(net.ext_grid.name.values))]
    pp.replace_ext_grid_by_gen(net, ext_grids=ext_grids, slack=False)

    # TODO: reactive power limits for gens?

    # TODO: reactive power limits for sgens?

    # NETWORK CONSTRAINTS

    # Maximize the branch limits

    # max_i_ka = list(net.line.max_i_ka.values)

    # for i in range(len(max_i_ka)):
    # max_i_ka[i] = max(max_i_ka)

    # Maximize line loading percents
    max_loading_percent = list(net.line.max_loading_percent.values)
    for i in range(len(max_loading_percent)):
        max_loading_percent[i] = 100.0
    net.line.max_loading_percent = max_loading_percent

    # Maximize trafo loading percent
    max_loading_percent = list(net.trafo.max_loading_percent.values)
    for i in range(len(max_loading_percent)):
        max_loading_percent[i] = 100.0
    net.trafo.max_loading_percent = max_loading_percent

    # Maximize trafo3w loading percent
    max_loading_percent = list(net.trafo3w.max_loading_percent.values)
    for i in range(len(max_loading_percent)):
        max_loading_percent[i] = 100.0
    net.trafo3w.max_loading_percent = max_loading_percent

    # Cost assignment
    pp.create_pwl_costs(net, [i for i in range(len(net.gen.name.values))], et="gen",
                        points=[[[0, 20, 1], [20, 30, 2]] for _ in range(len(net.gen.name.values))])
    pp.create_pwl_costs(net, [i for i in range(len(net.sgen.name.values))], et="sgen",
                        points=[[[0, 20, 0.25], [20, 30, 0.5]] for _ in range(len(net.sgen.name.values))])
    pp.create_pwl_costs(net, [i for i in range(len(net.ext_grid.name.values))], et="ext_grid",
                        points=[[[0, 20, 2], [20, 30, 5]] for _ in range(len(net.ext_grid.name.values))])

    try:
        pp.runpm_dc_opf(net)  # Run DCOPP-Julia
    except pp.OPFNotConverged:
        # print("DC OPTIMAL POWERFLOW COMPUTATION DID NOT CONVERGE. SKIPPING THIS DATASET.")
        return None

    df = pd.DataFrame(net.res_bus)  # Store the resulting busses in a dataframe

    df = df.drop(labels=['lam_p', 'lam_q'], axis=1)  # Drop columns

    for i in range(len(df)):
        df.iloc[i][0] = 1.0  # net.bus.iloc[i][1]  # Set the initial bus voltages

    for i in range(len(net.load)):
        idx_bus = net.load.iloc[i][1]  # Get the bus from the net.load dataframe

        for j in range(len(df.index)):  # Find the index of df with linear search where index = idx_bus
            if df.index[j] == idx_bus:
                idx_bus = j
                break

        df.iloc[idx_bus][3] = net.load.iloc[i][3]  # Set the q_mvar at idx_bus to that given in net.load

    df['q_mvar'] = df['q_mvar'].fillna(0)  # Drop all nAn and replace them with 0

    return df


def read_supervised_training_data(grid_name):

    # Define the graph using Pandapower
    net = sb.get_simbench_net(grid_name)

    print("Calculating edge index and edge weights for the grid " + grid_name + " ...")

    # Map indices of busses to ascending correct order
    idx_mapper = dict()
    for idx_given, idx_real in zip(net.bus.index.values, range(len(net.bus.index))):
        idx_mapper[idx_given] = idx_real

    edge_weights = []
    edge_index = [[], []]

    for from_bus, to_bus, r_ohm_per_km, x_ohm_per_km, length_km in zip(net.line.from_bus, net.line.to_bus,
                                                                       net.line.r_ohm_per_km, net.line.x_ohm_per_km,
                                                                       net.line.length_km):
        # Add self loops
        # edge_index[0].append(idx_mapper[from_bus])
        # edge_index[1].append(idx_mapper[from_bus])

        # Add interbus connections
        edge_index[0].append(idx_mapper[from_bus])
        edge_index[1].append(idx_mapper[to_bus])

        # Calculate ||Z_i|| = ||R_i - 1j * X_I||
        norm_Z_i = length_km * np.sqrt(r_ohm_per_km ** 2 + x_ohm_per_km ** 2)
        Y_i = 1 / norm_Z_i  # Y_i = 1 / ||Z_i||
        edge_weights.append(Y_i)

    # Convert edge connections to undirected
    edge_index, edge_weights = to_undirected(edge_index=torch.tensor(edge_index, dtype=torch.int),
                                             edge_attr=torch.tensor(normalize(edge_weights), dtype=torch.float32),
                                             num_nodes=len(net.bus.index))

    # Add self loops to edge connections
    edge_index, edge_weights = add_self_loops(edge_index=edge_index, edge_attr=edge_weights,
                                              num_nodes=len(net.bus.index))

    print("Reading all of the .csv files from the directory of " + grid_name + " ...")

    # Store path to the supervised datasets directory of the specified grid
    train_data, val_data, test_data = [], [], []
    path_to_dir = os.path.dirname(os.path.abspath("gnn.ipynb")) + "\\data\\Supervised\\" + grid_name
    datasets = []

    # Read all the csv files in the directory of the grid_name
    for dataset_name in os.listdir(path_to_dir):
        path2dataset = path_to_dir + "\\" + dataset_name
        datasets.append(pd.read_csv(path2dataset))

    # Process all the data according to  85 - 10 - 5
    random.shuffle(datasets)
    training = datasets[:85]
    validation = datasets[85:95]
    test = datasets[95:]

    print("Processing Training Data for " + grid_name + " ...")
    num_busses = int(len(training[0]) / 2)

    for data in training:
        x = data[:num_busses].drop(columns=["Unnamed: 0"])
        x_rows, x_cols = np.shape(x)
        x = np.array(x).reshape(x_rows, x_cols)
        y = data[num_busses:].drop(columns=["Unnamed: 0"])
        y_rows, y_cols = np.shape(y)
        y = np.array(y).reshape(y_rows, y_cols)
        x = torch.tensor(data=x, dtype=torch.float32)
        y = torch.tensor(data=y, dtype=torch.float32)
        assert (np.shape(x)[0] == np.shape(y)[0] and np.shape(x)[1] == np.shape(y)[1])
        train_data.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weights, num_nodes=num_busses))

    print("Processing Validation Data for " + grid_name + " ...")

    for data in validation:
        x = data[:num_busses].drop(columns=["Unnamed: 0"])
        x_rows, x_cols = np.shape(x)
        x = np.array(x).reshape(x_rows, x_cols)
        y = data[num_busses:].drop(columns=["Unnamed: 0"])
        y_rows, y_cols = np.shape(y)
        y = np.array(y).reshape(y_rows, y_cols)
        x = torch.tensor(data=x, dtype=torch.float32)
        y = torch.tensor(data=y, dtype=torch.float32)
        assert (np.shape(x)[0] == np.shape(y)[0] and np.shape(x)[1] == np.shape(y)[1])
        val_data.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weights, num_nodes=num_busses))

    print("Processing Test Data for " + grid_name + " ...")

    for data in test:
        x = data[:num_busses].drop(columns=["Unnamed: 0"])
        x_rows, x_cols = np.shape(x)
        x = np.array(x).reshape(x_rows, x_cols)
        y = data[num_busses:].drop(columns=["Unnamed: 0"])
        y_rows, y_cols = np.shape(y)
        y = np.array(y).reshape(y_rows, y_cols)
        x = torch.tensor(data=x, dtype=torch.float32)
        y = torch.tensor(data=y, dtype=torch.float32)
        assert (np.shape(x)[0] == np.shape(y)[0] and np.shape(x)[1] == np.shape(y)[1])
        test_data.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weights, num_nodes=num_busses))

    print("Processing complete.")
    return train_data, val_data, test_data, edge_index, edge_weights


def read_supervised_training_data_edge_attr(grid_name):

    # Define the graph using Pandapower
    net = sb.get_simbench_net(grid_name)

    print("Calculating edge index and edge attributes for the grid " + grid_name + " ...")

    # Map indices of busses to ascending correct order
    idx_mapper = dict()
    for idx_given, idx_real in zip(net.bus.index.values, range(len(net.bus.index))):
        idx_mapper[idx_given] = idx_real

    edge_attr = []
    edge_index = [[], []]

    for from_bus, to_bus, r_ohm_per_km, x_ohm_per_km, length_km in zip(net.line.from_bus, net.line.to_bus,
                                                                       net.line.r_ohm_per_km, net.line.x_ohm_per_km,
                                                                       net.line.length_km):
        # Add self loops
        # edge_index[0].append(idx_mapper[from_bus])
        # edge_index[1].append(idx_mapper[from_bus])

        # Add interbus connections
        edge_index[0].append(idx_mapper[from_bus])
        edge_index[1].append(idx_mapper[to_bus])

        # Calculate R_i and X_i
        R_i = length_km * r_ohm_per_km
        X_i = length_km * x_ohm_per_km

        edge_attr.append([R_i, X_i])

    #edge_attr = torch.tensor(StandardScaler().fit_transform(edge_attr), dtype=torch.float32)

    # Convert edge connections to undirected
    edge_index, edge_attr = to_undirected(edge_index=torch.tensor(edge_index, dtype=torch.int),
                                             edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                                             num_nodes=len(net.bus.index))

    # Add self loops to edge connections
    edge_index, edge_attr = torch_geometric.utils.add_self_loops(edge_index=edge_index, edge_attr=edge_attr,
                                              num_nodes=len(net.bus.index))

    print("Reading all of the .csv files from the directory of " + grid_name + " ...")

    # Store path to the supervised datasets directory of the specified grid
    train_data, val_data, test_data = [], [], []
    path_to_dir = os.path.dirname(os.path.abspath("gnn.ipynb")) + "\\data\\Supervised\\" + grid_name
    datasets = []

    # Read all the csv files in the directory of the grid_name
    for dataset_name in os.listdir(path_to_dir):
        path2dataset = path_to_dir + "\\" + dataset_name
        datasets.append(pd.read_csv(path2dataset))

    # Process all the data according to  85 - 10 - 5
    random.shuffle(datasets)
    training = datasets[:85]
    validation = datasets[85:95]
    test = datasets[95:]

    print("Processing Training Data for " + grid_name + " ...")
    num_busses = int(len(training[0]) / 2)

    for data in training:
        x = data[:num_busses].drop(columns=["Unnamed: 0"])
        x_rows, x_cols = np.shape(x)
        x = np.array(x).reshape(x_rows, x_cols)
        y = data[num_busses:].drop(columns=["Unnamed: 0"])
        y_rows, y_cols = np.shape(y)
        y = np.array(y).reshape(y_rows, y_cols)
        x = torch.tensor(data=x, dtype=torch.float32)
        y = torch.tensor(data=y, dtype=torch.float32)
        assert (np.shape(x)[0] == np.shape(y)[0] and np.shape(x)[1] == np.shape(y)[1])
        train_data.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_busses))

    print("Processing Validation Data for " + grid_name + " ...")

    for data in validation:
        x = data[:num_busses].drop(columns=["Unnamed: 0"])
        x_rows, x_cols = np.shape(x)
        x = np.array(x).reshape(x_rows, x_cols)
        y = data[num_busses:].drop(columns=["Unnamed: 0"])
        y_rows, y_cols = np.shape(y)
        y = np.array(y).reshape(y_rows, y_cols)
        x = torch.tensor(data=x, dtype=torch.float32)
        y = torch.tensor(data=y, dtype=torch.float32)
        assert (np.shape(x)[0] == np.shape(y)[0] and np.shape(x)[1] == np.shape(y)[1])
        val_data.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_busses))

    print("Processing Test Data for " + grid_name + " ...")

    for data in test:
        x = data[:num_busses].drop(columns=["Unnamed: 0"])
        x_rows, x_cols = np.shape(x)
        x = np.array(x).reshape(x_rows, x_cols)
        y = data[num_busses:].drop(columns=["Unnamed: 0"])
        y_rows, y_cols = np.shape(y)
        y = np.array(y).reshape(y_rows, y_cols)
        x = torch.tensor(data=x, dtype=torch.float32)
        y = torch.tensor(data=y, dtype=torch.float32)
        assert (np.shape(x)[0] == np.shape(y)[0] and np.shape(x)[1] == np.shape(y)[1])
        test_data.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_busses))

    print("Processing complete.")
    return train_data, val_data, test_data, edge_index, edge_attr


def read_multiple_supervised_datasets(grid_names):
    graphdata_lst = []
    for _ in grid_names:
        train_data, val_data, test_data, edge_index, edge_attr = read_supervised_training_data_edge_attr(_)
        # edge_attr = torch.tensor(StandardScaler().fit_transform(edge_attr), dtype=torch.float32)
        graphdata_lst.append(GraphData(_,train_data, val_data, test_data, edge_index, edge_attr))
    return graphdata_lst

def normalize(lst):
    _sum_ = sum(lst)
    return [float(i) / _sum_ for i in lst]


def train_one_epoch(epoch, grid_name, optimizer, training_loader, model, loss_fn, scaler):

    train_rmse_loss = 0.0
    train_mae_loss = 0.0
    train_mre_loss = 0.0

    # create a criterion to measure the mean absolute error (MAE)
    mae_loss_fn = nn.L1Loss()

    # create a criterion to measure the mean relative error (MRE), inputs_targets: List
    mre_loss_fn = lambda outputs, targets : get_mre_loss(outputs, targets)

    last_idx = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, targets = data.x, data.y

        # Get edge_index and edge_attr from DataLoader
        edge_index = training_loader.dataset[i].edge_index
        edge_attr = training_loader.dataset[i].edge_attr

        # Define Scaler and standardize inputs and targets
        targets = torch.tensor(scaler.fit_transform(targets), dtype=torch.float32)
        inputs = torch.tensor(scaler.transform(inputs), dtype=torch.float32)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs, edge_index, edge_attr=edge_attr)

        # Compute the loss and its gradients for RMSE loss
        rmse_loss = torch.sqrt(loss_fn(outputs, targets))
        rmse_loss.backward()

        # Compute MAE loss
        mae_loss = mae_loss_fn(outputs, targets)
        #mae_loss.backward()

        # Compute MRE loss
        mre_loss = mre_loss_fn(outputs, targets)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        train_rmse_loss += rmse_loss.item()
        train_mae_loss += mae_loss.item()
        train_mre_loss += mre_loss.item()

        last_idx = i + 1

        """
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
        """

    wandb.log({
        'grid name' : grid_name,
        'epoch': epoch,
        'train_rmse_loss': train_rmse_loss/ last_idx,
        'train_mae_loss': train_mae_loss / last_idx,
        'train_mre_loss': train_mre_loss / last_idx

    })


def validate_one_epoch(epoch, grid_name, validation_loader, model, loss_fn, scaler):

    val_rmse_loss = 0.0
    val_mae_loss = 0.0
    val_mre_loss = 0.0

    # create a criterion to measure the mean absolute error
    mae_loss_fn = nn.L1Loss()

    # create a criterion to measure the mean relative error (MRE), outputs, targets : Torch.Tensor
    mre_loss_fn = lambda outputs, targets: get_mre_loss(outputs, targets)

    last_idx = 0

    # Here, we use enumerate(validation_loader) instead of
    # iter(validation_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(validation_loader):
        inputs, targets = data.x, data.y

        # Get edge_index and edge_attr from DataLoader
        edge_index = validation_loader.dataset[i].edge_index
        edge_attr = validation_loader.dataset[i].edge_attr

        # Define Scaler and standardize inputs and targets
        targets = torch.tensor(scaler.fit_transform(targets), dtype=torch.float32)
        inputs = torch.tensor(scaler.transform(inputs), dtype=torch.float32)

        # Make predictions for this batch
        outputs = model(inputs, edge_index, edge_attr=edge_attr)

        # Compute the loss and its gradients
        rmse_loss = torch.sqrt(loss_fn(outputs, targets))

        # Compute MAE loss
        mae_loss = mae_loss_fn(outputs, targets)

        # Compute MRE loss
        mre_loss = mre_loss_fn(outputs, targets)

        # Gather data and report
        val_rmse_loss += rmse_loss.item()
        val_mae_loss += mae_loss.item()
        val_mre_loss += mre_loss.item()

        last_idx = i + 1

    wandb.log({
        'grid name': grid_name,
        'epoch': epoch,
        'val_rmse_loss': val_rmse_loss / last_idx,
        'val_mae_loss': val_mae_loss / last_idx,
        'val_mre_loss': val_mre_loss / last_idx

    })


def train_validate_one_epoch(epoch, grid_name, optimizer, training_loader, validation_loader, model, loss_fn, scaler):

    print("Training the model for epoch " + str(epoch))
    # Train for an epoch
    #model.train()
    train_one_epoch(epoch, grid_name, optimizer, training_loader, model, loss_fn, scaler)
    print("Validating the model on unseen Datasets for epoch " + str(epoch))
    # Validate for an epoch
    #model.eval()
    validate_one_epoch(epoch, grid_name, validation_loader, model, loss_fn, scaler)


def test_one_epoch(test_loader, grid_name, model, loss_fn, scaler):

    test_rmse_loss = 0.0
    test_mae_loss = 0.0
    test_mre_loss = 0.0
    output = None
    target = None

    # create a criterion to measure the mean absolute error
    mae_loss_fn = nn.L1Loss()

    # create a criterion to measure the mean relative error (MRE), outputs, targets : Torch.Tensor
    mre_loss_fn = lambda outputs, targets: get_mre_loss(outputs, targets)

    last_idx = 0

    # Here, we use enumerate(validation_loader) instead of
    # iter(validation_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(test_loader):
        inputs, targets = data.x, data.y

        # Get edge_index and edge_attr from DataLoader
        edge_index = test_loader.dataset[i].edge_index
        edge_attr = test_loader.dataset[i].edge_attr

        # Define Scaler and standardize inputs and targets
        targets = torch.tensor(scaler.fit_transform(targets), dtype=torch.float32)
        inputs = torch.tensor(scaler.transform(inputs), dtype=torch.float32)

        # Make predictions for this batch
        outputs = model(inputs, edge_index, edge_attr=edge_attr)

        # Compute the loss and its gradients
        rmse_loss = torch.sqrt(loss_fn(outputs, targets))

        # Compute MAE loss
        mae_loss = mae_loss_fn(outputs, targets)

        # Compute MRE loss
        mre_loss = mre_loss_fn(outputs, targets)

        # Gather data and report
        test_rmse_loss += rmse_loss.item()
        test_mae_loss += mae_loss.item()
        test_mre_loss += mre_loss.item()

        last_idx = i + 1

        if i == len(test_loader) - 1:
            output = scaler.inverse_transform(outputs.detach().numpy())
            target = scaler.inverse_transform(targets.detach().numpy())

    rmse = test_rmse_loss / last_idx
    mae = test_mae_loss / last_idx
    mre = test_mre_loss / last_idx
    wandb.log({
        'grid name': grid_name,
        'test_rmse_loss': rmse,
        'test_mae_loss': mae,
        'test_mre_loss': mre
    })

    return output,target, rmse, mae, mre



def get_mre_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    num_rows, num_cols = np.shape(outputs)
    diff = torch.sub(outputs, targets)
    div = torch.div(diff, targets)
    # difference = torch.sub(quotient, torch.ones_like(quotient))
    return torch.sum(torch.abs(div)) / (num_rows * num_cols)


def print_model(model):
    modules = [module for module in model.modules()]
    params = [param.shape for param in model.parameters()]

    # Print Model Summary
    print(modules[0])
    total_params = 0
    for i in range(1, len(modules)):
        j = 2 * i
        param = (params[j - 2][1] * params[j - 2][0]) + params[j - 1][0]
        total_params += param
        print("Layer", i, "->\t", end="")
        print("Weights:", params[j - 2][0], "x", params[j - 2][1],
              "\tBias: ", params[j - 1][0], "\tParameters: ", param)
    print("\nTotal Params: ", total_params)


# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def train_all_one_epoch(epoch, optimizer, training_loader_lst, model, loss_fn):

    train_rmse_loss = 0.0
    train_mae_loss = 0.0
    train_mre_loss = 0.0

    # create a criterion to measure the mean absolute error (MAE)
    mae_loss_fn = nn.L1Loss()

    # create a criterion to measure the mean relative error (MRE), inputs_targets: List
    mre_loss_fn = lambda outputs, targets : get_mre_loss(outputs, targets)

    for i, training_loader in enumerate(training_loader_lst):


        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for j, data in enumerate(training_loader):
            scaler = StandardScaler()

            # Every data instance is an input + label pair
            inputs, targets = data.x, data.y

            # Get edge_index and edge_attr from DataLoader
            edge_index = training_loader.dataset[j].edge_index
            edge_attr = training_loader.dataset[j].edge_attr

            # Define Scaler and standardize inputs and targets
            targets = torch.tensor(scaler.fit_transform(targets), dtype=torch.float32)
            inputs = torch.tensor(scaler.transform(inputs), dtype=torch.float32)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs, edge_index, edge_attr=edge_attr)

            # Compute the loss and its gradients for RMSE loss
            rmse_loss = torch.sqrt(loss_fn(outputs, targets))
            rmse_loss.backward()

            # Compute MAE loss
            mae_loss = mae_loss_fn(outputs, targets)
            #mae_loss.backward()

            # Compute MRE loss
            mre_loss = mre_loss_fn(outputs, targets)

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_rmse_loss += rmse_loss.item()
            train_mae_loss += mae_loss.item()
            train_mre_loss += mre_loss.item()

    num_samples = len(training_loader_lst) * len(training_loader_lst[0].dataset)
    wandb.log({
        'epoch': epoch,
        'train_rmse_loss': train_rmse_loss/ num_samples,
        'train_mae_loss': train_mae_loss / num_samples,
        'train_mre_loss': train_mre_loss / num_samples

    })


def validate_all_one_epoch(epoch, validation_loader_lst, model, loss_fn):

    val_rmse_loss = 0.0
    val_mae_loss = 0.0
    val_mre_loss = 0.0

    # create a criterion to measure the mean absolute error
    mae_loss_fn = nn.L1Loss()

    # create a criterion to measure the mean relative error (MRE), outputs, targets : Torch.Tensor
    mre_loss_fn = lambda outputs, targets: get_mre_loss(outputs, targets)

    for i, validation_loader in enumerate(validation_loader_lst):

        # Here, we use enumerate(validation_loader) instead of
        # iter(validation_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for j, data in enumerate(validation_loader):

            scaler = StandardScaler()
            inputs, targets = data.x, data.y

            # Get edge_index and edge_attr from DataLoader
            edge_index = validation_loader.dataset[j].edge_index
            edge_attr = validation_loader.dataset[j].edge_attr

            # Define Scaler and standardize inputs and targets
            targets = torch.tensor(scaler.fit_transform(targets), dtype=torch.float32)
            inputs = torch.tensor(scaler.transform(inputs), dtype=torch.float32)

            # Make predictions for this batch
            outputs = model(inputs, edge_index, edge_attr=edge_attr)

            # Compute the loss and its gradients
            rmse_loss = torch.sqrt(loss_fn(outputs, targets))

            # Compute MAE loss
            mae_loss = mae_loss_fn(outputs, targets)

            # Compute MRE loss
            mre_loss = mre_loss_fn(outputs, targets)

            # Gather data and report
            val_rmse_loss += rmse_loss.item()
            val_mae_loss += mae_loss.item()
            val_mre_loss += mre_loss.item()

    num_samples = len(validation_loader_lst) * len(validation_loader_lst[0].dataset)
    wandb.log({
        'epoch': epoch,
        'val_rmse_loss': val_rmse_loss / num_samples,
        'val_mae_loss': val_mae_loss / num_samples,
        'val_mre_loss': val_mre_loss / num_samples

    })

def test_all_one_epoch(test_loader_lst, model, loss_fn):

    test_rmse_loss = 0.0
    test_mae_loss = 0.0
    test_mre_loss = 0.0
    out = []

    # create a criterion to measure the mean absolute error
    mae_loss_fn = nn.L1Loss()

    # create a criterion to measure the mean relative error (MRE), outputs, targets : Torch.Tensor
    mre_loss_fn = lambda outputs, targets: get_mre_loss(outputs, targets)

    for i, test_loader in enumerate(test_loader_lst):

        # Here, we use enumerate(validation_loader) instead of
        # iter(validation_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for j, data in enumerate(test_loader):
            scaler = StandardScaler()
            inputs, targets = data.x, data.y

            # Get edge_index and edge_attr from DataLoader
            edge_index = test_loader.dataset[j].edge_index
            edge_attr = test_loader.dataset[j].edge_attr

            # Define Scaler and standardize inputs and targets
            targets = torch.tensor(scaler.fit_transform(targets), dtype=torch.float32)
            inputs = torch.tensor(scaler.transform(inputs), dtype=torch.float32)

            # Make predictions for this batch
            outputs = model(inputs, edge_index, edge_attr=edge_attr)

            # Compute the loss and its gradients
            rmse_loss = torch.sqrt(loss_fn(outputs, targets))

            # Compute MAE loss
            mae_loss = mae_loss_fn(outputs, targets)

            # Compute MRE loss
            mre_loss = mre_loss_fn(outputs, targets)

            # Gather data and report
            test_rmse_loss += rmse_loss.item()
            test_mae_loss += mae_loss.item()
            test_mre_loss += mre_loss.item()


            if j + 1 == len(test_loader.dataset):
                output = scaler.inverse_transform(outputs.detach().numpy())
                target = scaler.inverse_transform(targets.detach().numpy())
                out.append((output, target))

    num_samples = len(test_loader_lst) * len(test_loader_lst[0].dataset)
    rmse = test_rmse_loss / num_samples
    mae = test_mae_loss / num_samples
    mre = test_mre_loss / num_samples
    wandb.log({
        'test_rmse_loss': rmse,
        'test_mae_loss': mae,
        'test_mre_loss': mre
    })

    return out


def extract_node_types_as_dict(net: pp.pandapowerNet) -> Tuple[dict, dict]:
    node_type_bus_idx_dict = dict()
    node_type_bus_idx_dict["SB"] = []  # Slack Bus
    node_type_bus_idx_dict["PQ"] = []  # Load Busses
    node_type_bus_idx_dict["PV"] = []  # Generator Busses
    node_type_bus_idx_dict["NB"] = []  # Neutral Busses

    # Map indices of busses to ascending correct order
    idx_mapper = dict()
    for idx_given, idx_real in zip(net.bus.index.values, range(len(net.bus.index))):
        idx_mapper[idx_given] = idx_real

    # Set the bus indices list and remove the slack bus index
    bus_indices = list(net.bus.index.values)

    # Store the index of the slack bus
    """
    slack_bus_dict = pp.get_connected_elements_dict(net, net.ext_grid.iloc[0].bus)
    if "bus" in slack_bus_dict:
        node_type_bus_idx_dict["SB"].append(slack_bus_dict["bus"][0])
        # Remove the slack bus node idx from the list
        bus_indices.remove(slack_bus_dict["bus"][0])
    """
    if net.ext_grid.iloc[0].bus is not None:
        ext_grid_bus = pp.get_connected_buses(net, net.ext_grid.iloc[0].bus).pop()
        node_type_bus_idx_dict["SB"].append(ext_grid_bus)
        # Remove the slack bus node idx from the list
        bus_indices.remove(ext_grid_bus)

    # Interate over all bus indices as given in the PP framework
    for idx_given in bus_indices:

        # Get all generator indices connected to this bus
        gen_idx_set = pp.get_connected_elements(net, "gen", idx_given)

        # Get all load indices connected to this bus
        load_idx_set = pp.get_connected_elements(net, "load", idx_given)

        # Get the number of generators connected to the bus
        gen_count = len(gen_idx_set)

        # Get the number of loads connected to the bus
        load_count = len(load_idx_set)

        """
        sum_nominal_power_supply, sum_nominal_power_demand = 0.0, 0.0

        if idx_given in net.sgen.bus.values:
            sum_nominal_power_supply += list(net.sgen.loc[net.sgen.bus.values == idx_given].sn_mva).pop()

        if gen_count and load_count:

            for gen_idx in gen_idx_set:
                sum_nominal_power_supply += net.gen.iloc[gen_idx].sn_mva

            for load_idx in load_idx_set:
                sum_nominal_power_demand += net.load.iloc[load_idx].sn_mva

            total_sn_mva = sum_nominal_power_supply - sum_nominal_power_demand

            if total_sn_mva > 0.0:
                node_type_bus_idx_dict["PV"].append(idx_given)
            elif total_sn_mva == 0.0:
                node_type_bus_idx_dict["NB"].append(idx_given)
            else:
                node_type_bus_idx_dict["PQ"].append(idx_given)

        elif gen_count:
            node_type_bus_idx_dict["PV"].append(idx_given)

        elif load_count:

            for load_idx in load_idx_set:
                sum_nominal_power_demand += net.load.iloc[load_idx].sn_mva

            total_sn_mva = sum_nominal_power_supply - sum_nominal_power_demand

            if total_sn_mva > 0.0:
                node_type_bus_idx_dict["PV"].append(idx_given)
            elif total_sn_mva == 0.0:
                node_type_bus_idx_dict["NB"].append(idx_given)
            else:
                node_type_bus_idx_dict["PQ"].append(idx_given)

        else:
            if sum_nominal_power_supply == 0.0:
                node_type_bus_idx_dict["NB"].append(idx_given)
            else:
                node_type_bus_idx_dict["PV"].append(idx_given)
        """

        if gen_count:
            node_type_bus_idx_dict["PV"].append(idx_given)
        elif load_count:
            node_type_bus_idx_dict["PQ"].append(idx_given)
        else:
            node_type_bus_idx_dict["NB"].append(idx_given)

    return idx_mapper, node_type_bus_idx_dict



def extract_edge_features_as_dict(net: pp.pandapowerNet) -> Tuple[dict, dict]:
    """
    Bus Types: SB, PV, PQ, NB ( 4 Classes in total)
    Edge Types: undirected Edges (9 edge classes in total)
        SB - PV, SB - PQ, SB - NB
        PV - PQ, PV - NB, PV - PV
        PQ - NB; PQ - PQ
        NB - NB
    Args:
        net: pandapowerNet

    Returns:
        HeteroData

    """


    # Replace all ext_grids but the first one with generators and set the generators to slack= false
    ext_grids = [i for i in range(1, len(net.ext_grid.name.values))]
    pp.replace_ext_grid_by_gen(net, ext_grids=ext_grids, slack=False)

    # Get node types and idx mapper
    print("Extracting Node Types and Index Mapper..")
    idx_mapper, node_types_idx_dict = extract_node_types_as_dict(net)

    node_type_idx_mapper = dict()
    for node_type in node_types_idx_dict:
        length = len(node_types_idx_dict[node_type])
        for bus_idx, map_idx in zip(node_types_idx_dict[node_type], range(length)):
            real_idx = idx_mapper[bus_idx]
            node_type_idx_mapper[real_idx] = map_idx


    # Extract edge types
    edge_types_idx_dict = dict()

    edge_types_idx_dict["SB-PV"] = [[], []]
    edge_types_idx_dict["SB-PQ"] = [[], []]
    edge_types_idx_dict["SB-NB"] = [[], []]
    edge_types_idx_dict["PV-PQ"] = [[], []]
    edge_types_idx_dict["PV-NB"] = [[], []]
    edge_types_idx_dict["PQ-NB"] = [[], []]

    edge_types_idx_dict["PV-SB"] = [[], []]
    edge_types_idx_dict["PQ-SB"] = [[], []]
    edge_types_idx_dict["NB-SB"] = [[], []]
    edge_types_idx_dict["PQ-PV"] = [[], []]
    edge_types_idx_dict["NB-PV"] = [[], []]
    edge_types_idx_dict["NB-PQ"] = [[], []]

    edge_types_idx_dict["PV-PV"] = [[], []]
    edge_types_idx_dict["PQ-PQ"] = [[], []]
    edge_types_idx_dict["NB-NB"] = [[], []]


    # Store Edge Attributes
    edge_types_attr_dict = dict()

    edge_types_attr_dict["SB-PV"] = []
    edge_types_attr_dict["SB-PQ"] = []
    edge_types_attr_dict["SB-NB"] = []
    edge_types_attr_dict["PV-PQ"] = []
    edge_types_attr_dict["PV-NB"] = []
    edge_types_attr_dict["PQ-NB"] = []

    edge_types_attr_dict["PV-SB"] = []
    edge_types_attr_dict["PQ-SB"] = []
    edge_types_attr_dict["NB-SB"] = []
    edge_types_attr_dict["PQ-PV"] = []
    edge_types_attr_dict["NB-PV"] = []
    edge_types_attr_dict["NB-PQ"] = []

    edge_types_attr_dict["PV-PV"] = []
    edge_types_attr_dict["PQ-PQ"] = []
    edge_types_attr_dict["NB-NB"] = []

    print("Extracting Edge Index and Edge Attributes..")

    for from_bus, to_bus, r_ohm_per_km, x_ohm_per_km, length_km in zip(net.line.from_bus, net.line.to_bus, net.line.r_ohm_per_km, net.line.x_ohm_per_km, net.line.length_km):

        # Calculate R_i and X_i
        R_i = length_km * r_ohm_per_km
        X_i = length_km * x_ohm_per_km

        # Get the types of from and to busses, thus the edge type
        from_bus_type = get_node_type(from_bus, node_types_idx_dict)
        to_bus_type = get_node_type(to_bus, node_types_idx_dict)
        edge_type = from_bus_type + '-' + to_bus_type

        # if edge_type not in edge_types_attr_dict:
        #     str_lst = edge_type.split('-')
        #     edge_type = str_lst.pop() + "-" + str_lst.pop()

        # Add Edge indices to corresponding Edge Types
        from_bus_edge_idx = node_type_idx_mapper[idx_mapper[from_bus]]
        to_bus_edge_idx = node_type_idx_mapper[idx_mapper[to_bus]]

        #from_bus_edge_idx = idx_mapper[from_bus]
        #to_bus_edge_idx = idx_mapper[to_bus]

        edge_types_idx_dict[edge_type][0].append(from_bus_edge_idx)
        edge_types_idx_dict[edge_type][1].append(to_bus_edge_idx)

        # Add Edge Attributes to corresponding Edge Types
        edge_types_attr_dict[edge_type].append([R_i, X_i])

        if from_bus_type != to_bus_type:
            edge_type = to_bus_type + '-' + from_bus_type
            edge_types_idx_dict[edge_type][0].append(to_bus_edge_idx)
            edge_types_idx_dict[edge_type][1].append(from_bus_edge_idx)
            # Add Edge Attributes to corresponding Edge Types
            edge_types_attr_dict[edge_type].append([R_i, X_i])

    print("Adding Self Loops and Converting Edges to Undirected..")

    for key in edge_types_idx_dict:
        # Convert edge connections to undirected
        # n = len(edge_types_idx_dict[key][0])
        # edge_index, edge_attr = to_undirected(edge_index=torch.tensor(edge_types_idx_dict[key], dtype=torch.int),
        #                                       edge_attr=torch.tensor(edge_types_attr_dict[key], dtype=torch.float32),
        #                                       num_nodes=n)

        # Add self loops to edge connections
        #edge_index, edge_attr = add_self_loops(edge_index=edge_index, edge_attr=edge_attr, fill_val=1.)

        unique_edge_pairs = set()
        edge_index = edge_types_idx_dict[key]
        edge_attr = edge_types_attr_dict[key]
        i = 0
        while i < len(edge_index[0]):
            from_edge = edge_index[0][i]
            to_edge = edge_index[1][i]

            if (from_edge, to_edge) in unique_edge_pairs:
                del edge_index[0][i]
                del edge_index[1][i]
                del edge_attr[i]
                continue

            unique_edge_pairs.add((from_edge, to_edge))
            i+=1

        edge_types_idx_dict[key] = torch.tensor(edge_index, dtype=torch.int64)#edge_index
        edge_types_attr_dict[key] = torch.tensor(edge_attr, dtype=torch.float32)


    print("Hetero Data Created.")
    for key in edge_types_idx_dict:
        print(f"{key}: {edge_types_idx_dict[key]}" )

    return edge_types_idx_dict, edge_types_attr_dict



def get_node_type(bus_idx: int, node_types_idx_dict: dict) -> str:
    for key in node_types_idx_dict:
        for idx in node_types_idx_dict[key]:
            if bus_idx == idx:
                return key

    return None

def add_self_loops(edge_index: torch.Tensor, edge_attr: torch.Tensor, fill_val = 1.) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Must be applied after converting the edges to "undirected"
    Args:
        edge_index: Tensor
        edge_attr: Tensor

    Returns:
        edge_index_with_self_loops: Tensor
        edge_attr_with_self_loops: Tensor

    """
    idx_lst = edge_index.tolist()
    attr_lst = edge_attr.tolist()

    n = int(len(idx_lst[0])/2)
    s = set(idx_lst[0][:n])
    for i in s:

        # Add Self-Loops to Edge Index
        idx_lst[0].append(i)
        idx_lst[1].append(i)

        # Add the additional edge attributes
        attr_lst.append([fill_val, fill_val])


    return torch.tensor(idx_lst, dtype=torch.int), torch.tensor(attr_lst, dtype=torch.float32)

def read_unsupervised_dataset(grid_name: str) -> Tuple[list, list, list]:
    """
    Reads unsupervised datasets and returns HeteroData
    Args:
        grid_name: string

    Returns:
        HeteroData

    """
    # Define the graph using Pandapower
    net = sb.get_simbench_net(grid_name)

    print(f"Extracting Node Types for the grid {grid_name}..")

    idx_mapper, node_types_idx_dict = extract_node_types_as_dict(net)

    node_type_idx_mapper = dict()
    for node_type in node_types_idx_dict:
        length = len(node_types_idx_dict[node_type])
        for bus_idx, map_idx in zip(node_types_idx_dict[node_type], range(length)):
            real_idx = idx_mapper[bus_idx]
            node_type_idx_mapper[real_idx] = map_idx

    print(f"Extracting Edge Index and Edge Attributes for each Node Type for the grid {grid_name}")

    edge_types_idx_dict, edge_types_attr_dict = extract_edge_features_as_dict(net)

    print("Reading all of the .csv files from the directory of " + grid_name + " ...")

    # Store path to the supervised datasets directory of the specified grid
    train_data, val_data, test_data = [], [], []
    path_to_dir = os.path.dirname(os.path.abspath("gnn.ipynb")) + "\\data\\Unsupervised\\Training\\" + grid_name
    datasets = []

    # Read all the csv files in the directory of the grid_name
    for dataset_name in os.listdir(path_to_dir):
        path2dataset = path_to_dir + "\\" + dataset_name
        datasets.append(pd.read_csv(path2dataset))

    # Process all the data according to  85 - 10 - 5
    random.shuffle(datasets)
    training = datasets[:85]
    validation = datasets[85:95]
    test = datasets[95:]

    print("Processing Training Data for " + grid_name + " ...")
    num_busses = int(len(training[0]))

    for data in training:
        hetero_data = HeteroData()
        x = data.drop(columns=["Unnamed: 0"])
        x_rows, x_cols = np.shape(x)
        x = np.array(x).reshape(x_rows, x_cols)

        # Store the Feature Matrices for each bus type
        for bus_type in node_types_idx_dict:
            lst = []
            # Get the bus indices of the corresponding bus type
            type_idx_lst = node_types_idx_dict[bus_type]

            # For each index given, map it to the true index and append the corresponding feature vector
            for idx_given in type_idx_lst:
                lst.append(x[idx_mapper[idx_given], :])

            if len(lst) == 0:
                continue

            lst_rows, lst_cols = np.shape(lst)
            lst_np = np.array(lst).reshape(lst_rows, lst_cols)
            hetero_data[bus_type].x = torch.tensor(data=lst_np, dtype=torch.float32)
            if len(hetero_data[bus_type]) == 0:
                hetero_data[bus_type].num_nodes = 0

        # Store the Edge Attributes and Index for each edge type
        for edge_type in edge_types_idx_dict:
            str_lst = edge_type.split('-')
            #new_edge_type = str_lst[0], "connects", str_lst[1]
            if edge_types_idx_dict[edge_type].numel() != 0:
                hetero_data[str_lst[0], "isConnected", str_lst[1]].edge_index = edge_types_idx_dict[edge_type]
                hetero_data[str_lst[0], "isConnected", str_lst[1]].edge_attr = edge_types_attr_dict[edge_type]


        train_data.append(hetero_data)

    print("Processing Validation Data for " + grid_name + " ...")

    for data in validation:
        hetero_data = HeteroData()
        x = data.drop(columns=["Unnamed: 0"])
        x_rows, x_cols = np.shape(x)
        x = np.array(x).reshape(x_rows, x_cols)

        # Store the Feature Matrices for each bus type
        for bus_type in node_types_idx_dict:
            lst = []
            # Get the bus indices of the corresponding bus type
            type_idx_lst = node_types_idx_dict[bus_type]

            # For each index given, map it to the true index and append the corresponding feature vector
            for idx_given in type_idx_lst:
                lst.append(x[idx_mapper[idx_given], :])

            if len(lst) == 0:
                continue

            lst_rows, lst_cols = np.shape(lst)
            lst_np = np.array(lst).reshape(lst_rows, lst_cols)
            hetero_data[bus_type].x = torch.tensor(data=lst_np, dtype=torch.float32)
            if len(hetero_data[bus_type]) == 0:
                hetero_data[bus_type].num_nodes = 0

        # Store the Edge Attributes and Index for each edge type
        for edge_type in edge_types_idx_dict:
            str_lst = edge_type.split('-')
            # new_edge_type = str_lst[0], "connects", str_lst[1]
            if edge_types_idx_dict[edge_type].numel() != 0:
                hetero_data[str_lst[0], "isConnected", str_lst[1]].edge_index = edge_types_idx_dict[edge_type]
                hetero_data[str_lst[0], "isConnected", str_lst[1]].edge_attr = edge_types_attr_dict[edge_type]

        val_data.append(hetero_data)

    print("Processing Test Data for " + grid_name + " ...")

    for data in test:
        hetero_data = HeteroData()
        x = data.drop(columns=["Unnamed: 0"])
        x_rows, x_cols = np.shape(x)
        x = np.array(x).reshape(x_rows, x_cols)

        # Store the Feature Matrices for each bus type
        for bus_type in node_types_idx_dict:
            lst = []
            # Get the bus indices of the corresponding bus type
            type_idx_lst = node_types_idx_dict[bus_type]

            # For each index given, map it to the true index and append the corresponding feature vector
            for idx_given in type_idx_lst:
                lst.append(x[idx_mapper[idx_given], :])

            if len(lst) == 0:
                continue

            lst_rows, lst_cols = np.shape(lst)
            lst_np = np.array(lst).reshape(lst_rows, lst_cols)
            hetero_data[bus_type].x = torch.tensor(data=lst_np, dtype=torch.float32)
            if len(hetero_data[bus_type]) == 0:
                hetero_data[bus_type].num_nodes = 0

        # Store the Edge Attributes and Index for each edge type
        for edge_type in edge_types_idx_dict:
            str_lst = edge_type.split('-')
            # new_edge_type = str_lst[0], "connects", str_lst[1]
            if edge_types_idx_dict[edge_type].numel() != 0:
                hetero_data[str_lst[0],"isConnected",str_lst[1]].edge_index = edge_types_idx_dict[edge_type]
                hetero_data[str_lst[0],"isConnected",str_lst[1]].edge_attr = edge_types_attr_dict[edge_type]

        test_data.append(hetero_data)

    print("Processing complete.")
    return train_data, val_data, test_data

def to_json(grid_name: str):
    """

    Args:
        grid_name:

    Returns: None

    """

    # Define the graph using Pandapower
    net = sb.get_simbench_net(grid_name)

    # Map indices of busses to ascending correct order
    idx_mapper = dict()

    for idx_given, idx_real in zip(net.bus.index.values, range(len(net.bus.index))):
        idx_mapper[idx_given] = idx_real

    # Set the bus indices list and remove the slack bus index
    bus_indices = list(net.bus.index.values)
    x_dict = dict()
    x_dict["busses"] = dict()
    x_dict["lines"] = dict()

    # Interate over all bus indices as given in the PP framework
    for idx_given in bus_indices:
        idx = idx_mapper[idx_given]

        # Get all generator indices connected to this bus
        gen_idx_set = pp.get_connected_elements(net, "gen", idx_given)

        # Get all load indices connected to this bus
        load_idx_set = pp.get_connected_elements(net, "load", idx_given)

        # Get the number of generators connected to the bus
        gen_count = len(gen_idx_set)

        # Get the number of loads connected to the bus
        load_count = len(load_idx_set)

        x_dict["busses"][str(idx)] = dict()

        if idx_given == net.ext_grid.iloc[0].bus:
            x_dict["busses"][str(idx)]["Bus Type"] = "SB"
        elif gen_count:
            x_dict["busses"][str(idx)]["Bus Type"] = "PV"
        elif load_count:
            x_dict["busses"][str(idx)]["Bus Type"] = "PQ"
        else:
            x_dict["busses"][str(idx)]["Bus Type"] = "NB"

    edge_attr = []
    edge_index = []

    for from_bus, to_bus, r_ohm_per_km, x_ohm_per_km, length_km in zip(net.line.from_bus, net.line.to_bus,
                                                                       net.line.r_ohm_per_km, net.line.x_ohm_per_km,
                                                                       net.line.length_km):
        # Add self loops
        # edge_index[0].append(idx_mapper[from_bus])
        # edge_index[1].append(idx_mapper[from_bus])

        # Add interbus connections
        edge_index.append([idx_mapper[from_bus], idx_mapper[to_bus]])

        # Calculate R_i and X_i
        R_i = length_km * r_ohm_per_km
        X_i = length_km * x_ohm_per_km

        edge_attr.append([R_i, X_i])

    x_dict["lines"]["edge_index"] = edge_index
    x_dict["lines"]["edge_attr"] = edge_attr

    # Store path to the supervised datasets directory of the specified grid
    path_to_dir = os.path.dirname(os.path.abspath("gnn.ipynb")) + "\\data\\Supervised\\" + grid_name

    # Read all the csv files in the directory of the grid_name
    dataset_name = os.listdir(path_to_dir)[0]
    path2dataset = path_to_dir + "\\" + dataset_name
    data = pd.read_csv(path2dataset)

    num_busses = int(len(data) / 2)

    x = data[:num_busses].drop(columns=["Unnamed: 0"])
    x_rows, x_cols = np.shape(x)
    x = np.array(x).reshape(x_rows, x_cols)

    for i in range(num_busses):
        #x_dict["busses"][str(i)] = dict()
        x_dict["busses"][str(i)]["Node Features"] = list(x[i])

    json_object = json.dumps(x_dict, indent=1)
    with open("data.json", "w") as outfile:
        outfile.write(json_object)

def extract_edge_features(net: pp.pandapowerNet) -> Tuple[dict, dict]:
    """
    Bus Types: SB, PV, PQ, NB ( 4 Classes in total)
    Edge Types: undirected Edges (9 edge classes in total)
        SB - PV, SB - PQ, SB - NB
        PV - PQ, PV - NB, PV - PV
        PQ - NB; PQ - PQ
        NB - NB
    Args:
        net: pandapowerNet

    Returns:
        HeteroData

    """


    # Replace all ext_grids but the first one with generators and set the generators to slack= false
    ext_grids = [i for i in range(1, len(net.ext_grid.name.values))]
    pp.replace_ext_grid_by_gen(net, ext_grids=ext_grids, slack=False)

    # Get node types and idx mapper
    print("Extracting Node Types and Index Mapper..")
    idx_mapper, node_types_idx_dict = extract_node_types_as_dict(net)

    # Extract edge types
    edge_types_idx_dict = dict()

    edge_types_idx_dict["SB-PV"] = [[], []]
    edge_types_idx_dict["SB-PQ"] = [[], []]
    edge_types_idx_dict["SB-NB"] = [[], []]
    edge_types_idx_dict["PV-PQ"] = [[], []]
    edge_types_idx_dict["PV-NB"] = [[], []]
    edge_types_idx_dict["PQ-NB"] = [[], []]

    edge_types_idx_dict["PV-SB"] = [[], []]
    edge_types_idx_dict["PQ-SB"] = [[], []]
    edge_types_idx_dict["NB-SB"] = [[], []]
    edge_types_idx_dict["PQ-PV"] = [[], []]
    edge_types_idx_dict["NB-PV"] = [[], []]
    edge_types_idx_dict["NB-PQ"] = [[], []]

    edge_types_idx_dict["PV-PV"] = [[], []]
    edge_types_idx_dict["PQ-PQ"] = [[], []]
    edge_types_idx_dict["NB-NB"] = [[], []]


    # Store Edge Attributes
    edge_types_attr_dict = dict()

    edge_types_attr_dict["SB-PV"] = []
    edge_types_attr_dict["SB-PQ"] = []
    edge_types_attr_dict["SB-NB"] = []
    edge_types_attr_dict["PV-PQ"] = []
    edge_types_attr_dict["PV-NB"] = []
    edge_types_attr_dict["PQ-NB"] = []

    edge_types_attr_dict["PV-SB"] = []
    edge_types_attr_dict["PQ-SB"] = []
    edge_types_attr_dict["NB-SB"] = []
    edge_types_attr_dict["PQ-PV"] = []
    edge_types_attr_dict["NB-PV"] = []
    edge_types_attr_dict["NB-PQ"] = []

    edge_types_attr_dict["PV-PV"] = []
    edge_types_attr_dict["PQ-PQ"] = []
    edge_types_attr_dict["NB-NB"] = []

    print("Extracting Edge Index and Edge Attributes..")

    for from_bus, to_bus, r_ohm_per_km, x_ohm_per_km, length_km in zip(net.line.from_bus, net.line.to_bus, net.line.r_ohm_per_km, net.line.x_ohm_per_km, net.line.length_km):

        # Calculate R_i and X_i
        R_i = length_km * r_ohm_per_km
        X_i = length_km * x_ohm_per_km

        # Get the types of from and to busses, thus the edge type
        from_bus_type = get_node_type(from_bus, node_types_idx_dict)
        to_bus_type = get_node_type(to_bus, node_types_idx_dict)
        edge_type = from_bus_type + '-' + to_bus_type

        # if edge_type not in edge_types_attr_dict:
        #     str_lst = edge_type.split('-')
        #     edge_type = str_lst.pop() + "-" + str_lst.pop()

        # Add Edge indices to corresponding Edge Types
        from_bus_edge_idx = idx_mapper[from_bus]
        to_bus_edge_idx = idx_mapper[to_bus]

        #from_bus_edge_idx = idx_mapper[from_bus]
        #to_bus_edge_idx = idx_mapper[to_bus]

        edge_types_idx_dict[edge_type][0].append(from_bus_edge_idx)
        edge_types_idx_dict[edge_type][1].append(to_bus_edge_idx)

        # Add Edge Attributes to corresponding Edge Types
        edge_types_attr_dict[edge_type].append([R_i, X_i])

        if from_bus_type != to_bus_type:
            edge_type = to_bus_type + '-' + from_bus_type
            edge_types_idx_dict[edge_type][0].append(to_bus_edge_idx)
            edge_types_idx_dict[edge_type][1].append(from_bus_edge_idx)
            # Add Edge Attributes to corresponding Edge Types
            edge_types_attr_dict[edge_type].append([R_i, X_i])

    print("Adding Self Loops and Converting Edges to Undirected..")

    for key in edge_types_idx_dict:
        # Convert edge connections to undirected
        # n = len(edge_types_idx_dict[key][0])
        # edge_index, edge_attr = to_undirected(edge_index=torch.tensor(edge_types_idx_dict[key], dtype=torch.int),
        #                                       edge_attr=torch.tensor(edge_types_attr_dict[key], dtype=torch.float32),
        #                                       num_nodes=n)

        # Add self loops to edge connections
        #edge_index, edge_attr = add_self_loops(edge_index=edge_index, edge_attr=edge_attr, fill_val=1.)

        unique_edge_pairs = set()
        edge_index = edge_types_idx_dict[key]
        edge_attr = edge_types_attr_dict[key]
        i = 0
        while i < len(edge_index[0]):
            from_edge = edge_index[0][i]
            to_edge = edge_index[1][i]

            if (from_edge, to_edge) in unique_edge_pairs:
                del edge_index[0][i]
                del edge_index[1][i]
                del edge_attr[i]
                continue

            unique_edge_pairs.add((from_edge, to_edge))
            i+=1

        edge_types_idx_dict[key] = torch.tensor(edge_index, dtype=torch.int64)#edge_index
        edge_types_attr_dict[key] = torch.tensor(edge_attr, dtype=torch.float32)


    print("Hetero Data Created.")
    for key in edge_types_idx_dict:
        print(f"{key}: {edge_types_idx_dict[key]}" )

    return edge_types_idx_dict, edge_types_attr_dict