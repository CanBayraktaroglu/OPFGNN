import copy

import torch
import torch.nn as nn
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np
import pandapower as pp
import wandb
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import simbench as sb
import os
import random
from graphdata import GraphData
from typing import Tuple, List, Any
import json
from ACOPFData import ACOPFInput, ACOPFOutput
import pickle
from gnn import GNN
from heterognn import ACOPFGNN, custom_tanh, ACOPFEnforcer, ACOPFEmbedder
from torch.optim.lr_scheduler import ReduceLROnPlateau
import asyncio


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

    for init in ["pf", "flat", "results"]:
        try:
            pp.runopp(net, init=init)  # Calculate ACOPF with IPFOPT
        except pp.OPFNotConverged:
            # print("OPTIMAL POWERFLOW COMPUTATION DID NOT CONVERGE FOR START VECTOR CONFIG " + init)
            if init == "results":
                # print("SKIPPING THIS DATASET.")
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

    # edge_attr = torch.tensor(StandardScaler().fit_transform(edge_attr), dtype=torch.float32)

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
        graphdata_lst.append(GraphData(_, train_data, val_data, test_data, edge_index, edge_attr))
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
    mre_loss_fn = lambda outputs, targets: get_mre_loss(outputs, targets)

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
        # mae_loss.backward()

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
        'grid name': grid_name,
        'epoch': epoch,
        'train_rmse_loss': train_rmse_loss / last_idx,
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
    # model.train()
    train_one_epoch(epoch, grid_name, optimizer, training_loader, model, loss_fn, scaler)
    print("Validating the model on unseen Datasets for epoch " + str(epoch))
    # Validate for an epoch
    # model.eval()
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

    return output, target, rmse, mae, mre


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
    mre_loss_fn = lambda outputs, targets: get_mre_loss(outputs, targets)

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
            # mae_loss.backward()

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
        'train_rmse_loss': train_rmse_loss / num_samples,
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
        ext_grid_bus = net.ext_grid.iloc[0].bus  # pp.get_connected_buses(net, net.ext_grid.iloc[0].bus).pop()
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

        if gen_count or sum(net.load.loc[net.load.bus == idx_given].p_mw.values) < sum(
                net.sgen.loc[net.sgen.bus == idx_given].p_mw.values):
            node_type_bus_idx_dict["PV"].append(idx_given)
        elif load_count and sum(net.load.loc[net.load.bus == idx_given].p_mw.values) != 0:
            node_type_bus_idx_dict["PQ"].append(idx_given)
        else:
            node_type_bus_idx_dict["NB"].append(idx_given)

    return idx_mapper, node_type_bus_idx_dict


def extract_edge_features_as_dict(net: pp.pandapowerNet, suppress_info: bool = True) -> Tuple[dict, dict]:
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
    if not suppress_info:
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

    if not suppress_info:
        print("Extracting Edge Index and Edge Attributes..")

    for from_bus, to_bus, r_ohm_per_km, x_ohm_per_km, length_km in zip(net.line.from_bus, net.line.to_bus,
                                                                       net.line.r_ohm_per_km, net.line.x_ohm_per_km,
                                                                       net.line.length_km):

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

        # from_bus_edge_idx = idx_mapper[from_bus]
        # to_bus_edge_idx = idx_mapper[to_bus]

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

    # print("Adding Self Loops and Converting Edges to Undirected..")

    for key in edge_types_idx_dict:
        # Convert edge connections to undirected
        # n = len(edge_types_idx_dict[key][0])
        # edge_index, edge_attr = to_undirected(edge_index=torch.tensor(edge_types_idx_dict[key], dtype=torch.int),
        #                                       edge_attr=torch.tensor(edge_types_attr_dict[key], dtype=torch.float32),
        #                                       num_nodes=n)

        # Add self loops to edge connections
        # edge_index, edge_attr = add_self_loops(edge_index=edge_index, edge_attr=edge_attr, fill_val=1.)

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
            i += 1

        edge_types_idx_dict[key] = edge_index
        edge_types_attr_dict[key] = edge_attr

    # Add the external grid connections
    ext_grid = net.ext_grid.iloc[0].bus
    ext_grid_connected_busses = pp.get_connected_buses(net, buses=ext_grid)

    for bus_idx in ext_grid_connected_busses:
        to_bus = get_node_type(bus_idx, node_types_idx_dict)
        edge = "SB-" + to_bus
        reverse_edge = to_bus + "-SB"

        ext_grid_trafos = net.trafo.loc[(net.trafo.hv_bus == ext_grid) & (net.trafo.lv_bus == bus_idx)]

        vk = ext_grid_trafos.vk_percent.values[0]
        vkr = ext_grid_trafos.vkr_percent.values[0]
        s = ext_grid_trafos.sn_mva.values[0]
        vn = ext_grid_trafos.vn_lv_kv.values[0]
        zk = vk * net.sn_mva / (100 * s)
        r = torch.tensor(vkr * net.sn_mva / (100 * s))
        zn = vn ** 2 / net.sn_mva
        z_ref_trafo = vn ** 2 * 10 ** 6 * net.sn_mva / s
        z = zk * z_ref_trafo / zn
        x = torch.sqrt(z ** 2 - r ** 2)

        edge_types_idx_dict[edge][0].append(0)
        edge_types_idx_dict[edge][1].append(node_type_idx_mapper[idx_mapper[bus_idx]])
        edge_types_attr_dict[edge].append([r, x])

        edge_types_idx_dict[reverse_edge][0].append(node_type_idx_mapper[idx_mapper[bus_idx]])
        edge_types_idx_dict[reverse_edge][1].append(0)
        edge_types_attr_dict[reverse_edge].append([r, x])

    # Add generator bus features
    gen_busses_idx = list(net.gen.bus.values)
    gen_connected_busses = list(pp.get_connected_buses(net, gen_busses_idx))
    for i in range(len(gen_busses_idx)):
        bus_idx = gen_busses_idx[i]
        to_bus = get_node_type(bus_idx, node_types_idx_dict)
        edge = "PV-" + to_bus
        reverse_edge = to_bus + "-PV"

        gen_trafos = net.trafo.loc[(net.trafo.hv_bus == bus_idx) & (net.trafo.lv_bus == gen_connected_busses[i])]

        vk = gen_trafos.vk_percent.values[0]
        vkr = gen_trafos.vkr_percent.values[0]
        s = gen_trafos.sn_mva.values[0]
        vn = gen_trafos.vn_lv_kv.values[0]
        zk = vk * net.sn_mva / (100 * s)
        r = torch.tensor(vkr * net.sn_mva / (100 * s))
        zn = vn ** 2 / net.sn_mva
        z_ref_trafo = vn ** 2 * 10 ** 6 * net.sn_mva / s
        z = zk * z_ref_trafo / zn
        x = torch.sqrt(z ** 2 - r ** 2)

        edge_types_idx_dict[edge][0].append(node_type_idx_mapper[idx_mapper[bus_idx]])
        edge_types_idx_dict[edge][1].append(node_type_idx_mapper[idx_mapper[gen_connected_busses[i]]])
        edge_types_attr_dict[edge].append([r, x])

        edge_types_idx_dict[reverse_edge][0].append(node_type_idx_mapper[idx_mapper[gen_connected_busses[i]]])
        edge_types_idx_dict[reverse_edge][1].append(node_type_idx_mapper[idx_mapper[bus_idx]])
        edge_types_attr_dict[reverse_edge].append([r, x])

    for key in edge_types_idx_dict:
        edge_types_idx_dict[key] = torch.tensor(edge_types_idx_dict[key], dtype=torch.int64)
        edge_types_attr_dict[key] = torch.tensor(edge_types_attr_dict[key], dtype=torch.float32)
        # if key[0] == key[-1]:
        # edge_types_idx_dict[key] = to_undirected(edge_types_idx_dict[key])

    if not suppress_info:
        print("Hetero Data Created.")

    return edge_types_idx_dict, edge_types_attr_dict


def get_node_type(bus_idx: int, node_types_idx_dict: dict) -> Any:
    for key in node_types_idx_dict:
        for idx in node_types_idx_dict[key]:
            if bus_idx == idx:
                return key

    return None


def add_self_loops(edge_index: torch.Tensor, edge_attr: torch.Tensor, fill_val=1.) -> Tuple[torch.Tensor, torch.Tensor]:
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

    n = int(len(idx_lst[0]) / 2)
    s = set(idx_lst[0][:n])
    for i in s:
        # Add Self-Loops to Edge Index
        idx_lst[0].append(i)
        idx_lst[1].append(i)

        # Add the additional edge attributes
        attr_lst.append([fill_val, fill_val])

    return torch.tensor(idx_lst, dtype=torch.int), torch.tensor(attr_lst, dtype=torch.float32)


def process_network(grid_name: str, suppress_info: bool = True):
    if not suppress_info:
        print(f"Loading Network {grid_name}..")
    net = sb.get_simbench_net(grid_name)  # '1-HV-mixed--0-no_sw'

    # Sample loads uniformly
    net.load = sample_uniform_from_df(net.load)

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

    # NETWORK CONSTRAINTS

    # Maximize the branch limits

    # max_i_ka = list(net.line.max_i_ka.values)

    # for i in range(len(max_i_ka)):
    # max_i_ka[i] = max(max_i_ka)

    if not suppress_info:
        print(f"Processing Network named {grid_name}..")

    # Calculate reactive power from nominal and acive power for sgens
    s_sgen = np.array(net.sgen.sn_mva.values ** 2)
    p_sgen = np.array(net.sgen.p_mw.values ** 2)
    q_sgen = np.sqrt(s_sgen - p_sgen)
    net.sgen.q_mvar = list(q_sgen)

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

    pp.drop_out_of_service_elements(net)

    """
    # Cost assignment
    pp.create_pwl_costs(net, [i for i in range(len(net.gen.name.values))], et="gen",
                        points=[[[0, 20, 1], [20, 30, 2]] for _ in range(len(net.gen.name.values))])
    pp.create_pwl_costs(net, [i for i in range(len(net.sgen.name.values))], et="sgen",
                        points=[[[0, 20, 0.25], [20, 30, 0.5]] for _ in range(len(net.sgen.name.values))])
    pp.create_pwl_costs(net, [i for i in range(len(net.ext_grid.name.values))], et="ext_grid",
                        points=[[[0, 20, 2], [20, 30, 5]] for _ in range(len(net.ext_grid.name.values))])
    """
    if not suppress_info:
        print("Network Processing Finished.")
    return net


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
        # x_dict["busses"][str(i)] = dict()
        x_dict["busses"][str(i)]["Node Features"] = list(x[i])

    json_object = json.dumps(x_dict, indent=1)
    with open("data.json", "w") as outfile:
        outfile.write(json_object)


def extract_edge_features(net: pp.pandapowerNet, suppress_info: bool = True) -> Tuple[dict, dict]:
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
    if not suppress_info:
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

    if not suppress_info:
        print("Extracting Edge Index and Edge Attributes..")

    for from_bus, to_bus, r_ohm_per_km, x_ohm_per_km, length_km in zip(net.line.from_bus, net.line.to_bus,
                                                                       net.line.r_ohm_per_km, net.line.x_ohm_per_km,
                                                                       net.line.length_km):

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

        # from_bus_edge_idx = idx_mapper[from_bus]
        # to_bus_edge_idx = idx_mapper[to_bus]

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

    if not suppress_info:
        print("Adding Self Loops and Converting Edges to Undirected..")

    for key in edge_types_idx_dict:
        # Convert edge connections to undirected
        # n = len(edge_types_idx_dict[key][0])
        # edge_index, edge_attr = to_undirected(edge_index=torch.tensor(edge_types_idx_dict[key], dtype=torch.int),
        #                                       edge_attr=torch.tensor(edge_types_attr_dict[key], dtype=torch.float32),
        #                                       num_nodes=n)

        # Add self loops to edge connections
        # edge_index, edge_attr = add_self_loops(edge_index=edge_index, edge_attr=edge_attr, fill_val=1.)

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
            i += 1

        edge_types_idx_dict[key] = torch.tensor(edge_index, dtype=torch.int64)  # edge_index
        edge_types_attr_dict[key] = torch.tensor(edge_attr, dtype=torch.float32)

    if not suppress_info:
        print("Hetero Data Created.")
        for key in edge_types_idx_dict:
            print(f"{key}: {edge_types_idx_dict[key]}")

    return edge_types_idx_dict, edge_types_attr_dict


def read_unsupervised_dataset(grid_name: str) -> Tuple[list, list, list]:
    """
    Reads unsupervised datasets and returns HeteroData
    Args:
        grid_name: string

    Returns:
        HeteroData

    """
    # Define the graph using Pandapower
    net = process_network(grid_name)

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
            # new_edge_type = str_lst[0], "connects", str_lst[1]
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
                hetero_data[str_lst[0], "isConnected", str_lst[1]].edge_index = edge_types_idx_dict[edge_type]
                hetero_data[str_lst[0], "isConnected", str_lst[1]].edge_attr = edge_types_attr_dict[edge_type]

        test_data.append(hetero_data)

    print("Processing complete.")
    return train_data, val_data, test_data


def generate_unsupervised_input(grid_name: str, suppress_info: bool = True) -> Any:
    """
        Processes the PandaPower Network and generates inputs,returns HeteroData
        Args:
            grid_name: string

        Returns:
            each node feature is a 1 x 11 vector with parameters being in order:
            V_i, delta_iSB, P_i, Q_i,
            min_V_i, max_V_i, sn_mva, min_P_i,
            max_P_i, min_Q_i, max_Q_i
            HeteroData

        """

    # Process the network via Grid Name
    net = process_network(grid_name)

    if not suppress_info:
        print(f"Extracting Node Types for the grid {grid_name}..")

    idx_mapper, node_types_idx_dict = extract_node_types_as_dict(net)
    index_mappers = (idx_mapper, node_types_idx_dict)

    node_type_idx_mapper = dict()
    for node_type in node_types_idx_dict:
        length = len(node_types_idx_dict[node_type])
        for bus_idx, map_idx in zip(node_types_idx_dict[node_type], range(length)):
            real_idx = idx_mapper[bus_idx]
            node_type_idx_mapper[real_idx] = map_idx

    if not suppress_info:
        print(f"Extracting Edge Index and Edge Attributes for each Node Type for the grid {grid_name}")

    edge_types_idx_dict, edge_types_attr_dict = extract_edge_features_as_dict(net)

    # Initialize a Hetero Data Instance
    data = HeteroData()
    ext_grid_bus_idx = net.ext_grid.iloc[0].bus
    gen_busses_idx = list(net.gen.bus.values)

    for node_type in node_types_idx_dict:
        X_node_type = []  # number of busses of the corresponding node type x 11 vector
        node_indices = node_types_idx_dict[node_type]

        for bus_idx in node_indices:
            node_features = []  # 1 x 11 vector

            # Core Features
            V_i = torch.tensor(net.bus.loc[net.bus.index == bus_idx].vn_kv[bus_idx], requires_grad=True)
            delta_iSB = torch.tensor(0.0, dtype=torch.float32,
                                     requires_grad=True) if node_type == "SB" else torch.tensor(
                1.0, dtype=torch.float32, requires_grad=True)

            # Add Core Features to the node features vector
            node_features.append(V_i), node_features.append(delta_iSB)

            # Constraint Features
            # Voltage Magnitudes V_i
            min_V_i = torch.tensor(0.9 * V_i.item(), dtype=torch.float32, requires_grad=True)
            max_V_i = torch.tensor(1.1 * V_i.item(), dtype=torch.float32, requires_grad=True)

            sgen_min_p = sum(net.sgen.loc[net.sgen.bus == bus_idx].p_mw.values) if node_type != "NB" else 0.0
            load_min_p = sum(net.load.loc[net.load.bus == bus_idx].p_mw.values) if node_type != "NB" else 0.0

            sgen_min_q = sum(net.sgen.loc[net.sgen.bus == bus_idx].q_mvar.values) if node_type != "NB" else 0.0
            load_min_q = sum(net.load.loc[net.load.bus == bus_idx].q_mvar.values) if node_type != "NB" else 0.0

            min_p_i = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            max_p_i = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            min_q_i = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            max_q_i = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

            if node_type == "PV":

                # Min Active Power min_p_i

                min_p_i = torch.tensor(-sgen_min_p, dtype=torch.float32, requires_grad=True)

                # Max Active Power max_p_i

                max_p_i = torch.tensor(-load_min_p, dtype=torch.float32,
                                       requires_grad=True)  # PV busses must ensure at least their own load demand is satisfied

                # Min Reactive Power min_q_i

                min_q_i = torch.tensor(-load_min_q, dtype=torch.float32, requires_grad=True)

                # Max Reactive Power max_q_i

                max_q_i = torch.tensor(-sgen_min_q, dtype=torch.float32,
                                       requires_grad=True)  # enforcing self-sustenance

            elif node_type == "PQ":
                # Min Active Power min_p_i

                min_p_i = torch.tensor(load_min_p - sgen_min_p, dtype=torch.float32,
                                       requires_grad=True)  # PQ busses shall at least consume the required remaining demand from their loads, subtracted what they generate

                # Max Active Power max_p_i

                max_p_i = torch.tensor(load_min_p, dtype=torch.float32, requires_grad=True)

                # Min Reactive Power min_q_i

                min_q_i = torch.tensor(load_min_q - sgen_min_q, dtype=torch.float32, requires_grad=True)

                # Max Reactive Power max_q_i

                max_q_i = torch.tensor(load_min_q, dtype=torch.float32, requires_grad=True)  # enforcing self-sustenance

            # Nominal Apparent Power sn_mva

            sn_mva = torch.sqrt(
                torch.square(
                    torch.tensor(max(abs(max_p_i.item()), abs(min_p_i.item())), dtype=torch.float32,
                                 requires_grad=True)) + torch.square(
                    torch.tensor(max(abs(min_q_i.item()), abs(max_q_i.item())), dtype=torch.float32,
                                 requires_grad=True)))

            # External Grid and Gen PVs

            if bus_idx == ext_grid_bus_idx:
                ext_grid_sn_mva = net.trafo.loc[net.trafo.hv_bus == ext_grid_bus_idx].sn_mva.values[0]
                sn_mva = torch.tensor(ext_grid_sn_mva, dtype=torch.float32, requires_grad=True)
                min_p_i = torch.tensor(-sn_mva.item(), dtype=torch.float32, requires_grad=True)
                max_p_i = torch.tensor(sn_mva.item(), dtype=torch.float32, requires_grad=True)
                min_q_i = torch.tensor(-sn_mva.item(), dtype=torch.float32, requires_grad=True)
                max_q_i = torch.tensor(sn_mva.item(), dtype=torch.float32, requires_grad=True)
            elif bus_idx in gen_busses_idx:
                sn_mva = torch.tensor(net.trafo.loc[net.trafo.hv_bus == bus_idx].sn_mva.values[0], dtype=torch.float32,
                                      requires_grad=True)
                min_p_i = torch.tensor(-sn_mva.item(), dtype=torch.float32, requires_grad=True)
                max_p_i = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                min_q_i = torch.tensor(-sn_mva.item(), dtype=torch.float32, requires_grad=True)
                max_q_i = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

            # Flat initialization of P_i based on constraints
            P_i = torch.tensor(0.5 * (max_p_i.item() + min_p_i.item()), dtype=torch.float32,
                               requires_grad=True) if node_type != "NB" else torch.tensor(0.0, dtype=torch.float32,
                                                                                          requires_grad=True)

            # Flat initialization of Q_i based on constraints
            Q_i = torch.tensor(0.5 * (max_q_i.item() + min_q_i.item()), dtype=torch.float32,
                               requires_grad=True) if node_type != "NB" else torch.tensor(0.0, dtype=torch.float32,
                                                                                          requires_grad=True)

            # Add the remaining core features to the node feature vector
            node_features.append(P_i)
            node_features.append(Q_i)

            # Add the constraint features to the node feature vector
            node_features.append(min_V_i), node_features.append(max_V_i), node_features.append(sn_mva)
            node_features.append(min_p_i), node_features.append(max_p_i), node_features.append(min_q_i),
            node_features.append(max_q_i)

            # Add the node feature vector to the collection of feature vectors of the corresponding node type
            X_node_type.append(torch.stack(node_features))

        X_node_type = torch.stack(X_node_type)
        # Add the collection of node features to the corresponding node of Hetero Data
        data[node_type].x = X_node_type  # torch.tensor(X_node_type, dtype=torch.float32)

    # Add the edge indices and edge attributes to the Hetero Data
    for edge_type in edge_types_idx_dict:
        bus_names = edge_type.split("-")
        from_bus, to_bus = bus_names
        edge = (from_bus, "isConnected", to_bus)
        data[edge].edge_index = edge_types_idx_dict[edge_type]
        data[edge].edge_attr = edge_types_attr_dict[edge_type]

    if not suppress_info:
        print("Hetero Data Input has been generated.")
    return index_mappers, net, data


def extract_unsupervised_inputs(data: HeteroData, net, index_mappers):
    x_dict = dict()
    constraint_dict = dict()
    edge_idx_dict = dict()
    edge_attr_dict = dict()
    bus_idx_neighbors_dict = dict()
    """
    # x_dict and constraint_dict
    for node_type in data.node_types:
        x: torch.Tensor
        c: torch.Tensor
        scaler = MinMaxScaler()
        if len(data[node_type]) != 0:
            c = scaler.fit_transform(data[node_type].x[:, 4:].detach().numpy())
            c = 2 * c - 1  # Scale between -1 and 1
            if torch.isnan(torch.tensor(c)).any():
                print(f"nan value found in node_type {node_type} in c")
            c = torch.tensor(c, dtype=torch.float32, requires_grad=False)
            x, min_angle, max_angle = custom_minmax_transform(scaler, data[node_type].x[:, :4].detach().numpy())

            if node_type == "SB":
                x[0, 1] = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

            if torch.isnan(torch.tensor(x)).any():
                print(f"nan value found in node_type {node_type} in x")
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            x_dict[node_type] = x
            constraint_dict[node_type] = c

            scalers_dict[node_type] = scaler
            angle_params_dict[node_type] = (min_angle, max_angle)
    """
    node_type_len_mapper = dict()
    constraints = []
    features = []
    scaler = StandardScaler()  # MinMaxScaler(feature_range=(-1, 1))

    for node_type in data.node_types:
        # x = data[node_type].x
        x = data[node_type].x[:, :4].detach().numpy()
        c = data[node_type].x[:, 4:].detach().numpy()
        node_type_len_mapper[node_type] = (len(features), len(features) + len(x))
        features.extend(x)
        constraints.extend(c)

    # features = [feature.detach().numpy() for feature in features]

    try:
        pp.runopp(net)
        res_bus = net.res_bus
    except:
        res_bus = None

    if res_bus is not None:
        res_bus = res_bus
        res_bus = res_bus.drop(labels=['lam_p', 'lam_q'], axis=1)
        res_bus = np.array(res_bus)
        res_bus[:, 0] *= net.bus.vn_kv.values

        # Combine X and Y for fitting
        combined_data = np.vstack((features, res_bus))

        # Fit the scaler on the combined data
        scaler.fit(combined_data)

        # Transform X and Y
        features = scaler.transform(features)
        res_bus = scaler.transform(res_bus)
        constraints = custom_standard_transform(scaler, np.array(constraints))
        # res_bus = 2 * scaler.fit_transform(res_bus) - 1

        idx_mapper, node_types_idx_dict = index_mappers

        node_type_idx_mapper = dict()
        for node_type in node_types_idx_dict:
            length = len(node_types_idx_dict[node_type])
            for bus_idx, map_idx in zip(node_types_idx_dict[node_type], range(length)):
                real_idx = idx_mapper[bus_idx]
                node_type_idx_mapper[real_idx] = map_idx

        res_bus_dict = dict()
        for node_type in node_types_idx_dict:
            node_indices = node_types_idx_dict[node_type]
            idx = [idx_mapper[i] for i in node_indices]
            res_bus_dict[node_type] = torch.tensor(res_bus[idx], dtype=torch.float32, requires_grad=False)
    else:
        try:
            pp.runpm_ac_opf(net)
            res_bus = net.res_bus
        except:
            res_bus = None

        if res_bus is not None:
            res_bus = res_bus.drop(labels=['lam_p', 'lam_q'], axis=1)
            res_bus = np.array(res_bus)
            res_bus[:, 0] *= net.bus.vn_kv.values
            # Combine X and Y for fitting
            combined_data = np.vstack((features, res_bus))

            # Fit the scaler on the combined data
            scaler.fit(combined_data)

            # Transform X and Y
            features = scaler.transform(features)
            res_bus = scaler.transform(res_bus)
            constraints = custom_standard_transform(scaler, np.array(constraints))
            # res_bus = 2 * scaler.fit_transform(res_bus) - 1

            idx_mapper, node_types_idx_dict = index_mappers

            node_type_idx_mapper = dict()
            for node_type in node_types_idx_dict:
                length = len(node_types_idx_dict[node_type])
                for bus_idx, map_idx in zip(node_types_idx_dict[node_type], range(length)):
                    real_idx = idx_mapper[bus_idx]
                    node_type_idx_mapper[real_idx] = map_idx

            res_bus_dict = dict()
            for node_type in node_types_idx_dict:
                node_indices = node_types_idx_dict[node_type]
                idx = [idx_mapper[i] for i in node_indices]
                res_bus_dict[node_type] = torch.tensor(res_bus[idx], dtype=torch.float32, requires_grad=False)

    # constraints = scaler.fit_transform(np.array(constraints))
    # constraints = 2 * constraints - 1
    # features, min_angle, max_angle = custom_minmax_transform(scaler, np.array(features))
    angle_params = (None, None)

    for node_type in data.node_types:
        start_idx, end_idx = node_type_len_mapper[node_type]
        core_features = features[start_idx:end_idx, :]
        constraint_features = constraints[start_idx:end_idx, :]
        x_dict[node_type] = torch.tensor(core_features, dtype=torch.float32, requires_grad=True)
        constraint_dict[node_type] = torch.tensor(constraint_features, dtype=torch.float32, requires_grad=True)

    # edge_idx_dict and edge_attr_dict
    for edge_type in data.edge_types:
        if data[edge_type].edge_attr.numel() != 0:
            edge_idx_dict[edge_type] = data[edge_type].edge_index
            edge_attr_dict[edge_type] = data[edge_type].edge_attr

    # bus index to neighbor(s) mapper
    # Add each node type as a dict
    for node_type in x_dict:
        bus_idx_neighbors_dict[node_type] = dict()

    for edge_type, edge_idx in edge_idx_dict.items():
        from_bus, _, to_bus = edge_type
        for i, from_edge, to_edge in zip(range(len(edge_idx[0])), edge_idx[0], edge_idx[1]):
            pair = (to_bus, to_edge.item(), edge_attr_dict[edge_type][i])
            if from_edge.item() in bus_idx_neighbors_dict[from_bus]:
                bus_idx_neighbors_dict[from_bus][from_edge.item()].append(pair)
            else:
                bus_idx_neighbors_dict[from_bus][from_edge.item()] = [pair]

    return x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, bus_idx_neighbors_dict, scaler, angle_params, res_bus_dict


def save_unsupervised_inputs(grid_name: str, num_samples: int):
    path = "../code/data/Heterogeneous/" + grid_name
    inputs = []  # List of ACOPFData instances
    for i in range(num_samples):
        print(f"Epoch: {i}")
        index_mappers, network, data = generate_unsupervised_input(grid_name)
        _input_ = ACOPFInput(data, network, index_mappers)
        inputs.append(_input_)

    print("Data Prep finished.")
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, 'inputs.pkl'), 'wb') as f:
        pickle.dump(inputs, f)

    print("Inputs Saved.")


def save_multiple_unsupervised_inputs(grid_names, num_samples):
    path = "../code/data/Heterogeneous/"
    inputs = []  # List of ACOPFData instances
    for grid_name in grid_names:
        print(f"Grid: {grid_name}")
        i = 0
        while i < num_samples:
            print(f"Epoch: {i}")
            i += 1
            index_mappers, network, data = generate_unsupervised_input(grid_name)
            _input_ = ACOPFInput(data, network, index_mappers)
            if _input_.res_bus is None:
                i -= 1
                continue
            else:
                inputs.append(_input_)

    print("Data Prep finished.")
    os.makedirs(path, exist_ok=True)

    random.shuffle(inputs)

    with open(os.path.join(path, 'inputs.pkl'), 'wb') as f:
        pickle.dump(inputs, f)

    print("Inputs Saved.")


def load_multiple_unsupervised_inputs():
    path = "../code/data/Heterogeneous/"
    with open(os.path.join(path, 'inputs.pkl'), 'rb') as f:
        inputs = pickle.load(f)

    return inputs


def load_unsupervised_inputs(grid_name: str):
    path = "../code/data/Heterogeneous/" + grid_name
    with open(os.path.join(path, 'inputs.pkl'), 'rb') as f:
        inputs = pickle.load(f)

    return inputs


def save_unsupervised_output(acopfoutput: ACOPFOutput, grid_name: str):
    path = "../code/data/Heterogeneous/" + grid_name
    with open(os.path.join(path, 'output.pkl'), 'wb') as f:
        pickle.dump(acopfoutput, f)

    print("Output Saved.")


def load_unsupervised_output(grid_name: str):
    path = "../code/data/Heterogeneous/" + grid_name
    with open(os.path.join(path, 'output.pkl'), 'rb') as f:
        output = pickle.load(f)

    return output


def self_supervised_hetero_obj_fn(out_dict, bus_idx_neighbors_dict, constraints_dict,
                                  scaler: StandardScaler, alpha, beta, gamma):
    unsupervised_loss = calc_unsupervised_loss(out_dict, scaler)
    physics_loss = calc_physics_loss(out_dict, bus_idx_neighbors_dict, scaler)
    constraint_loss = calc_constraint_loss(out_dict, constraints_dict, scaler)

    Loss = alpha * unsupervised_loss + beta * physics_loss + gamma * constraint_loss

    return physics_loss, torch.tensor(0.0), Loss, constraint_loss, unsupervised_loss


def self_supervised_enforcer_obj_fn(out_dict, constraints_dict, scaler):
    constraint_loss = calc_constraint_loss(out_dict, constraints_dict, scaler)

    return constraint_loss


def self_supervised_minimizer_obj_fn(out_dict, scaler):
    unsupervised_loss = calc_unsupervised_loss(out_dict, scaler)

    return unsupervised_loss


def self_supervised_embedder_obj_fn(out_dict,powers_dict, bus_idx_neighbors_dict, scaler):
    physics_loss = calc_physics_loss(out_dict,powers_dict, bus_idx_neighbors_dict, scaler)

    return physics_loss


def load_homogeneous_supervised_model(in_channels=4, hidden_channels=256, out_channels=4, activation="elu",
                                      num_layers=5, dropout=0.0, jk="last", layer_type="TransConv"):
    # Create the GNN Model
    model = GNN(in_channels, hidden_channels, num_layers, out_channels, dropout=dropout,
                norm=torch_geometric.nn.norm.batch_norm.BatchNorm(hidden_channels),
                jk=jk, layer_type=layer_type, activation=activation)

    # Load the saved model parameter
    ordered_dict = torch.load(r"C:\Users\canba\OneDrive\Masaüstü\OPFGNN\code\Models\Supervised\supervisedmodel.pt")
    model.load_state_dict(ordered_dict)

    return model


def create_ACOPFGNN_model(init_data, net, index_mappers, hidden_channels=4096, out_channels=4, num_layers=5,
                          dropout=0.0, act_fn="elu"):
    hetero_model = ACOPFGNN(hidden_channels, out_channels, num_layers, dropout, act_fn, init_data=init_data)

    # Lazy Initialize Parameters
    hetero_model.lazy_init(init_data, net, index_mappers)

    return hetero_model


def create_ACOPFEnforcer_model(init_data, net, index_mappers, hidden_channels=32, out_channels=4, num_layers=1,
                               dropout=0.0, act_fn="elu"):
    hetero_model = ACOPFEnforcer(hidden_channels, out_channels, num_layers, dropout, act_fn, init_data=init_data)

    # Lazy Initialize Parameters
    hetero_model.lazy_init(init_data, net, index_mappers)

    return hetero_model


def create_ACOPFEmbedder_model(init_data, net, index_mappers, hidden_channels=32, out_channels=4, num_layers=1,
                               dropout=0.0, act_fn="elu"):
    hetero_model = ACOPFEmbedder(hidden_channels, out_channels, num_layers, dropout, act_fn, init_data=init_data)

    # Lazy Initialize Parameters
    hetero_model.lazy_init(init_data, net, index_mappers)

    return hetero_model


def train_validate_ACOPF(model: ACOPFGNN, model_name: str, optimizer: torch.optim.AdamW, train_inputs: List[ACOPFInput],
                         val_inputs: List[ACOPFInput], loss_weights: Tuple, start_epoch=0, num_epochs: int = 100,
                         return_outputs: bool = True, save_model: bool = True) -> Tuple[Any, Any, Any, Any]:
    train_loss = 0.0
    train_unsupervised_loss = 0.0
    train_mre_loss = 0.0
    train_physics_loss = 0.0
    train_penalty_loss = 0.0
    val_loss = 0.0
    val_mre_loss = 0.0
    val_physics_loss = 0.0
    val_penalty_loss = 0.0
    val_unsupervised_loss = 0.0
    alpha, beta, gamma = loss_weights

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=15)

    torch.autograd.set_detect_anomaly(True)
    # torch.autograd.gradcheck()
    train_output = None
    train_input = None
    val_output = None
    val_input = None
    prev_lr = optimizer.param_groups[0]['lr']
    print_lr(prev_lr)

    for i in range(start_epoch - 1, num_epochs):
        print(f"Epoch: {i}")
        print("###########################################")
        print("   TRAINING")

        # Shuffle the inputs for both Training and Validation
        random.shuffle(train_inputs)
        random.shuffle(val_inputs)

        # TRAINING

        for j, ACOPFinput in enumerate(train_inputs):

            # Store the attributes in variables to be used later on
            x_dict = ACOPFinput.x_dict
            constraint_dict = ACOPFinput.constraint_dict
            edge_idx_dict = ACOPFinput.edge_idx_dict
            edge_attr_dict = ACOPFinput.edge_attr_dict
            bus_idx_neighbors_dict = ACOPFinput.bus_idx_neighbors_dict
            net = ACOPFinput.net
            scaler = ACOPFinput.scaler
            index_mappers = ACOPFinput.index_mappers
            angle_params = ACOPFinput.angle_params
            res_bus = ACOPFinput.res_bus

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            out_dict = model(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, scaler)

            # Compute the loss and its gradients for RMSE loss
            physics_loss, mre_loss, loss, penalty_loss, unsupervised_loss = self_supervised_hetero_obj_fn(out_dict,
                                                                                                          bus_idx_neighbors_dict,
                                                                                                          res_bus,
                                                                                                          constraint_dict,
                                                                                                          scaler, alpha,
                                                                                                          beta, gamma)
            loss.backward()

            """
            # After loss.backward()
            for i, param in enumerate(model.parameters()):
                if param.requires_grad:
                    print(f'param_{i}', param.grad)
                    if param.data is not None:
                        print(f"param data: {param.data}")"""

            # Adjust learning weights
            optimizer.step()

            # Store the outputs at the last iteration and last input
            if return_outputs and i == num_epochs - 1 and j == len(train_inputs) - 1:
                train_input = ACOPFInput
                train_output = ACOPFOutput(out_dict, scaler, net, index_mappers, angle_params, res_bus)

            # Gather data and report
            train_loss += loss.item()
            train_mre_loss += mre_loss.item()
            train_physics_loss += physics_loss.item()
            train_penalty_loss += penalty_loss.item()
            train_unsupervised_loss += unsupervised_loss.item()

            print(f"     Training Step: {j} Training Loss: {loss.item()} ")

        # Validation
        print("###########################################")
        print("   VALIDATION")

        # VALIDATION

        for j, ACOPFinput in enumerate(val_inputs):

            # Store the attributes in variables to be used later on
            x_dict = ACOPFinput.x_dict
            constraint_dict = ACOPFinput.constraint_dict
            edge_idx_dict = ACOPFinput.edge_idx_dict
            edge_attr_dict = ACOPFinput.edge_attr_dict
            bus_idx_neighbors_dict = ACOPFinput.bus_idx_neighbors_dict
            net = ACOPFinput.net
            scaler = ACOPFinput.scaler
            index_mappers = ACOPFinput.index_mappers
            angle_params = ACOPFinput.angle_params
            res_bus = ACOPFinput.res_bus

            # Make predictions for this batch
            out_dict = model(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, scaler)

            # Compute the loss and its gradients for RMSE loss
            physics_loss, mre_loss, loss, penalty_loss, unsupervised_loss = self_supervised_hetero_obj_fn(out_dict,
                                                                                                          bus_idx_neighbors_dict,
                                                                                                          res_bus,
                                                                                                          constraint_dict,
                                                                                                          scaler, alpha,
                                                                                                          beta, gamma)

            # Store the outputs at the last iteration and last input
            if return_outputs and i == num_epochs - 1 and j == len(val_inputs) - 1:
                train_input = ACOPFinput
                val_output = ACOPFOutput(out_dict, scaler, net, index_mappers, angle_params, res_bus)

            # Gather data and report
            val_loss += loss.item()
            val_mre_loss += mre_loss.item()
            val_physics_loss += physics_loss.item()
            val_penalty_loss += penalty_loss.item()
            val_unsupervised_loss += unsupervised_loss.item()

            print(f"     Validation Step: {j} Validation Loss: {loss.item()} ")

        total_train_loss = train_loss / len(train_inputs)
        total_train_mre_loss = train_mre_loss / len(train_inputs)
        total_train_physics_loss = train_physics_loss / len(train_inputs)
        total_train_penalty_loss = train_penalty_loss / len(train_inputs)
        total_train_unsupervised_loss = train_unsupervised_loss / len(train_inputs)
        total_val_loss = val_loss / len(val_inputs)
        total_val_mre_loss = val_mre_loss / len(val_inputs)
        total_val_physics_loss = val_physics_loss / len(val_inputs)
        total_val_penalty_loss = val_penalty_loss / len(val_inputs)
        total_val_unsupervised_loss = val_unsupervised_loss / len(val_inputs)

        # Log the losses to wandb run
        wandb.log({
            'epoch': i,
            'train_loss': total_train_loss,
            'train_mre_loss': total_train_mre_loss,
            'train_physics_loss': total_train_physics_loss,
            'train_penalty_loss': total_train_penalty_loss,
            'train_unsupervised_loss': total_train_unsupervised_loss,
            'val_loss': total_val_loss,
            'val_mre_loss': total_val_mre_loss,
            'val_physics_loss': total_val_physics_loss,
            'val_penalty_loss': total_val_penalty_loss,
            'val_unsupervised_loss': total_val_unsupervised_loss,
        })

        # Adjust the learning Rate based on Validation Loss
        lr_scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        if prev_lr != lr:
            print_lr(lr, first_prompt=False)
            prev_lr = lr

        alpha = 1 / total_train_unsupervised_loss if alpha > 0.0 else 0.0

        beta = 1 / total_train_physics_loss if beta > 0.0 else 0.0

        gamma = 1 / total_train_penalty_loss if gamma > 0.0 else 0.0

        """
        if alpha_enabled:
            alpha = scale_a / (scale_a + scale_b + scale_g)
        if beta_enabled:
            beta = scale_b / (scale_a + scale_b + scale_g)
        if gamma_enabled:
            gamma = scale_g / (scale_a + scale_b + scale_g)
        """

        # Reset the metrics
        train_loss = 0.0
        train_mre_loss = 0.0
        train_physics_loss = 0.0
        train_penalty_loss = 0.0
        train_unsupervised_loss = 0.0
        val_loss = 0.0
        val_mre_loss = 0.0
        val_physics_loss = 0.0
        val_penalty_loss = 0.0
        val_unsupervised_loss = 0.0

    # Save the Model
    if save_model:
        try:
            save_ACOPFGNN_model(model, model_name)
        except:
            print("Model cant be saved.")

    return train_input, train_output, val_input, val_output


def train_validate_ACOPF_chained(minimizer_model: ACOPFGNN, enforcer_model: ACOPFEnforcer, embedder_model: ACOPFGNN,
                                 ACOPF_optimizer: torch.optim.Adam, train_inputs: List[ACOPFInput],
                                 val_inputs: List[ACOPFInput], loss_weights: Tuple, start_epoch=0,
                                 num_epochs: int = 100,
                                 return_outputs: bool = True, save_model: bool = True) -> Tuple[Any, Any, Any, Any]:
    train_loss = 0.0
    train_unsupervised_loss = 0.0
    train_physics_loss = 0.0
    train_penalty_loss = 0.0
    val_loss = 0.0
    val_physics_loss = 0.0
    val_penalty_loss = 0.0
    val_unsupervised_loss = 0.0
    alpha, beta, gamma = loss_weights

    torch.autograd.set_detect_anomaly(True)
    # torch.autograd.gradcheck()
    train_output = None
    train_input = None
    val_output = None
    val_input = None

    for i in range(start_epoch - 1, num_epochs):
        print(f"Epoch: {i}")
        print("###########################################")
        print("   TRAINING")

        # Shuffle the inputs for both Training and Validation
        random.shuffle(train_inputs)
        random.shuffle(val_inputs)

        # TRAINING

        for j, ACOPFinput in enumerate(train_inputs):

            # Store the attributes in variables to be used later on
            x_dict = ACOPFinput.x_dict
            constraint_dict = ACOPFinput.constraint_dict
            edge_idx_dict = ACOPFinput.edge_idx_dict
            edge_attr_dict = ACOPFinput.edge_attr_dict
            bus_idx_neighbors_dict = ACOPFinput.bus_idx_neighbors_dict
            net = ACOPFinput.net
            scaler = ACOPFinput.scaler
            index_mappers = ACOPFinput.index_mappers
            angle_params = ACOPFinput.angle_params
            res_bus = ACOPFinput.res_bus


            # Zero gradients for each optimizer
            ACOPF_optimizer.zero_grad()

            # Forward pass with minimizer
            # out_dict = minimizer_model(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict)

            # Compute minimization loss
            unsupervised_loss = torch.tensor(0.0)  # self_supervised_minimizer_obj_fn(out_dict, scaler)

            out_dict = embedder_model(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, scaler)

            physics_loss = self_supervised_embedder_obj_fn(x_dict,out_dict, bus_idx_neighbors_dict, scaler)

            # Forward pass with enforcer
            # out_dict = enforcer_model(out_dict, constraint_dict, edge_idx_dict, edge_attr_dict, scaler)

            # Compute constraint loss
            penalty_loss = torch.tensor(0.0)  # self_supervised_enforcer_obj_fn(out_dict, constraint_dict, scaler)

            # Compute Total with weightings
            loss = alpha * unsupervised_loss + beta * physics_loss + gamma * penalty_loss
            # loss = beta * physics_loss + gamma * penalty_loss
            loss.backward()
            ACOPF_optimizer.step()

            # Store the outputs at the last iteration and last input
            if return_outputs and i == num_epochs - 1 and j == len(train_inputs) - 1:
                train_input = ACOPFInput
                train_output = ACOPFOutput(out_dict, scaler, net, index_mappers, angle_params, res_bus)

            # Gather data and report
            train_loss += loss.item()
            train_physics_loss += physics_loss.item()
            train_penalty_loss += penalty_loss.item()
            train_unsupervised_loss += unsupervised_loss.item()

            print(f"     Training Step: {j} Training Loss: {loss.item()} ")

        # Validation
        print("###########################################")
        print("   VALIDATION")

        # VALIDATION

        for j, ACOPFinput in enumerate(val_inputs):

            # Store the attributes in variables to be used later on
            x_dict = ACOPFinput.x_dict
            constraint_dict = ACOPFinput.constraint_dict
            edge_idx_dict = ACOPFinput.edge_idx_dict
            edge_attr_dict = ACOPFinput.edge_attr_dict
            bus_idx_neighbors_dict = ACOPFinput.bus_idx_neighbors_dict
            net = ACOPFinput.net
            scaler = ACOPFinput.scaler
            index_mappers = ACOPFinput.index_mappers
            angle_params = ACOPFinput.angle_params
            res_bus = ACOPFinput.res_bus


            # Forward pass with minimizer
            # out_dict = minimizer_model(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict)

            # Compute minimization loss
            unsupervised_loss = torch.tensor(0.0)  # self_supervised_minimizer_obj_fn(out_dict, scaler)

            out_dict = embedder_model(x_dict, constraint_dict, edge_idx_dict, edge_attr_dict, scaler)

            physics_loss = self_supervised_embedder_obj_fn(x_dict, out_dict, bus_idx_neighbors_dict, scaler)

            # Forward pass with enforcer
            # out_dict = enforcer_model(out_dict, constraint_dict, edge_idx_dict, edge_attr_dict, scaler)

            # Compute constraint loss
            penalty_loss = torch.tensor(0.0)  # self_supervised_enforcer_obj_fn(out_dict, constraint_dict, scaler)

            # Compute Total with weightings
            loss = alpha * unsupervised_loss + beta * physics_loss + gamma * penalty_loss
            # loss = beta * physics_loss + gamma * penalty_loss

            # Store the outputs at the last iteration and last input
            if return_outputs and i == num_epochs - 1 and j == len(val_inputs) - 1:
                train_input = ACOPFinput
                val_output = ACOPFOutput(out_dict, scaler, net, index_mappers, angle_params, res_bus)

            # Gather data and report
            val_loss += loss.item()
            val_physics_loss += physics_loss.item()
            val_penalty_loss += penalty_loss.item()
            val_unsupervised_loss += unsupervised_loss.item()

            print(f"     Validation Step: {j} Validation Loss: {loss.item()} ")

        total_train_loss = train_loss / len(train_inputs)
        total_train_physics_loss = train_physics_loss / len(train_inputs)
        total_train_penalty_loss = train_penalty_loss / len(train_inputs)
        total_train_unsupervised_loss = train_unsupervised_loss / len(train_inputs)
        total_val_loss = val_loss / len(val_inputs)
        total_val_physics_loss = val_physics_loss / len(val_inputs)
        total_val_penalty_loss = val_penalty_loss / len(val_inputs)
        total_val_unsupervised_loss = val_unsupervised_loss / len(val_inputs)

        # Log the losses to wandb run
        wandb.log({
            'epoch': i,
            'train_loss': total_train_loss,
            'train_physics_loss': total_train_physics_loss,
            'train_penalty_loss': total_train_penalty_loss,
            'train_unsupervised_loss': total_train_unsupervised_loss,
            'val_loss': total_val_loss,
            'val_physics_loss': total_val_physics_loss,
            'val_penalty_loss': total_val_penalty_loss,
            'val_unsupervised_loss': total_val_unsupervised_loss
        })

        # Adjust the learning Rate based on Validation Loss

        # alpha = 1 / total_train_unsupervised_loss if alpha > 0.0 else 0.0

        # beta = 1 / total_train_physics_loss if beta > 0.0 else 0.0

        # gamma = 1 / total_train_penalty_loss if gamma > 0.0 else 0.0

        """
        if alpha_enabled:
            alpha = scale_a / (scale_a + scale_b + scale_g)
        if beta_enabled:
            beta = scale_b / (scale_a + scale_b + scale_g)
        if gamma_enabled:
            gamma = scale_g / (scale_a + scale_b + scale_g)
        """

        # Reset the metrics
        train_loss = 0.0
        train_physics_loss = 0.0
        train_penalty_loss = 0.0
        train_unsupervised_loss = 0.0
        val_loss = 0.0
        val_physics_loss = 0.0
        val_penalty_loss = 0.0
        val_unsupervised_loss = 0.0

    # Save the Model
    if save_model:
        # try:
        # save_ACOPFGNN_model(minimizer_model, "minimizer_model.pt")
        # except:
        # print("Model cant be saved.")

        try:
            save_ACOPFGNN_model(embedder_model, "embedder_model_specialized.pt")
        except:
            print("Model cant be saved.")

        # try:
        # save_ACOPFGNN_model(enforcer_model, "enforcer_model.pt")
        # except:
        # print("Model cant be saved.")

    return train_input, train_output, val_input, val_output


def custom_standard_transform(scaler, constraint_features):
    # Core feature columns: V_i, delta_iSB, P_i, Q_i
    means = scaler.mean_  # mean
    standard_deviations = scaler.scale_  # standard deviation

    # Transforming min and max voltage magnitude constraints
    # according to the fitted values on the core feature: V_i
    constraint_features[:, :2] = (constraint_features[:, :2] - means[0]) / standard_deviations[0]

    # Transforming min and max active power constraints
    # according to the fitted values on the core feature: P_i
    constraint_features[:, 3:5] = (constraint_features[:, 3:5] - means[2]) / standard_deviations[2]

    # Transforming min and max active power constraints
    # according to the fitted values on the core feature: P_i
    constraint_features[:, 5:] = (constraint_features[:, 5:] - means[3]) / standard_deviations[3]

    # Transforming max apparent Power S constraints
    # according to the fitted values on the average of core feature: P_i and Q_i
    # mean = np.sqrt(means[2]**2 + means[3]**2)
    # mean = (means[2] + means[3])*0.5
    # mean = min(means[2], means[3])
    # standard_deviation = min(standard_deviations[2],standard_deviations[3])
    # standard_deviation = (standard_deviations[2] + standard_deviations[3])*0.5
    # standard_deviation = np.sqrt(standard_deviations[2]**2 + standard_deviations[3]**2)
    S = []
    for i in range(np.shape(constraint_features)[0]):
        P_min_abs = np.abs(constraint_features[i, 3])
        P_max_abs = np.abs(constraint_features[i, 4])
        Q_min_abs = np.abs(constraint_features[i, 5])
        Q_max_abs = np.abs(constraint_features[i, 6])

        max_p = max(P_min_abs, P_max_abs)
        max_q = max(Q_min_abs, Q_max_abs)

        S_val = np.sqrt(max_p ** 2 + max_q ** 2)
        S.append(S_val)

    constraint_features[:, 2] = np.array(S)

    return constraint_features


def custom_standard_inverse_transform(scaler: StandardScaler, constraint_features):
    # Core feature columns: V_i, delta_iSB, P_i, Q_i
    means = scaler.mean_  # mean
    standard_deviations = scaler.scale_  # standard deviation

    # Inverse Transforming min and max voltage magnitude constraints
    # according to the fitted values on the core feature: V_i
    constraint_features[:, :2] = constraint_features[:, :2] * standard_deviations[0] + means[0]

    # Inverse Transforming max apparent Power S constraints
    # according to the fitted values on the average of core feature: P_i and Q_i
    mean = np.sqrt(means[2] ** 2 + means[3] ** 2)
    standard_deviation = np.sqrt(standard_deviations[2] ** 2 + standard_deviations[3] ** 2)
    constraint_features[:, 2] = constraint_features[:, 2] * standard_deviation + mean

    # Inverse Transforming min and max active power constraints
    # according to the fitted values on the core feature: P_i
    constraint_features[:, 3:5] = constraint_features[:, 3:5] * standard_deviations[2] + means[2]

    # Inverse Transforming min and max active power constraints
    # according to the fitted values on the core feature: P_i
    constraint_features[:, 5:] = constraint_features[:, 5:] * standard_deviations[3] + means[3]

    return constraint_features


def custom_constraint_robust_transform(scaler: RobustScaler, constraints):
    # Extract attributes from the RobustScaler instance
    centering = scaler.center_
    scaling = scaler.scale_

    # Map indices based on your mentioned columns
    v_index = 0
    p_index = 2
    q_index = 3

    # Scale V_min and V_max
    constraints[:, 0] = (constraints[:, 0] - centering[v_index]) / scaling[v_index]
    constraints[:, 1] = (constraints[:, 1] - centering[v_index]) / scaling[v_index]

    # Scale P_min and P_max
    constraints[:, 3] = (constraints[:, 3] - centering[p_index]) / scaling[p_index]
    constraints[:, 4] = (constraints[:, 4] - centering[p_index]) / scaling[p_index]

    # Scale Q_min and Q_max
    constraints[:, 5] = (constraints[:, 5] - centering[q_index]) / scaling[q_index]
    constraints[:, 6] = (constraints[:, 6] - centering[q_index]) / scaling[q_index]

    # Scale S using a combination of P and Q's median and scale
    combined_median = np.sqrt(centering[p_index] ** 2 + centering[q_index] ** 2)
    combined_scale = np.sqrt(scaling[p_index] ** 2 + scaling[q_index] ** 2)

    constraints[:, 2] = (constraints[:, 2] - combined_median) / combined_scale

    return constraints


def custom_minmax_constraints_transform(scaler: MinMaxScaler, constraints):
    # Constraint feature columns: min_V_i, max_V_i, max_S, min_P_i, max_P_i, min_Q_i, max_Q_i
    max_values = scaler.data_max_
    min_values = scaler.data_min_

    # Transform the Voltage Magnitude Columns min_V_i and max_V_i
    _min_ = min_values[0]
    _max_ = max_values[0]

    constraints[:, 0] = (constraints[:, 0] - _min_) / (_max_ - _min_)
    constraints[:, 1] = (constraints[:, 1] - _min_) / (_max_ - _min_)

    # Transform the Active Power Columns min_P_i, max_P_i
    _min_ = min_values[2]
    _max_ = max_values[2]

    if _max_ - _min_ == 0:
        constraints[:, 3] = np.zeros_like(constraints[:, 3])
        constraints[:, 4] = np.zeros_like(constraints[:, 4])
    else:
        constraints[:, 3] = (constraints[:, 3] - _min_) / (_max_ - _min_)
        constraints[:, 4] = (constraints[:, 4] - _min_) / (_max_ - _min_)

    # Transform the Reactive Power Columns min_Q_i, max_Q_i
    _min_ = min_values[3]
    _max_ = max_values[3]

    if _max_ - _min_ == 0:
        constraints[:, 5] = np.zeros_like(constraints[:, 5])
        constraints[:, 6] = np.zeros_like(constraints[:, 6])
    else:
        constraints[:, 5] = (constraints[:, 5] - _min_) / (_max_ - _min_)
        constraints[:, 6] = (constraints[:, 6] - _min_) / (_max_ - _min_)

    # Transform the max S column
    min_p = min(abs(min_values[2]), abs(max_values[2]))
    max_p = max(abs(min_values[2]), abs(max_values[2]))
    min_q = min(abs(min_values[3]), abs(max_values[3]))
    max_q = max(abs(min_values[3]), abs(max_values[3]))

    _min_ = np.sqrt(np.square(min_p) + np.square(min_q))
    _max_ = np.sqrt(np.square(max_p) + np.square(max_q))

    constraints[:, 2] = (constraints[:, 2] - _min_) / (_max_ - _min_)
    return constraints


def custom_inverse_minmax_constraints_transform(scaler: MinMaxScaler, constraints):
    # Constraint feature columns: min_V_i, max_V_i, max_S, min_P_i, max_P_i, min_Q_i, max_Q_i
    max_values = scaler.data_max_
    min_values = scaler.data_min_

    # Transform the Voltage Magnitude Columns min_V_i and max_V_i
    _min_ = min_values[0]
    _max_ = max_values[0]

    constraints[:, 0] = constraints[:, 0] * (_max_ - _min_) + _min_
    constraints[:, 1] = constraints[:, 1] * (_max_ - _min_) + _min_

    # Transform the Active Power Columns min_P_i, max_P_i
    _min_ = min_values[2]
    _max_ = max_values[2]

    if _max_ - _min_ == 0:
        constraints[:, 3] = np.zeros_like(constraints[:, 3])
        constraints[:, 4] = np.zeros_like(constraints[:, 4])
    else:
        constraints[:, 3] = constraints[:, 3] * (_max_ - _min_) + _min_
        constraints[:, 4] = constraints[:, 4] * (_max_ - _min_) + _min_

    # Transform the Reactive Power Columns min_Q_i, max_Q_i
    _min_ = min_values[3]
    _max_ = max_values[3]

    if _max_ - _min_ == 0:
        constraints[:, 5] = np.zeros_like(constraints[:, 5])
        constraints[:, 6] = np.zeros_like(constraints[:, 6])
    else:
        constraints[:, 5] = constraints[:, 5] * (_max_ - _min_) + _min_
        constraints[:, 6] = constraints[:, 6] * (_max_ - _min_) + _min_

    # Transform the max S column
    min_p = min(abs(min_values[2]), abs(max_values[2]))
    max_p = max(abs(min_values[2]), abs(max_values[2]))
    min_q = min(abs(min_values[3]), abs(max_values[3]))
    max_q = max(abs(min_values[3]), abs(max_values[3]))

    _min_ = np.sqrt(np.square(min_p) + np.square(min_q))
    _max_ = np.sqrt(np.square(max_p) + np.square(max_q))

    constraints[:, 2] = constraints[:, 2] * (_max_ - _min_) + _min_
    return constraints


def custom_minmax_transform(scaler: MinMaxScaler, core_features):
    # Constraint feature columns: min_V_i, max_V_i, max_S, min_P_i, max_P_i, min_Q_i, max_Q_i
    max_values = scaler.data_max_
    min_values = scaler.data_min_

    # Transform the Voltage Magnitude Column V_i of the Core Features based on min_V_i, max_V_i
    _min_ = min(min_values[0], min_values[1])
    _max_ = max(max_values[0], max_values[1])
    core_features[:, 0] = 2 * ((core_features[:, 0] - _min_) / (_max_ - _min_)) - 1

    # Transform the voltage angle column delta_i of the core features based on the percentile of min_V_i and max_V_i
    min_angle = min(core_features[:, 1])
    max_angle = max(core_features[:, 1])
    core_features[:, 1] = 2 * ((core_features[:, 1] - min_angle) / (max_angle - min_angle)) - 1

    # Transform the Active Power Column P_i of the Core Features based on min_P_i, max_P_i
    _min_ = min(min_values[3], min_values[4])
    _max_ = max(max_values[3], max_values[4])
    if _max_ - _min_ == 0:
        core_features[:, 2] = np.zeros_like(core_features[:, 2])
    else:
        core_features[:, 2] = 2 * ((core_features[:, 2] - _min_) / (_max_ - _min_)) - 1

    # if np.isnan(core_features[:, 2]).any():
    # core_features[:, 2] = np.zeros_like(core_features[:, 2])

    # Transform the Reactive Power Column Q_i of the Core Features based on min_Q_i, max_Q_i
    _min_ = min(min_values[5], min_values[6])
    _max_ = max(max_values[5], max_values[6])
    if _max_ - _min_ == 0:
        core_features[:, 3] = np.zeros_like(core_features[:, 3])
    else:
        core_features[:, 3] = 2 * ((core_features[:, 3] - _min_) / (_max_ - _min_)) - 1

    # if np.isnan(core_features[:, 3]).any():
    # core_features[:, 3] = np.zeros_like(core_features[:, 3])

    return core_features, min_angle, max_angle


def custom_minmax_res_bus_transform(scaler: MinMaxScaler, core_features):
    # Constraint feature columns: min_V_i, max_V_i, max_S, min_P_i, max_P_i, min_Q_i, max_Q_i
    max_values = scaler.data_max_
    min_values = scaler.data_min_

    # Transform the Voltage Magnitude Column V_i of the Core Features based on min_V_i, max_V_i
    _min_ = min_values[0]
    _max_ = max_values[0]
    core_features[:, 0] = 2 * ((core_features[:, 0] - _min_) / (_max_ - _min_)) - 1

    # Transform the voltage angle column delta_i of the core features based on the percentile of min_V_i and max_V_i
    _min_ = min_values[1]
    _max_ = max_values[1]
    core_features[:, 1] = 2 * ((core_features[:, 1] - _min_) / (_max_ - _min_)) - 1

    # Transform the Active Power Column P_i of the Core Features based on min_P_i, max_P_i
    _min_ = min_values[2]
    _max_ = max_values[2]
    if _max_ - _min_ == 0:
        core_features[:, 2] = np.zeros_like(core_features[:, 2])
    else:
        core_features[:, 2] = 2 * ((core_features[:, 2] - _min_) / (_max_ - _min_)) - 1

    # if np.isnan(core_features[:, 2]).any():
    # core_features[:, 2] = np.zeros_like(core_features[:, 2])

    # Transform the Reactive Power Column Q_i of the Core Features based on min_Q_i, max_Q_i
    _min_ = min_values[3]
    _max_ = max_values[3]

    if _max_ - _min_ == 0:
        core_features[:, 3] = np.zeros_like(core_features[:, 3])
    else:
        core_features[:, 3] = 2 * ((core_features[:, 3] - _min_) / (_max_ - _min_)) - 1

    # if np.isnan(core_features[:, 3]).any():
    # core_features[:, 3] = np.zeros_like(core_features[:, 3])

    return core_features  # , min_angle, max_angle


def custom_minmax_inverse_transform(scaler: MinMaxScaler, core_features, min_angle, max_angle):
    # Constraint feature columns: min_V_i, max_V_i, max_S, min_P_i, max_P_i, min_Q_i, max_Q_i
    max_values = scaler.data_max_
    min_values = scaler.data_min_

    # Inverse Transform the Voltage Magnitude Column V_i of the Core Features based on min_V_i, max_V_i
    _min_ = min(min_values[0], min_values[1])
    _max_ = max(max_values[0], max_values[1])
    core_features[:, 0] = (core_features[:, 0] + 1) * 0.5 * (_max_ - _min_) + _min_

    # Inverse Transform the Voltage Angle Column delta_i of the Core Features based on the percentile of min_V_i, max_V_i

    core_features[:, 1] = (core_features[:, 1] + 1) * 0.5 * (max_angle - min_angle) + min_angle

    # Inverse Transform the Active Power Column P_i of the Core Features based on min_P_i, max_P_i
    _min_ = min(min_values[3], min_values[4])
    _max_ = max(max_values[3], max_values[4])
    core_features[:, 2] = (core_features[:, 2] + 1) * 0.5 * (_max_ - _min_) + _min_

    if np.isnan(core_features[:, 2]).any():
        core_features[:, 2] = np.zeros_like(core_features[:, 2])

    # Inverse Transform the Reactive Power Column Q_i of the Core Features based on min_Q_i, max_Q_i
    _min_ = min(min_values[5], min_values[6])
    _max_ = max(max_values[5], max_values[6])
    core_features[:, 3] = (core_features[:, 3] + 1) * 0.5 * (_max_ - _min_) + _min_

    if np.isnan(core_features[:, 3]).any():
        core_features[:, 3] = np.zeros_like(core_features[:, 3])

    return core_features


def print_lr(lr, first_prompt: bool = True):
    if not first_prompt:
        print('-' * 15 + f"   LEARNING RATE HALVING DOWN TO: {lr}   " + '-' * 15)
    print('-' * 15 + f"   CURRENT LEARNING RATE: {lr}   " + '-' * 15)


async def monitor_lr(optimizer: torch.optim.AdamW, prompt_first: bool = True):
    prev_lr = optimizer.param_groups[0]['lr']
    print_lr(prev_lr) if prompt_first else None

    while True:
        await asyncio.sleep(30)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print_lr(current_lr)
            prev_lr = current_lr


async def run_thread(optimizer):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(monitor_lr(optimizer))
    loop.close()


def save_ACOPFGNN_model(model: ACOPFGNN, model_name: str):
    path = r"./Models/SelfSupervised/" + model_name

    torch.save(model.state_dict(), path)


def load_ACOPFGNN_model(grid_name, model_name, hidden_channels, num_layers):
    path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split("\\")) + r"/Models/SelfSupervised/" + model_name
    index_mappers, net, data = generate_unsupervised_input(grid_name)
    model = create_ACOPFGNN_model(data, net, index_mappers, hidden_channels=hidden_channels, num_layers=num_layers)

    # Load the saved model parameter
    ordered_dict = torch.load(path)
    model.load_state_dict(ordered_dict)

    return model


def softabs(x, alpha=1e-2):
    return torch.sqrt(x ** 2 + alpha ** 2)


def arcsin_approx(x_):
    # 5th-degree Taylor approximation: arcsin(x) ≈ x + x^3/6 + 3x^5/40
    x = torch.tanh(x_)
    return x + (x ** 3) / 6 + (3 * (x ** 5)) / 40


def arccos_approx(x):
    # arccos(x) = pi/2 - arcsin(x)
    # Use the arcsin approximation to calculate arccos
    return (torch.pi / 2) - arcsin_approx(x)


def inverse_map_index(node_type: str, index_mappers) -> list:
    indices = []
    idx_mapper = index_mappers[0]
    node_type_idx_mapper = index_mappers[1]

    for i in range(len(node_type_idx_mapper[node_type])):
        idx = idx_mapper[node_type_idx_mapper[node_type][i]]
        indices.append(idx)

    return indices


def calc_physics_loss(x_dict, out_dict, bus_idx_neighbors_dict, scaler: StandardScaler):
    loss = torch.nn.functional.mse_loss
    P_physics = []
    P_model = []
    Q_physics = []
    Q_model = []

    V_mean = scaler.mean_[0]
    V_std = scaler.scale_[0]
    delta_mean = scaler.mean_[1]
    delta_std = scaler.scale_[1]
    P_mean = scaler.mean_[2]
    P_std = scaler.scale_[2]
    Q_mean = scaler.mean_[3]
    Q_std = scaler.scale_[3]

    for from_bus in bus_idx_neighbors_dict:
        for bus_idx in bus_idx_neighbors_dict[from_bus]:

            V_i = out_dict[from_bus][bus_idx][0] #* torch.tensor(V_std) + torch.tensor(V_mean)
            volt_angle_i = out_dict[from_bus][bus_idx][1] #* torch.tensor(delta_std) + torch.tensor(delta_mean)
            P_i = x_dict[from_bus][bus_idx][2] #* torch.tensor(P_std) + torch.tensor(P_mean)
            Q_i = x_dict[from_bus][bus_idx][3] #* torch.tensor(Q_std) + torch.tensor(Q_mean)

            P = []
            Q = []

            for pair in bus_idx_neighbors_dict[from_bus][bus_idx]:
                # For each neighbor of the iterated bus
                to_bus, to_bus_idx, edge_attr = pair

                V_j = out_dict[to_bus][to_bus_idx][0] #* torch.tensor(V_std) + torch.tensor(V_mean)

                if to_bus_idx >= len(out_dict[to_bus]):
                    print("INDEX PROBLEM")

                volt_angle_j = out_dict[to_bus][to_bus_idx][1]# * torch.tensor(delta_std) + torch.tensor(delta_mean)
                delta_ij = volt_angle_i - volt_angle_j

                G_ij = edge_attr[0] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)

                B_ij = -edge_attr[1] / (edge_attr[0] ** 2 + edge_attr[1] ** 2)

                # ACOPF Equation for P_i
                P_ij = 1.0 * softabs(V_i) * softabs(V_j) * (G_ij * torch.cos(delta_ij) + B_ij * torch.sin(delta_ij))

                P.append(P_ij)

                # ACOPF Equation for Q_i
                Q_ij = 1.0 * softabs(V_i) * softabs(V_j) * (G_ij * torch.sin(delta_ij) - B_ij * torch.cos(delta_ij))

                Q.append(Q_ij)

            # Sums of all P_ij and Q_ij equal to P_i and Q_i respectively
            P_ = torch.stack(P)
            sum_P = torch.sum(P_)
            P_physics.append(sum_P)
            P_model.append(P_i)
            Q_ = torch.stack(Q)
            sum_Q = torch.sum(Q_)
            Q_physics.append(sum_Q)
            Q_model.append(Q_i)

    print(f"P: Output= {P_physics[-1].item() * P_std + P_mean} --- Target= {P_model[-1].item() * P_std + P_mean}")
    print(f"Q: Output= {Q_physics[-1].item() * Q_std + Q_mean} --- Target= {Q_model[-1].item() * Q_std + Q_mean}")
    LOSS_P = loss(torch.stack(P_physics),torch.stack(P_model))
    LOSS_Q = loss(torch.stack(Q_physics),torch.stack(Q_model))

    return LOSS_P + LOSS_Q


def soft_sqrt(x, epsilon=1e-8):
    return torch.sqrt(x + epsilon)


def calc_constraint_loss(out_dict, constraints_dict, scaler):
    # Objective Function on basis of ACOPF Equations for P and Q
    delta_mean = scaler.mean_[1]
    delta_std = scaler.scale_[1]
    delta_sum_target = -delta_mean / delta_std

    P_mean = scaler.mean_[2]
    P_std = scaler.scale_[2]
    P_sum_target = -P_mean / P_std
    Q_mean = scaler.mean_[3]
    Q_std = scaler.scale_[3]
    Q_sum_target = -Q_mean / Q_std

    P_penalties = []
    Q_penalties = []
    V_penalties = []
    delta_penalties = []
    S_penalties = []
    loss = torch.nn.functional.mse_loss

    for from_bus in out_dict:
        V_node = out_dict[from_bus][:, 0].view(-1, 1)
        delta_node = out_dict[from_bus][:, 1].view(-1, 1)
        P_node = out_dict[from_bus][:, 2].view(-1, 1)
        Q_node = out_dict[from_bus][:, 3].view(-1, 1)

        V_min = constraints_dict[from_bus][:, 0].view(-1, 1)
        V_max = constraints_dict[from_bus][:, 1].view(-1, 1)
        S = constraints_dict[from_bus][:, 2].view(-1, 1)
        P_min = constraints_dict[from_bus][:, 3].view(-1, 1)
        P_max = constraints_dict[from_bus][:, 4].view(-1, 1)
        Q_min = constraints_dict[from_bus][:, 5].view(-1, 1)
        Q_max = constraints_dict[from_bus][:, 6].view(-1, 1)

        delta_violation = torch.nn.functional.relu(delta_sum_target - delta_node)

        if from_bus == "NB":
            P_violation = softabs(P_node - P_sum_target)  # softabs(P_node - P_sum_target)
            Q_violation = softabs(Q_node - Q_sum_target)  # softabs(Q_node - Q_sum_target)
        else:
            P_upper_violation = torch.nn.functional.relu(P_node - P_max)
            P_lower_violation = torch.nn.functional.relu(P_min - P_node)
            Q_upper_violation = torch.nn.functional.relu(Q_node - Q_max)
            Q_lower_violation = torch.nn.functional.relu(Q_min - Q_node)
            P_violation = P_upper_violation + P_lower_violation
            Q_violation = Q_upper_violation + Q_lower_violation

        # Penalty for exceeding upper bounds

        V_upper_violation = torch.nn.functional.relu(V_node - V_max)
        V_lower_violation = torch.nn.functional.relu(V_min - V_node)
        V_violation = V_lower_violation + V_upper_violation

        # Max apparent power penalty
        S_violation = torch.nn.functional.relu(soft_sqrt(torch.square(P_node) + torch.square(Q_node)) - S)

        P_penalties.extend(P_violation)
        Q_penalties.extend(Q_violation)
        V_penalties.extend(V_violation)
        S_penalties.extend(S_violation)

        if from_bus != "SB":
            delta_penalties.extend(delta_violation)
        else:
            delta_penalties.extend(
                torch.tensor([0.0] * len(delta_penalties), dtype=torch.float32, requires_grad=False).view(-1, 1))

    P_pen_ = torch.stack(P_penalties)
    S_pen_ = torch.stack(S_penalties)
    Q_pen_ = torch.stack(Q_penalties)
    V_pen_ = torch.stack(V_penalties)
    delta_pen_ = torch.stack(delta_penalties)

    P_pen = loss(P_pen_, torch.tensor([0.0] * len(P_penalties), dtype=torch.float32, requires_grad=False).view(-1, 1))
    Q_pen = loss(Q_pen_, torch.tensor([0.0] * len(Q_penalties), dtype=torch.float32, requires_grad=False).view(-1, 1))
    V_pen = loss(V_pen_, torch.tensor([0.0] * len(V_penalties), dtype=torch.float32, requires_grad=False).view(-1, 1))
    S_pen = loss(S_pen_, torch.tensor([0.0] * len(S_penalties), dtype=torch.float32, requires_grad=False).view(-1, 1))
    delta_pen = loss(delta_pen_,
                     torch.tensor([0.0] * len(delta_penalties), dtype=torch.float32, requires_grad=False).view(-1, 1))

    Penalties = torch.sum(torch.stack([P_pen, Q_pen, V_pen, S_pen, delta_pen]))

    return Penalties


def calc_unsupervised_loss(out_dict, scaler):
    # Objective Function on basis of ACOPF Equations for P and Q
    # mre_loss_fn = lambda outputs, targets: get_mre_loss(outputs, targets)
    P_mean = scaler.mean_[2]
    P_std = scaler.scale_[2]
    # P_sum_target = -P_mean / P_std
    Q_mean = scaler.mean_[3]
    Q_std = scaler.scale_[3]
    # Q_sum_target = -Q_mean / Q_std
    P_supplied = []
    Q_supplied = []
    loss = torch.nn.functional.mse_loss

    for from_bus in out_dict:
        P_node = out_dict[from_bus][:, 2].view(-1, 1)
        Q_node = out_dict[from_bus][:, 3].view(-1, 1)
        P_supplied.extend(P_node.view(-1, 1))
        Q_supplied.extend(Q_node.view(-1, 1))

    # Unscale the outputs and calculate the unsupervised loss from the inverse transformed inputs
    P_supplied_unscaled = [P * torch.tensor(P_std) + torch.tensor(P_mean) for P in P_supplied]
    Q_supplied_unscaled = [Q * torch.tensor(Q_std) + torch.tensor(Q_mean) for Q in Q_supplied]

    unsupervised_loss_P = torch.sqrt(loss(torch.sum(torch.stack(P_supplied_unscaled)),
                                          torch.tensor(0.0, dtype=torch.float32, requires_grad=False)))
    unsupervised_loss_Q = torch.sqrt(loss(torch.sum(torch.stack(Q_supplied_unscaled)),
                                          torch.tensor(0.0, dtype=torch.float32, requires_grad=False)))
    unsupervised_loss = unsupervised_loss_P + unsupervised_loss_Q

    return unsupervised_loss
