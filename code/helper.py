import pandapower.topology
import simbench
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_geometric.data import Data
import sklearn.metrics as metrics
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import pandapower as pp
#import networkx as nx
#import pandapower.plotting as plot
import wandb
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, MaxAbsScaler, QuantileTransformer
import simbench as sb
import os
import random
import julia


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

    print("Calculating edge index and edge weights for the grid " + grid_name + " ...")

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

    # Convert edge connections to undirected
    edge_index, edge_attr = to_undirected(edge_index=torch.tensor(edge_index, dtype=torch.int),
                                             edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                                             num_nodes=len(net.bus.index))

    # Add self loops to edge connections
    edge_index, edge_attr = add_self_loops(edge_index=edge_index, edge_attr=edge_attr,
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


def normalize(lst):
    _sum_ = sum(lst)
    return [float(i) / _sum_ for i in lst]


def train_one_epoch(epoch, optimizer, training_loader, model, loss_fn, edge_index, edge_weights, scaler):

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

        # Define Scaler and standardize inputs and targets
        targets = torch.tensor(scaler.fit_transform(targets), dtype=torch.float32)
        inputs = torch.tensor(scaler.transform(inputs), dtype=torch.float32)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs, edge_index, edge_weights, edge_weights)

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
        'epoch': epoch,
        'train_rmse_loss': train_rmse_loss/ last_idx,
        'train_mae_loss': train_mae_loss / last_idx,
        'train_mre_loss': train_mre_loss / last_idx

    })


def validate_one_epoch(epoch, validation_loader, model, loss_fn, edge_index, edge_weights, scaler):

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

        # Define Scaler and standardize inputs and targets
        targets = torch.tensor(scaler.fit_transform(targets), dtype=torch.float32)
        inputs = torch.tensor(scaler.transform(inputs), dtype=torch.float32)

        # Make predictions for this batch
        outputs = model(inputs, edge_index, edge_weights, edge_weights)

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
        'epoch': epoch,
        'val_rmse_loss': val_rmse_loss / last_idx,
        'val_mae_loss': val_mae_loss / last_idx,
        'val_mre_loss': val_mre_loss / last_idx

    })


def train_validate_one_epoch(epoch, optimizer, training_loader, validation_loader, model, loss_fn, edge_index, edge_weights, scaler):

    print("Training the model for epoch " + str(epoch))
    # Train for an epoch
    train_one_epoch(epoch, optimizer, training_loader, model, loss_fn, edge_index, edge_weights, scaler)
    print("Validating the model on unseen Datasets for epoch " + str(epoch))
    # Validate for an epoch
    validate_one_epoch(epoch, validation_loader, model, loss_fn, edge_index, edge_weights, scaler)


def test_one_epoch(test_loader, model, loss_fn, edge_index, edge_weights, scaler):

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

        # Define Scaler and standardize inputs and targets
        targets = torch.tensor(scaler.fit_transform(targets), dtype=torch.float32)
        inputs = torch.tensor(scaler.transform(inputs), dtype=torch.float32)

        # Make predictions for this batch
        outputs = model(inputs, edge_index, edge_weights)

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

    wandb.log({
        'test_rmse_loss': test_rmse_loss / last_idx,
        'test_mae_loss': test_mae_loss / last_idx,
        'test_mre_loss': test_mre_loss / last_idx
    })

    return output, target



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
