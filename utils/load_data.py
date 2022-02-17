#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pickle
import os
import numpy as np
from dataloader import DataLoader
from utils.cal_adj import *

def re_normalization(x, mean, std):
    r"""
    Standard re-normalization

    mean: float
        Mean of data
    std: float
        Standard of data
    """
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    r"""
    Max-min normalization

    _max: float
        Max
    _min: float
        Min
    """
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    r"""
    Max-min re-normalization

    _max: float
        Max
    _min: float
        Min
    """
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

class StandardScaler():
    r"""
    Description:
    -----------
    Standard the input.

    Args:
    -----------
    mean: float
        Mean of data.
    std: float
        Standard of data.

    Attributes:
    -----------
    Same as Args.
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    r"""
    Description:
    -----------
    Load pickle data.
    
    Parameters:
    -----------
    pickle_file: str
        File path.

    Returns:
    -----------
    pickle_data: any
        Pickle data.
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    r"""
    Description:
    -----------
    Load the whole datasets.

    Parameters:
    -----------
    data_dir: str
        Dictionary of data, e.g., 'datasets/METR'.
    batch_size: int
        Batch size.
    valid_batch_size: int
        Valid batchs size.
    test_batch_size: int
        test batch size.
    
    Returns:
    -----------
    datasets:
    
    """
    data_dict = {}
    # read data: train_x, train_y, val_x, val_y, test_x, test_y
    # the data has been processed and stored in datasets/{dataset}/{mode}.npz
    for mode in ['train', 'val', 'test']:
        _   = np.load(os.path.join(data_dir, mode + '.npz'))
        # length  = int(len(_['x']) * 0.1)
        # data_dict['x_' + mode]  = _['x'][:length, :, :, :]
        # data_dict['y_' + mode]  = _['y'][:length, :, :, :]
        data_dict['x_' + mode]  = _['x']
        data_dict['y_' + mode]  = _['y']
    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':    # traffic flow
        _min = pickle.load(open("datasets/" + dataset_name + "/min.pkl", 'rb'))
        _max = pickle.load(open("datasets/" + dataset_name + "/max.pkl", 'rb'))

        # normalization
        y_train = np.squeeze(np.transpose(data_dict['y_train'], axes=[0, 2, 1, 3]), axis=-1)
        y_val = np.squeeze(np.transpose(data_dict['y_val'], axes=[0, 2, 1, 3]), axis=-1)
        y_test = np.squeeze(np.transpose(data_dict['y_test'], axes=[0, 2, 1, 3]), axis=-1)
    
        y_train_new = max_min_normalization(y_train, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_train']    = np.transpose(y_train_new, axes=[0, 2, 1])
        y_val_new = max_min_normalization(y_val, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_val']      = np.transpose(y_val_new, axes=[0, 2, 1])
        y_test_new = max_min_normalization(y_test, _max[:, :, 0, :], _min[:, :, 0, :])
        data_dict['y_test']     = np.transpose(y_test_new, axes=[0, 2, 1])

        data_dict['train_loader']   = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']     = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader']    = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler']         = re_max_min_normalization

    else:   # traffic speed
        scaler  = StandardScaler(mean=data_dict['x_train'][..., 0].mean(), std=data_dict['x_train'][..., 0].std())    # we only see the training data.

        for mode in ['train', 'val', 'test']:
            # continue
            data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
            data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])
        
        data_dict['train_loader']   = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
        data_dict['val_loader']     = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
        data_dict['test_loader']    = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
        data_dict['scaler']         = scaler

    return data_dict

def load_adj(file_path, adj_type):
    r"""
    Description:
    -----------
    Load adjacent matrix and preprocessed it.

    Parameters:
    -----------
    file_path: str
        Adjacent matrix file path (pickle file).
    adj_type: str
        How to preprocess adj matrix.
    
    Returns:
    -----------
        adj_matrix    
    """
    try:
        # METR and PEMS_BAY
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    except:
        # PEMS04
        adj_mx = load_pickle(file_path)
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [transition_matrix(adj_mx).T, transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32).todense()]
    elif adj_type == 'original':
        adj = adj_mx
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx
