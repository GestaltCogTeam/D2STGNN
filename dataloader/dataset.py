from os import setxattr
import os
from pickle import load
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.load_data import StandardScaler, load_dataset
from utils.log import clock

class TrafficDataset(Dataset):
    def __init__(self, xs, ys, batch_size, pad=True) -> None:
        super().__init__()
        if pad:
            num_padding     = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding       = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding       = np.repeat(ys[-1:], num_padding, axis=0)
            xs  = np.concatenate([xs, x_padding], axis=0)
            ys  = np.concatenate([ys, y_padding], axis=0)
        self.history    = xs
        self.future     = ys

        self.size = len(xs)

    def __len__(self):
            return self.size
        
    def __getitem__(self, idx):
        xi = self.history[idx]
        yi = self.future[idx]
        return xi, yi

def load_dataset_my(data_dir, batch_size, valid_batch_size, test_batch_size):
    data_dict = {}
    for mode in ['train', 'val', 'test']:
        _   = np.load(os.path.join(data_dir, mode + '.npz'))
        # length  = int(len(_['x']) * 0.1)
        # data_dict['x_' + mode]  = _['x'][:length, :, :, :]
        # data_dict['y_' + mode]  = _['y'][:length, :, :, :]
        data_dict['x_' + mode]  = _['x']
        data_dict['y_' + mode]  = _['y']
    scaler  = StandardScaler(mean=data_dict['x_train'][..., 0].mean(), std=data_dict['x_train'][..., 0].std())    # we only see the training data.
    # data standardization 
    for mode in ['train', 'val', 'test']:
        # continue
        data_dict['x_' + mode][..., 0] = scaler.transform(data_dict['x_' + mode][..., 0])
        data_dict['y_' + mode][..., 0] = scaler.transform(data_dict['y_' + mode][..., 0])
    train_data  = TrafficDataset(data_dict['x_train'], data_dict['y_train'], batch_size)
    val_data    = TrafficDataset(data_dict['x_val'], data_dict['y_val'], batch_size)
    test_data   = TrafficDataset(data_dict['x_test'], data_dict['y_test'], batch_size)
    
    train_dataloader    = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=32)
    val_dataloader      = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=32)
    test_dataloader     = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=32)

    data_dict['train_loader']   = train_dataloader
    data_dict['val_loader']     = val_dataloader
    data_dict['test_loader']    = test_dataloader
    data_dict['scaler']         = scaler

    return data_dict

data_dict_my = load_dataset_my("datasets/METR", 64, 64, 64)
data_dict = load_dataset("datasets/METR", 64, 64, 64)

@clock
def iter_dataloader_my(data_dict):
    train_dataloader = data_dict['train_loader']
    val_dataloader = data_dict['val_loader']
    test_dataloader = data_dict['test_loader']

    for itera, (x, y) in enumerate(train_dataloader):
        pass
    for itera, (x, y) in enumerate(val_dataloader):
        pass
    for itera, (x, y) in enumerate(test_dataloader):
        pass
@clock
def iter_dataloader(data_dict):
    train_dataloader = data_dict['train_loader']
    val_dataloader = data_dict['val_loader']
    test_dataloader = data_dict['test_loader']

    for itera, (x, y) in enumerate(train_dataloader.get_iterator()):
        pass
    for itera, (x, y) in enumerate(val_dataloader.get_iterator()):
        pass
    for itera, (x, y) in enumerate(test_dataloader.get_iterator()):
        pass

iter_dataloader(data_dict)
iter_dataloader_my(data_dict_my)
# 速度比他的慢好多