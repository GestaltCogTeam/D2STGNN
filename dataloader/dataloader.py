from operator import length_hint
import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import torch 
from torch.utils.data import Dataset,DataLoader


class DataLoader(object):
    r"""
    Description:
    -----------
    Load train/val/test data and get a dataloader.
        
    Args:
    -----------
    xs: np.array
        History sequence X, num_samples x T_in x num_nodes x features.
    ys: np.array
        Predict sequence Y, num_samples x T_out x num_nodes x features. xs[i] is corresponding to xy[i].
    batch_size: int
        Batch Size
    pad_with_last_sample: bool
        Pad with the last sample to make number of samples divisible to batch_size.
        
    Attributes:
    -----------
    num_batch: int
        Number of batch.
    size: int
        Length of input X.
    """
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size     = batch_size
        self.current_ind    = 0

        if pad_with_last_sample:
            num_padding     = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding       = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding       = np.repeat(ys[-1:], num_padding, axis=0)
            xs  = np.concatenate([xs, x_padding], axis=0)
            ys  = np.concatenate([ys, y_padding], axis=0)

        self.size       = len(xs)
        self.num_batch  = int(self.size // self.batch_size)
        self.xs         = xs
        self.ys         = ys
        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys  = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
    
    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        r"""
        Description:
        -----------
        Fetch a batch of data.

        Parameters:
        -----------
        None

        Returns:
        -----------
        (xi, yi): tuple
            xi: batch_size x T_in x num_nodes x features.
            yi: batch_size x T_out x num_nodes x features.
        """
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind   = self.batch_size * self.current_ind
                end_ind     = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i         = self.xs[start_ind: end_ind, ...]
                y_i         = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
