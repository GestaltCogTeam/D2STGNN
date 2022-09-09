import numpy as np


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """Load train/val/test data and get a dataloader.
            Ref code: https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
        Args:
            xs (np.array): history sequence, [num_samples, history_len, num_nodes, num_feats].
            ys (np.array):  future sequence, ]num_samples, future_len, num_nodes, num_feats].
            batch_size (int): batch size
            pad_with_last_sample (bool, optional): pad with the last sample to make number of samples divisible to batch_size. Defaults to True.
            shuffle (bool, optional): shuffle dataset. Defaults to False.
        """

        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        # number of batches
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        if shuffle:
            self.shuffle()

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        """Fetch a batch of data."""
        
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size *
                              (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
