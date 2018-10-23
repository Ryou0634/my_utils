import numpy as np
import random

def raw_data(batch):
    return zip(*batch)

class DataLoader():
    '''
    Data loader for mini-batch stochastic gradient decent models.
    Shuffle dataset and feed it into a model. Typically used with Trainer.

    Attributes
    ----------
    dataset : List
        List of tuples (feature, target)
    batch_size : int
        The batchsize.
    n_batches : int
        The number of mini-batches for one epoch.
    train : bool
        If True, shuffle the dataset and create infinite loop.
    trans_func :
        Function to process data before feeding them into a model.
    _i : int
        Just a counter.
    '''

    def __init__(self, dataset, trans_func=raw_data, batch_size=1, train=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = len(dataset)//self.batch_size + int(bool(len(dataset)%self.batch_size))
        self.train = train
        self.trans_func = trans_func
        self._i = 0
        self.n_epochs = 0

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._i = 0
        if self.train:
            random.shuffle(self.dataset)
        return self

    def __next__(self):
        if self._i >= len(self.dataset):
            if self.train:
                iter(self)
            else:
                raise StopIteration()
        batch = self.dataset[self._i:self._i+self.batch_size]
        inputs, targets = self.trans_func(batch)
        self._i += self.batch_size
        if self._i >= len(self.dataset):
            self.n_epochs += 1
        return inputs, targets

    def __repr__(self):
        repr = 'DataLoader(\n' + \
                    '\tdatasize: {}\n'.format(len(self.dataset)) + \
                    '\tbatchsize: {}\n'.format(self.batch_size) + \
                    '\tn_batches: {}\n'.format(self.n_batches) + \
                    '\ttrans_func: {}\n'.format(self.trans_func.__class__.__name__) + \
                    '\tdevice: {}\n)'.format(self.trans_func.device)
        return repr

class MultiDataLoader():
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        for loader in self.loaders:
            iter(loader)
        return self

    def __next__(self):
        datas = [next(loader) for loader in self.loaders]
        return zip(*datas)
