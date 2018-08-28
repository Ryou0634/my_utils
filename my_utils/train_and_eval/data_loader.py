import numpy as np
import random

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
    n_batch : int
        The number of mini-batches for one epoch.
    shuffle : bool
        If True, shuffle the dataset.
    trans_func :
        Function to process data before feeding them into a model.
    _i : int
        Just a counter.
    '''

    def __init__(self, dataset, batch_size=1, trans_func=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batch = len(dataset)//self.batch_size + int(bool(len(dataset)%self.batch_size))
        self.shuffle = shuffle
        self.trans_func = trans_func
        self._i = 0

    def __len__(self):
        return len(self.n_batch)

    def __iter__(self):
        self._i = 0
        if self.shuffle:
            random.shuffle(self.dataset)
        return self

    def __next__(self):
        if self._i >= len(self.dataset):
            raise StopIteration()
        batch = self.dataset[self._i:self._i+self.batch_size]
        inputs, targets = zip(*batch)
        if self.trans_func:
            inputs, targets = self.trans_func(inputs, targets)
        self._i += self.batch_size
        return inputs, targets

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
