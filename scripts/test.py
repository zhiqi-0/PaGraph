import torch
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

class Fab(object): 
    def __init__(self, max): 
        self.max = max 
        self.n, self.a, self.b = 0, 0, 1 
    def __iter__(self): 
        return self 
    def __next__(self): 
        if self.n < self.max: 
            r = self.b 
            self.a, self.b = self.b, self.a + self.b
            self.n = self.n + 1
            return r
        raise StopIteration()

class myIterDataset(IterableDataset):
    def __init__(self, itera):
        self._itera = itera
    def __iter__(self):
        return self._itera

class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.
    Used as sampler for :class:`~torch.utils.data.IterableDataset`.
    """

    def __init__(self):
        super(_InfiniteConstantSampler, self).__init__(None)

    def __iter__(self):
        while True:
            yield None

    def __len__(self):
        # This has to be a TypeError, otherwise, since this is used in
        # `len(dataloader)`, `list(dataloader)` will fail.
        # see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        raise TypeError('Cannot determine the DataLoader length of a IterableDataset')

if __name__ == '__main__':
    fab = Fab(5)
    iterfab = myIterDataset(fab)
    loader = DataLoader(dataset=iterfab)

    sampler = _InfiniteConstantSampler()
    sampler_iter = iter(sampler)
    for i in range(10):
        index = next(sampler_iter)
        print(index)

    for i in loader:
        print(i)


