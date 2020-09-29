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
            print("before return:", r)
            return r
        raise StopIteration()

class myIterDataset(IterableDataset):
    def __init__(self, itera):
        self._itera = itera
    def __iter__(self):
        return self._itera


if __name__ == '__main__':
    fab = Fab(5)
    iterfab = myIterDataset(fab)
    loader = DataLoader(dataset=iterfab, batch_size=2)

    for batch in loader: 
        print(batch)


