

"""
This class should provide a way to load different datasets.
"""
import torchvision
from torch.utils.data import DataLoader

class Data:

    # TODO default dataset is CIFAR10 but we should allow for different options 
    # (maybe not ImageNet right now since it's huge)
    # TODO provide options for other datasets
    def __init__(self, root, dataset='cifar', train=True, download=True):
        
        self.dataset = None
        if dataset == 'cifar' and download:
           self.dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)



    """
    This function will return a data loader that batches and shuffles the data.
    """
    def load(self, batch_size=32, shuffle=True):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

