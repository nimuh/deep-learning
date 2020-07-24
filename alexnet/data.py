

"""
This class should provide a way to load different datasets.
"""
import torchvision
from torch.utils.data import DataLoader

class Data:

    # TODO default dataset is CIFAR10 but we should allow for different options 
    # (maybe not ImageNet right now since it's huge)
    def __init__(self, dataset='cifar10', train=True, download=True):
        self.dataset_train, self.dataset_test = self.__get_this_dataset(dataset, download=download)
        

    """
    This function will return a data loader that batches and shuffles the data.
    """
    def load(self, batch_size=32, shuffle=True):
        return (DataLoader(self.dataset_train, batch_size=batch_size, shuffle=shuffle), 
                DataLoader(self.dataset_test, batch_size=batch_size, shuffle=shuffle),
                )


    """
    Private function for getting the dataset by name from torchvision module.

    Dataset options: CIFAR10, CIFAR100, MNIST

    TODO Add more datasets
    """
    def __get_this_dataset(self, dataset_name, download):
        dataset_train = None
        dataset_test = None

        if dataset_name == 'cifar10' :
            dataset_train = torchvision.datasets.CIFAR10(root='./', train=True, download=download)
            dataset_test = torchvision.datasets.CIFAR10(root='./', train=False, download=download)

        elif dataset_name == 'cifar100':
            dataset_train = torchvision.datasets.CIFAR100(root='./', train=True, download=download)
            dataset_test = torchvision.datasets.CIFAR100(root='./', train=False, download=download)

        elif dataset_name == 'mnist':
            dataset_train = torchvision.datasets.MNIST(root='./', train=True, download=download)
            dataset_test = torchvision.datasets.MNIST(root='./', train=False, download=download)

        return dataset_train, dataset_test