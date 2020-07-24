

import torch
import torch.nn as nn
import torch.nn.functional as F



class AlexNet(nn.Module):


    def __init__(self):
        super(AlexNet, self).__init__()
        # TODO (keyan) define architecture here



    # TODO (keyan) forward propgation for input x
    def forward(self, x):
        pass



    # TODO (nima) training loop on dataset
    def train_model(self, dataset, batch_size, epochs, learning_rate):
        pass



    # TODO (nima) testing phase with given dataset
    def test_model(self, dataset):
        pass