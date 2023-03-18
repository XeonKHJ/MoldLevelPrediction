import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import numpy

class CoefficientPredictModel(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, tempFeatSize, staticFeatSize, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.fc1 = nn.Linear(3,3)
        self.fc1ac = nn.LeakyReLU()
        self.fc2 = nn.Linear(3,5)
        self.fc3 = nn.Linear(5,3)

    def forward(self, x, xTimestampSizes, context=None, outputLength = None):
        x = self.fc1(x)
        x = self.fc1ac(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x