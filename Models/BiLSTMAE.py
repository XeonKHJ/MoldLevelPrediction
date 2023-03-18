import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import numpy

class BiLSTMAE(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, tempFeatSize, staticFeatSize, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.tempFeatSize = tempFeatSize
        self.staticFeatSize = staticFeatSize
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnnEncoder = nn.LSTM(tempFeatSize, hidden_size, num_layers,batch_first =True, bidirectional=True) # utilize the LSTM model in torch.nn 
        self.encodeFc = nn.Linear(2*hidden_size, hidden_size)
        self.rnnDecoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True) 
        
        self.decodeFc = nn.Linear(2*hidden_size,output_size)

        self.staticBindingFc1 = nn.Linear(self.tempFeatSize + self.staticFeatSize, 4)
        self.relu1 = nn.LeakyReLU()
        self.staticBindingFc2 = nn.Linear(4, 3)
        self.relu2 = nn.LeakyReLU()
        self.staticBindingFc3 = nn.Linear(3, 1)
        # self.finalCalculation = nn.Sigmoid()
        self.isCudaSupported = torch.cuda.is_available()

    def forward(self, to_x, xTimestampSizes, context=None, outputLength = None):
        xTimestampSizes = xTimestampSizes.tolist()
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, (b, h) = self.rnnEncoder(x)  # _x is input, size (seq_len, batch, input_size)
        x, _ = torchrnn.pad_packed_sequence(x, batch_first=True)
        x = self.encodeFc(x)
        x = torchrnn.pack_padded_sequence(x, xTimestampSizes, True)
        x, (b, H) = self.rnnDecoder(x)

        x, lengths = torchrnn.pad_packed_sequence(x, batch_first=True)
        x = self.decodeFc(x)

        # x = self.finalCalculation(x)
        if outputLength == None:
            outputLength = to_x.shape[1]
        x = x[:, x.shape[1]-outputLength:x.shape[1]]

        repeatedContext = context.reshape([context.shape[0], -1, context.shape[1]]).repeat(1, to_x.shape[1],1)
        x = torch.cat((x, repeatedContext), 2)

        x = self.staticBindingFc1(x)
        x = self.relu1(x)
        x = self.staticBindingFc2(x)
        x = self.relu2(x)
        x = self.staticBindingFc3(x)

        return x