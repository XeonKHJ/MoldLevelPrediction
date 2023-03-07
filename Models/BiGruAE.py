import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import numpy

class BiGruAE(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, feature_size, extraFeatSize, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnnEncoder = nn.GRU(feature_size, hidden_size, num_layers,batch_first =True, bidirectional=True) # utilize the LSTM model in torch.nn 
        self.encodeFc = nn.Linear(2*hidden_size, hidden_size)
        self.rnnDecoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True) 
        
        self.decodeFc = nn.Linear(2*hidden_size,output_size)

        self.staticBindingFc1 = nn.Linear(1, 3)
        self.staticBindingFc2 = nn.Linear(4, 3)
        self.staticBindingFc2 = nn.Linear(3, 1)
        # self.finalCalculation = nn.Sigmoid()
        self.isCudaSupported = torch.cuda.is_available()

    def forward(self, to_x, xTimestampSizes, context=None, outputLength = None):
        xTimestampSizes = xTimestampSizes.tolist()
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, b = self.rnnEncoder(x)  # _x is input, size (seq_len, batch, input_size)
        x, _ = torchrnn.pad_packed_sequence(x, batch_first=True)
        x = self.encodeFc(x)
        x = torchrnn.pack_padded_sequence(x, xTimestampSizes, True)
        x, b = self.rnnDecoder(x)

        x, lengths = torchrnn.pad_packed_sequence(x, batch_first=True)
    
        x = self.decodeFc(x)
        # x = self.finalCalculation(x)
        if outputLength == None:
            outputLength = to_x.shape[1]
        x = x[:, x.shape[1]-outputLength:x.shape[1]]
        return x