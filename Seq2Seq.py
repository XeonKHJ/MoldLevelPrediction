import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class Seq2Seq(nn.Module):
    """
        Parameters:
        - feature_nums: list of feature num for every detector.
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self):
        super().__init__()
        hidden_size = 4
        num_layers = 2
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.forwardCalculation = nn.Linear(hidden_size, 2)
        self.decoder = nn.LSTM(input_size=1, hidden_size=10, num_layers=num_layers, batch_first=True)
        self.finalFc = nn.Linear(10, 1)


    def forward(self, x, context):
        lstm_out, (h,c) = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        # reshape_lstm_out = h.reshape([-1])
        foward_out = self.forwardCalculation(h[-1])


        noiseOutput = torch.zeros([x.shape[0],0,1], device=torch.device('cuda'))
        for i in range(x.shape[1]):
            singleNoiseOutput, (h, c) = self.noiseLstm(h[-1,:,:].reshape([-1,1,1]), (h, c))
            noiseOutput = torch.cat((noiseOutput, singleNoiseOutput), 1)
        noiseOutput = self.noiseFc(noiseOutput)
        
        finalOutput = noiseOutput + foward_out

        return finalOutput, foward_out, noise_out
    