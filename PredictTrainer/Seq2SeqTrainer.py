from Models.Seq2SeqLstmAE import Seq2SeqLstmAE
import torch.optim
import torch.nn as nn

class Seq2SeqTrainer():
    def __init__(self) -> None:
        self.predictModel = Seq2SeqLstmAE(1,4,10,1,2)
        self.lossFunction = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.predictModel.parameters(), 1e-2)
        pass

    def train(self, hs, dataLengths, context, ls):
        self.predictModel.train()
        epoch += 1
        # totalLv, theoryLv, noise_out = predictModel(hs, context, fullDataLenghts, preLv)
        totalLv = self.predictModel(hs, dataLengths, context, hs.shape[1])
        mseloss2 = self.lossFunction(totalLv, ls)
        # noiseLoss = loss_function(noise_out,zeros)
        # loss = mseloss2 + noiseLoss
        loss = mseloss2
        loss.backward()
        print('loss', loss.item())

    def eval():
        
