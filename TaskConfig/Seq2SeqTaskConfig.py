from Models.Seq2SeqLstmAE import Seq2SeqLstmAE
from Trainer.FEGAETrainer import FEGAETrainer
from Models.BiGruAE import BiGruAE
from Models.BiLSTMAE import BiLSTMAE
from Models.Seq2SeqGruAE import Seq2SeqGruAE

import torch
import os.path as path

from Trainer.Trainer import Trainer

class Seq2SeqTaskConfig():
    def __init__(self, logger, modelName, showTrainingInfo=True, windowSize = 10):
        self.logger = logger
        self.modelName = modelName
        self.showTrainningInfo = showTrainingInfo
        self.windowSize = windowSize

    def getConfig(self, isCuda = False):
        feature_size = 1
        extraFeatSize = 4
        output_size = 1
        forcastModel = Seq2SeqLstmAE(feature_size,extraFeatSize,10,output_size,4)
        if torch.cuda.is_available():
            forcastModel.cuda()
        trainer = Trainer(forcastModel, self.logger, 1e-3, self.modelName, self.showTrainningInfo, windowSize=self.windowSize)
        try:
            trainer.load()
        except:
            pass
        
        return trainer
