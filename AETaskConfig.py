from FEGAETrainer import FEGAETrainer
from Models.BiGruAE import BiGruAE
from Models.BiLSTMAE import BiLSTMAE
from Models.Seq2SeqGruAE import Seq2SeqGruAE

import torch
import os.path as path

from Trainer import Trainer

class AETaskConfig():
    def __init__(self, logger, modelName, showTrainingInfo=True):
        self.logger = logger
        self.modelName = modelName
        self.showTrainningInfo = showTrainingInfo

    def getConfig(self, isCuda = False):
        feature_size = 1
        extraFeatSize = 4
        output_size = 1
        forcastModel = BiLSTMAE(feature_size,extraFeatSize,10,output_size,2)
        if torch.cuda.is_available():
            forcastModel.cuda()
        trainer = Trainer(forcastModel, self.logger, 1e-3, self.modelName, self.showTrainningInfo)
        try:
            trainer.load()
        except:
            pass
        
        return trainer
