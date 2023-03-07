from FEGAETrainer import FEGAETrainer
from Models.BiGruAE import BiGruAE
from Models.Seq2SeqGruAE import Seq2SeqGruAE

import torch
import os.path as path

class FEGAETaskConfig():
    def __init__(self, logger, modelName, showTrainingInfo=True):
        self.logger = logger
        self.modelName = modelName
        self.showTrainningInfo = showTrainingInfo

    def getConfig(self, isCuda = False):
        feature_size = 1
        extraFeatSize = 2
        output_size = 1
        forcastModel = BiGruAE(feature_size,2,10,output_size,2)
        backwardModel = BiGruAE(feature_size,2,10,output_size,2)
        errorModel = Seq2SeqGruAE(feature_size,2,10,output_size,2)
        if torch.cuda.is_available():
            forcastModel.cuda()
            errorModel.cuda()
            backwardModel.cuda()
        trainer = FEGAETrainer(forcastModel,backwardModel,errorModel, self.modelName, self.logger, 1e-3, self.showTrainningInfo)
        try:
            trainer.load()
        except:
            pass
        
        return trainer
