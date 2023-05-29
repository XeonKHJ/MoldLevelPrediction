import pandas as pd
import torch
import torch.nn
import os.path as path
from Experiment.AECastingExperiment import AECastingExperiment
from Dataset.AnomalyDetectDataset import AnomalyDetectDataset
from DataProcessor.DeviationProcessor import DeviationProcessor
from Experiment.DeviationExperiment import DeviationExperiment
from Experiment.FEGAECastingExperiment import FEGAECastingExperiment
from Experiment.ValidAEExperiment import ValidAEExperiment
from Experiment.ValidDeviationExperiment import ValidDeviationExperiment
from Experiment.ValidSeq2SeqExperiment import ValidSeq2SeqExperiment
from Logger.PlotCSVLogger import PlotCSVLogger
from Logger.PlotLogger import PlotLogger
from PredictDataset import CastingDataset

from globalConfig import globalConfig

from torch.utils.data import DataLoader




def saveLosses(losses, filename):
    toSave = {"loss": losses}
    df = pd.DataFrame(toSave)
    df.to_csv(path.join(filename+".csv"), index=False)
    

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is avaliable.")
    else:
        print("CUDA is unavaliable")

    logger = PlotCSVLogger(False, globalConfig.getSavedPicturePath(), globalConfig.getCsvPath())

    # experiment = RAENABExperiment(logger, "realTweets", "Twitter")
    # experiment = AECastingExperiment(logger)
    # experiment = ValidSeq2SeqExperiment(logger)
    # experiment = ValidDeviationExperiment(logger)
    # experiment = FEGAECastingExperiment(logger)
    experiment = ValidAEExperiment(logger)
    trainer, trainDataReader, validDataReader, processers = experiment.getExperimentConfig()

    # load data
    fullDataTensor, fullDataLenghts, fullDataLabels, context, fileList = trainDataReader.read()

    # dataNormalizer.addDatasetToRef(fullDataTensor, fullDataLenghts)

    # displayDataTensor, displayDataLenghts, displayDataLabels, displayFileList = dataReader.read()
    for i in range(fullDataTensor.shape[0]):
        stopperPoses = fullDataTensor[i,0:fullDataLenghts[i].int().item(),0].reshape(-1).tolist()
        liqLevels = fullDataTensor[i,0:fullDataLenghts[i].int().item(),1].reshape(-1).tolist()
        # logger.logResults([stopperPoses, liqLevels], ['stoper','liqlv'], path.splitext(path.basename(fileList[i]))[0], globalConfig.getOriginalPicturePath())
        logger.logResults([liqLevels], ['liqlv'], path.splitext(path.basename(fileList[i]))[0], globalConfig.getOriginalPicturePath())

    # data preprocess
    dataTensor = fullDataTensor[0:7]
    dataLengths = fullDataLenghts[0:7]
    dataContext = context
    for processor in processers:
        dataTensor, dataLengths, dataContext = processor.process(dataTensor, dataLengths, context)

    validDataTensor, validLengths, validContext = DeviationProcessor().process(fullDataTensor, fullDataLenghts, context)
    # validDataTensor, validLengths, validContext =fullDataTensor, fullDataLenghts, context

    trainDataset = AnomalyDetectDataset(dataTensor, torch.zeros(dataTensor.shape), dataContext, dataLengths)
    validDataset = AnomalyDetectDataset(validDataTensor,torch.zeros(validDataTensor.shape), context, validLengths)

    # start trainning
    epoch = 0
    keepTrainning = True

    trainDataLoader = DataLoader(trainDataset, batch_size=100, shuffle=False)
    validDataLaoder = DataLoader(validDataset, shuffle=False, batch_size = validDataTensor.shape[0])

    losses = []
    curLosses = list()
    while keepTrainning:
        for trainData, context, trainLabels, lengths in trainDataLoader:
            loss = trainer.train(trainData, lengths, trainLabels, context)
            curLosses.append(loss.item())
        if epoch % 100 == 0:
            trainer.save()
            # for testData, testLabels in testDataLoader:
            #     lengths = testLabels[:, testLabels.shape[1]-1]
            #     labels = testLabels[:, 0:testLabels.shape[1]-1]            
            #     trainer.evalResult(testData, lengths, labels)
            for validData, validContext, validLabels, validLengths in validDataLaoder:
                newFileList = list() 
                for fileName in fileList:
                     newFileList.append(path.splitext(path.basename(fileName))[0])
                trainer.evalResult(validData, validLengths, validLabels, validContext)  
                trainer.recordResult(validData, validLengths, validContext, newFileList) 
            saveLosses(curLosses, "losses.csv")
        epoch += 1