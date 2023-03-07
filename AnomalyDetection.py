import torch
import torch.nn
import os.path as path
from AnomalyDetectDataset import AnomalyDetectDataset
from FEGAECastingExperiment import FEGAECastingExperiment
from PlotLogger import PlotLogger
from PredictDataset import CastingDataset

from globalConfig import globalConfig

from torch.utils.data import DataLoader

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is avaliable.")
    else:
        print("CUDA is unavaliable")

    logger = PlotLogger(False)

    # experiment = RAENABExperiment(logger, "realTweets", "Twitter")
    experiment = FEGAECastingExperiment(logger)
    trainer, trainDataReader, validDataReader, processers = experiment.getExperimentConfig()

    # load data
    fullDataTensor, fullDataLenghts, fullDataLabels, context, fileList = trainDataReader.read()

    # dataNormalizer.addDatasetToRef(fullDataTensor, fullDataLenghts)

    # displayDataTensor, displayDataLenghts, displayDataLabels, displayFileList = dataReader.read()
    for i in range(fullDataTensor.shape[0]):
        curList = fullDataTensor[i,0:fullDataLenghts[i].int().item(),0].reshape(-1).tolist()
        logger.logResults([curList], ['lv_act'], 'lv-' + path.splitext(path.basename(fileList[i]))[0], globalConfig.getOriginalPicturePath())
        stoper = fullDataTensor[i,0:fullDataLenghts[i].int().item(),1].reshape(-1).tolist()
        logger.logResults([curList], ['stoper'], 'stp-' + path.splitext(path.basename(fileList[i]))[0], globalConfig.getOriginalPicturePath())

    # data preprocess
    dataTensor = fullDataTensor
    dataLengths = fullDataLenghts
    for processor in processers:
        dataTensor, dataLengths = processor.process(dataTensor, dataLengths)

    trainDataset = AnomalyDetectDataset(dataTensor, fullDataLabels, context, dataLengths)
    testDataset = AnomalyDetectDataset(dataTensor, fullDataLabels, context, dataLengths)
    validDataset = AnomalyDetectDataset(dataTensor,fullDataLabels, context, dataLengths)

    # start trainning
    epoch = 0
    keepTrainning = True

    trainDataLoader = DataLoader(trainDataset, batch_size=fullDataTensor.shape[0], shuffle=False)
    testDataLoader = DataLoader(testDataset, shuffle=False, batch_size=fullDataTensor.shape[0])
    validDataLaoder = DataLoader(validDataset, shuffle=False, batch_size = fullDataTensor.shape[0])

    while keepTrainning:
        for trainData, context, trainLabels, lengths in trainDataLoader:
            loss = trainer.train(trainData, lengths, trainLabels, context)
        if epoch % 50 == 0:
            trainer.save()
            # for testData, testLabels in testDataLoader:
            #     lengths = testLabels[:, testLabels.shape[1]-1]
            #     labels = testLabels[:, 0:testLabels.shape[1]-1]            
            #     trainer.evalResult(testData, lengths, labels)
            for validData, validLabels in validDataLaoder:
                lengths = validLabels[:, validLabels.shape[1]-1]
                labels = validLabels[:, 0:validLabels.shape[1]-1] 
                newFileList = list() 
                for fileName in validFileList:
                     newFileList.append(path.splitext(path.basename(fileName))[0])
                trainer.evalResult(validData, lengths, labels)  
                trainer.recordResult(validData, lengths, newFileList)           
        epoch += 1