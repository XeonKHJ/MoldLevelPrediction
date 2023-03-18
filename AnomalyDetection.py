import torch
import torch.nn
import os.path as path
from AECastingExperiment import AECastingExperiment
from AnomalyDetectDataset import AnomalyDetectDataset
from DataProcessor.DeviationProcessor import DeviationProcessor
from FEGAECastingExperiment import FEGAECastingExperiment
from Logger.PlotCSVLogger import PlotCSVLogger
from Logger.PlotLogger import PlotLogger
from PredictDataset import CastingDataset

from globalConfig import globalConfig

from torch.utils.data import DataLoader

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is avaliable.")
    else:
        print("CUDA is unavaliable")

    logger = PlotCSVLogger(False, 'SavedPics', 'SavedCsvs')

    # experiment = RAENABExperiment(logger, "realTweets", "Twitter")
    # experiment = AECastingExperiment(logger)
    experiment = FEGAECastingExperiment(logger)
    trainer, trainDataReader, validDataReader, processers = experiment.getExperimentConfig()

    # load data
    fullDataTensor, fullDataLenghts, fullDataLabels, context, fileList = trainDataReader.read()

    # dataNormalizer.addDatasetToRef(fullDataTensor, fullDataLenghts)

    # displayDataTensor, displayDataLenghts, displayDataLabels, displayFileList = dataReader.read()
    for i in range(fullDataTensor.shape[0]):
        curList = fullDataTensor[i,0:fullDataLenghts[i].int().item(),0].reshape(-1).tolist()
        stoper = fullDataTensor[i,0:fullDataLenghts[i].int().item(),1].reshape(-1).tolist()
        logger.logResults([curList, stoper], ['lv_act','stoper'], path.splitext(path.basename(fileList[i]))[0], globalConfig.getOriginalPicturePath())

    # data preprocess
    dataTensor = fullDataTensor
    dataLengths = fullDataLenghts
    for processor in processers:
        dataTensor, dataLengths, dataContext = processor.process(dataTensor, dataLengths, context)

    # validDataTensor, validLengths, validContext = DeviationProcessor().process(fullDataTensor, fullDataLenghts, context)
    validDataTensor, validLengths, validContext =fullDataTensor, fullDataLenghts, context

    trainDataset = AnomalyDetectDataset(dataTensor, torch.zeros(dataTensor.shape), dataContext, dataLengths)
    validDataset = AnomalyDetectDataset(validDataTensor,torch.zeros(validDataTensor.shape), context, validLengths)

    # start trainning
    epoch = 0
    keepTrainning = True

    trainDataLoader = DataLoader(trainDataset, batch_size=1000, shuffle=False)
    validDataLaoder = DataLoader(validDataset, shuffle=False, batch_size = validDataTensor.shape[0])

    while keepTrainning:
        for trainData, context, trainLabels, lengths in trainDataLoader:
            loss = trainer.train(trainData, lengths, trainLabels, context)
        if epoch % 50 == 0:
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
        epoch += 1