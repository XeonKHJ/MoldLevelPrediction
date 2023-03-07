from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor
import time
import torch
import os.path as path
from Utils import DynamicThreshold

from globalConfig import globalConfig

class FEGAETrainer():
    def __init__(self, forcastModel, backwardModel, errorModel, taskName, logger, learningRate=1e-3, showTrainningInfo=True):
        self.forcastModel = forcastModel
        self.backwardModel = backwardModel
        self.errorModel = errorModel
        self.lossFunc = torch.nn.MSELoss()
        self.forcastOptimizer = torch.optim.Adam(self.forcastModel.parameters(), lr=learningRate)
        self.errorOptimizer = torch.optim.Adam(self.errorModel.parameters(), lr=learningRate)
        self.backwardOptimzer = torch.optim.Adam(self.backwardModel.parameters(), lr=learningRate)
        self.modelName = taskName
        self.logger = logger
        self.windowSize = 100
        self.step = 1
        self.splitData = None
        self.showTrainningInfo = showTrainningInfo
        self.lambda1 = 0.8
        self.lambda2 = 1e-3
        self.toRecordThresholds = None
        self.toRecordDiffs =None

    def train(self, trainSet, lengths, labelSet, context):
        self.forcastModel.train()
        self.errorModel.train()
        self.backwardModel.train()
        startTime = time.perf_counter()
        preSet = trainSet[:,0:int(trainSet.shape[1]/2), :]
        latterSet = trainSet[:,int(trainSet.shape[1]/2):trainSet.shape[1], :]
        
        self.forcastOptimizer.zero_grad()
        self.errorOptimizer.zero_grad()
        error = self.errorModel(trainSet, lengths, context, int(trainSet.shape[1] / 2))
        output = self.forcastModel(preSet, lengths / 2, context)
        t = output + error
        loss2 = self.lossFunc(t, latterSet) + self.lambda2 * 1/torch.norm(error, p=1)
        loss2.backward()
        self.forcastOptimizer.step()
        self.errorOptimizer.step()

        self.forcastOptimizer.zero_grad()
        self.errorOptimizer.zero_grad()
        error = self.errorModel(trainSet, lengths, context, int(trainSet.shape[1] / 2)).detach()
        output = self.forcastModel(preSet, lengths / 2, context)
        tl = latterSet - error
        forcastLoss = self.lossFunc(output, tl)
        realLoss = self.lossFunc(output, latterSet)
        loss1 = forcastLoss + self.lambda1 * realLoss
        loss1.backward()
        self.forcastOptimizer.step()
        # self.errorOptimizer.step()

        # update error model
        self.errorOptimizer.zero_grad()
        error = self.errorModel(trainSet, lengths, context, int(trainSet.shape[1] / 2))
        output = self.forcastModel(preSet, lengths / 2, context).detach()
        diff = latterSet - output
        loss3 = self.lossFunc(error, diff)
        loss3.backward()
        self.errorOptimizer.step()

        self.backwardOptimzer.zero_grad()
        backwardOutput = self.backwardModel(latterSet, lengths / 2, context)
        realLoss = self.lossFunc(backwardOutput, preSet)
        realLoss.backward()
        self.backwardOptimzer.step()
        
        backwardTime = time.perf_counter()
        if self.showTrainningInfo:
            # print(loss3)
            print("\tforcast\t", format(forcastLoss.item(), ".7f"), "\treal\t", format(realLoss.item(), ".7f"),"\tloss1\t", format(loss1.item(),".7f"),  "\tloss2\t", format(loss2.item(),".7f"), '\terror\t', torch.norm(error, p=1).item())

        return loss3

    def evalResult(self, validDataset, validsetLengths, labels):
        self.forcastModel.eval()
        reconstructData = self.reconstruct(self.forcastModel, validDataset, validsetLengths)
        self.toRecordThresholds = None
        self.toRecordDiffs =None
        evalWindowSize = 100
        step = 20
        thresholders = list()
        for threadhold in [0.4,0.3, 0.2, 0.1]:
            for stdMean in [1, 0.75, 0.5, 0.4, 0.3]:
                thresholders.append(DynamicThreshold(threadhold, stdMean,evalWindowSize))
        maxf1 = 0
        for threadholder in thresholders:
            truePositive = 0
            falsePostive = 0
            falseNegative = 0
            
            thresholds = threadholder.getThreshold(validDataset, validsetLengths)
            if self.toRecordDiffs == None:
                self.toRecordDiffs = threadholder.getDiffs(validDataset, reconstructData, validsetLengths)
                
            compareResult = threadholder.compare(thresholds, validDataset, reconstructData, validsetLengths)
            for dataIdx in range(0, len(validsetLengths)):
                detectResult = compareResult[dataIdx, 0:validsetLengths[dataIdx].int().item()]
                curLabel = labels[dataIdx, 0:validsetLengths[dataIdx].int().item()]
                compareResultWindows = list()
                labelWindows = list()
                for windowIdx in range(0, validsetLengths[dataIdx].int().item() - evalWindowSize + 1):
                    compareResultWindows.append(detectResult[windowIdx:windowIdx+evalWindowSize].reshape(-1, evalWindowSize))
                    labelWindows.append(curLabel[windowIdx:windowIdx+evalWindowSize].reshape(-1, evalWindowSize))
                compareResultWindows = torch.cat(compareResultWindows, 0)
                labelWindows = torch.cat(labelWindows, 0)
                for windowIdx in range(0, validsetLengths[dataIdx].int().item(), evalWindowSize):
                    realPosCount = 0
                    predPosCount = 0

                    labelSegement = labels[dataIdx, windowIdx:windowIdx+evalWindowSize]
                    realPosCount = torch.sum(labelSegement)

                    evalBeginIdx = windowIdx + step - evalWindowSize
                    evalEndIdx = windowIdx + evalWindowSize - step

                    for rangeIdx in range(evalBeginIdx, evalEndIdx, step):
                        if rangeIdx >= 0 and rangeIdx < evalEndIdx:
                            diff = detectResult[rangeIdx:rangeIdx+evalWindowSize]
                            predPosCount += torch.sum(diff).int().item()

                    # If a known anomalous window overlaps any predicted windows, a TP is recorded.
                    if realPosCount != 0 and predPosCount != 0:
                        truePositive += 1

                    # If a known anomalous window does not overlap any predicted windows, a FN is recorded.
                    elif realPosCount != 0 and predPosCount == 0:
                        falseNegative += 1

                for predIdx in range(0, detectResult.shape[0], evalWindowSize):
                    realPosCount = 0
                    predPosCount = 0
                    diff = detectResult[predIdx:predIdx+evalWindowSize]
                    predPosCount = torch.sum(diff).int().item()
                    evalBeginIdx = predIdx + step - evalWindowSize
                    evalEndIdx = predIdx + evalWindowSize - step

                    for rangeIdx in range(evalBeginIdx, evalEndIdx, step):
                        if rangeIdx >= 0 and rangeIdx < evalEndIdx:
                            realPosCount += torch.sum(labels[dataIdx, rangeIdx:rangeIdx+evalWindowSize]).int().item()
                    
                    # If a predicted window does not overlap any labeled anomalous region, a FP is recorded.
                    if predPosCount != 0 and realPosCount == 0:
                        falsePostive += 1

            precision = truePositive
            recall = truePositive
            f1 = 0
            if truePositive != 0:
                precision = truePositive / (truePositive + falsePostive)
                recall = truePositive / (truePositive + falseNegative)
                f1 = 2*(recall * precision) / (recall + precision)

            
            if f1 >= maxf1:
                maxf1 = f1
                self.toRecordThresholds = thresholds
            print('stdrate', threadholder.stdRate, '\t', threadholder.meanRate, '\tth\t', format(threadhold, '.5f'), '\tprecision\t', format(precision, '.5f'), '\trecall\t', format(recall, '.3f'), '\tf1\t', format(f1, '.5f')) 

    def recordResult(self, dataset, lengths, storeNames):
        self.forcastModel.eval()
        lengths = lengths.int()
        validOutput = self.reconstruct(self.forcastModel, dataset, lengths)
        errorOutput = self.reconstructError(dataset, lengths)
        sumOutput = validOutput + errorOutput
        threshold = 0.1
        for validIdx in range(len(lengths)):
            for featIdx in range(dataset.shape[2]):
                tl = validOutput[validIdx,0:lengths[validIdx],featIdx]
                error = errorOutput[validIdx, 0:lengths[validIdx], featIdx]
                t = dataset[validIdx,0:lengths[validIdx],featIdx]
                sum = sumOutput[validIdx, 0:lengths[validIdx], featIdx]
                ts = t - tl
                tlList = tl.reshape([-1]).tolist()
                tList = t.reshape([-1]).tolist()
                tsNoAbs = ts.reshape([-1]).tolist()
                tsList = ts.reshape([-1]).abs().tolist()
                errorList = error.reshape([-1]).tolist()
                sumList = sum.reshape([-1]).tolist()
                fixedThresholdList = torch.zeros(ts.shape)
                fixedThresholdList[:] = threshold
                fixedThresholdList = fixedThresholdList.reshape([-1]).tolist()
                self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.modelName + '-' + storeNames[validIdx] + '-forcast-feat' + str(featIdx))
                self.logger.logResults([tList, tsNoAbs, errorList], ["t", "tsNoAbs", "error"], self.modelName + '-' + storeNames[validIdx] + '-error-feat' + str(featIdx))
                self.logger.logResults([tList, sumList], ["t", "sum"], self.modelName + '-' + storeNames[validIdx] + '-sum-feat' + str(featIdx))
                self.logger.logResults([tsList, fixedThresholdList], ["ts", "fixed threshold"], self.modelName + '-' + storeNames[validIdx] + '-fixthreshold-feat' + str(featIdx))
                if self.toRecordThresholds != None:
                    self.logger.logResults([self.toRecordDiffs[validIdx, 0:lengths[validIdx]].abs().reshape(-1).tolist(), self.toRecordThresholds[validIdx, 0:lengths[validIdx]].reshape(-1).tolist()], ["error", "treshold"], self.modelName + '-' + storeNames[validIdx] + '-threshold-' '-feat' + str(featIdx))
    def save(self):
        forcastModelName = self.modelName + "-forcast.pt"
        errorModelName = self.modelName + "-error.pt"
        backwardName = self.modelName+"-backward.pt"
        torch.save(self.forcastModel.state_dict(), path.join(globalConfig.getModelPath(), forcastModelName))
        torch.save(self.errorModel.state_dict(), path.join(globalConfig.getModelPath(), errorModelName))
        torch.save(self.backwardModel.state_dict(), path.join(globalConfig.getModelPath(), backwardName))

    def load(self):
        forcastModelName = self.modelName + "-forcast.pt"
        errorModelName = self.modelName + "-error.pt"
        backwardName = self.modelName+"-backward.pt"
        self.forcastModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), forcastModelName))) 
        self.errorModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), errorModelName))) 
        self.backwardModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), backwardName))) 

    def reconstruct(self, mlModel, validDataset, validsetLength):
        reconstructSeqs = torch.zeros(validDataset.shape, device=torch.device('cuda'))
        preIdx = -100
        for idx in range(0, validDataset.shape[1] - self.windowSize, self.windowSize):
            if idx+2*self.windowSize > reconstructSeqs.shape[1]:
                break
            lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
            reconstructSeqs[:,idx+self.windowSize:idx+2*self.windowSize,:] = mlModel(validDataset[:,idx:idx+self.windowSize,:], lengths)
            preIdx = idx
            
        lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
        reconstructSeqs[:,reconstructSeqs.shape[1]-self.windowSize:reconstructSeqs.shape[1],:] = mlModel(validDataset[:,validDataset.shape[1]-2*self.windowSize:validDataset.shape[1]-self.windowSize:,:], lengths)
        reconstructSeqs[:,0:self.windowSize, :] = self.backwardModel(validDataset[:, self.windowSize:2*self.windowSize, :], lengths)
        return reconstructSeqs

    def reconstructError(self, validDataset, validsetLength):
        reconstructSeqs = torch.zeros(validDataset.shape, device=torch.device('cuda'))
        preIdx = -100
        for idx in range(0, validDataset.shape[1] - 2 * self.windowSize, int(self.windowSize/4)):
            if idx+2*self.windowSize > reconstructSeqs.shape[1]:
                break
            lengths = torch.tensor(2*self.windowSize).repeat(validDataset.shape[0]).int()
            reconstructSeqs[:,idx+self.windowSize:idx+2*self.windowSize,:] = self.errorModel(validDataset[:,idx:idx+2*self.windowSize,:], lengths, self.windowSize)
            preIdx = idx
            
        lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
        reconstructSeqs[:,0:self.windowSize, :] = 0
        return reconstructSeqs