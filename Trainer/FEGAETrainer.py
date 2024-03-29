import pandas as pd
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor
import time
import torch
import torch.nn
import os.path as path
from DilateLoss import DialteLoss
from Utils import DynamicThreshold

from globalConfig import globalConfig
from loss.dilate_loss import dilate_loss

class FEGAETrainer():
    def __init__(self, forcastModel, backwardModel, errorModel, taskName, logger, learningRate=1e-3, showTrainningInfo=True, windowSize=100):
        self.forcastModel = forcastModel
        self.backwardModel = backwardModel
        self.errorModel = errorModel
        self.lossFunc = torch.nn.MSELoss()
        # self.dialteLoss = DialteLoss()
        self.forcastOptimizer = torch.optim.Adam(self.forcastModel.parameters(), lr=learningRate)
        self.errorOptimizer = torch.optim.Adam(self.errorModel.parameters(), lr=learningRate)
        self.backwardOptimzer = torch.optim.Adam(self.backwardModel.parameters(), lr=learningRate)
        self.modelName = taskName
        self.logger = logger
        self.windowSize = windowSize
        self.step = 1
        self.splitData = None
        self.showTrainningInfo = showTrainningInfo
        self.lambda1 = 0.7
        self.lambda2 = 1e-17
        self.toRecordThresholds = None
        self.toRecordDiffs =None

        if torch.cuda.is_available():
            self.initDevice = torch.device('cuda')
        else:
            self.initDevice = torch.device('cpu')

        self.recordLoss = {'dialte':list(),'mse':list()}

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
        # realLoss = self.lossFunc(output, latterSet)
        realLoss = self.lossFunc(output, latterSet)
        dialteloss,tempLoss, shapeLoss= dilate_loss(t, latterSet, 0.5, 0.001, device=torch.device('cuda'))
        zeros = torch.zeros(error.shape, device=torch.device('cuda'))
        mseloss = self.lossFunc(t, latterSet)
        # errorExpand = torch.norm(error, p=2) / (error.shape[0] * error.shape[1] * error.shape[2])
        errorExpand = self.lossFunc(error, zeros)
        totalLoss = mseloss + 10 * errorExpand
        totalLoss.backward()
        self.forcastOptimizer.step()
        self.errorOptimizer.step()


        self.forcastOptimizer.zero_grad()
        self.errorOptimizer.zero_grad()
        error = self.errorModel(trainSet, lengths, context, int(trainSet.shape[1] / 2)).detach()
        output = self.forcastModel(preSet, lengths / 2, context)
        tl = latterSet - error
        forcastLoss = self.lossFunc(output, tl)
        forcastLoss.backward()
        self.forcastOptimizer.step()
        # self.errorOptimizer.step()

        # update error model
        self.errorOptimizer.zero_grad()
        error = self.errorModel(trainSet, lengths, context, int(trainSet.shape[1] / 2))
        output = self.forcastModel(preSet, lengths / 2, context).detach()
        diff = latterSet - output
        errorLoss = self.lossFunc(error, diff)
        errorLoss.backward()
        self.errorOptimizer.step()

        self.backwardOptimzer.zero_grad()
        backwardOutput = self.backwardModel(latterSet, lengths / 2, context)
        backLoss = self.lossFunc(backwardOutput, preSet)
        backLoss.backward()
        self.backwardOptimzer.step()
        
        backwardTime = time.perf_counter()
        if self.showTrainningInfo:
            print("real\t", format(realLoss.item(), ".5f"),
                  "\tfore\t", format(forcastLoss.item(),".5f"),  
                  "\ttotal\t", format(totalLoss.item(),".5f"), 
                  '\terror\t', format(errorExpand.item(), ".5f"))

        self.recordLoss['mse'].append(mseloss.item())
        self.recordLoss['dialte'].append(forcastLoss.item())

        return errorLoss

    def saveLoss(self):
        toSave = {"mse": self.recordLoss['mse'],"dialte": self.recordLoss['dialte']}
        df = pd.DataFrame(toSave)
        df.to_csv(path.join('fegaloss.csv'), index=False)

    def evalResult(self, validDataset, validsetLengths, labels, context):
        self.forcastModel.eval()
        reconstructData = self.reconstruct(self.forcastModel, validDataset, validsetLengths, context)
        self.toRecordThresholds = None
        self.toRecordDiffs =None
        evalWindowSize = self.windowSize
        step = 5
        thresholders = list()
        for meanRate in [0.4,0.3, 0.2, 0.1]:
            for stdRate in [1, 0.75, 0.5, 0.4, 0.3]:
                thresholders.append(DynamicThreshold(meanRate, stdRate,evalWindowSize))
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

            
            # if f1 >= maxf1:
            #     maxf1 = f1
            #     self.toRecordThresholds = thresholds

            if threadholder.stdRate == 0.3 and threadholder.meanRate == 0.1:
                maxf1 = f1
                self.toRecordThresholds = thresholds

            print('stdrate', threadholder.stdRate, '\t', threadholder.meanRate, '\tth\t', format(meanRate, '.5f'), '\tprecision\t', format(precision, '.5f'), '\trecall\t', format(recall, '.3f'), '\tf1\t', format(f1, '.5f')) 

    def recordResult(self, dataset, lengths, context, storeNames):
        self.forcastModel.eval()
        lengths = lengths.int()
        validOutput = self.reconstruct(self.forcastModel, dataset, lengths, context)
        errorOutput = self.reconstructError(dataset, lengths, context)
        sumOutput = validOutput + errorOutput
        threshold = 0.1
        self.saveLoss()
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

    def reconstruct(self, mlModel, validDataset, validsetLength, context):
        reconstructSeqs = torch.zeros(validDataset.shape, device=self.initDevice)
        preIdx = -100
        halfWindowSize = int(self.windowSize/2)
        for idx in range(0, validDataset.shape[1] - self.windowSize, int(self.windowSize/2)):
            if idx+2*self.windowSize > reconstructSeqs.shape[1]:
                break
            lengths = torch.tensor(halfWindowSize).repeat(validDataset.shape[0]).int()
            reconstructSeqs[:,idx+halfWindowSize:idx+self.windowSize,:] = mlModel(validDataset[:,idx:idx+halfWindowSize,:], lengths, context)
            preIdx = idx
            
        lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
        reconstructSeqs[:,reconstructSeqs.shape[1]-self.windowSize:reconstructSeqs.shape[1],:] = mlModel(validDataset[:,validDataset.shape[1]-2*self.windowSize:validDataset.shape[1]-self.windowSize:,:], lengths, context)
        reconstructSeqs[:,0:self.windowSize, :] = self.backwardModel(validDataset[:, self.windowSize:2*self.windowSize, :], lengths, context)
        return reconstructSeqs

    def reconstructError(self, validDataset, validsetLength, context):
        reconstructSeqs = torch.zeros(validDataset.shape, device=self.initDevice)
        preIdx = -100
        halfWindowSize = int(self.windowSize / 2)
        for idx in range(0, validDataset.shape[1] - 2 * self.windowSize, halfWindowSize):
            if idx+2*self.windowSize > reconstructSeqs.shape[1]:
                break
            lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
            reconstructSeqs[:,idx+halfWindowSize:idx+self.windowSize,:] = self.errorModel(validDataset[:,idx:idx+self.windowSize,:], lengths, context, halfWindowSize)
            preIdx = idx
            
        lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
        reconstructSeqs[:,0:self.windowSize, :] = 0
        return reconstructSeqs