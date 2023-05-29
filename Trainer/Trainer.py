from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor
import time
import torch
import os.path as path
from globalConfig import globalConfig

class Trainer():
    def __init__(self, model, logger, learningRate=1e-3, exprimentName=None, showInfo=True, windowSize=100):
        self.mlModel = model
        if torch.cuda.is_available():
            self.mlModel.cuda()
        self.lossFunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=learningRate)
        self.logger = logger
        self.experimentName = exprimentName
        self.showInfo = showInfo
        self.windowSize = windowSize

    def train(self, trainSet, trainSetLength, labelSet, context):
        self.mlModel.train()
        startTime = time.perf_counter()
        output = self.mlModel(trainSet, trainSetLength, context)
        outputedTime = time.perf_counter()
        loss = self.lossFunc(output, trainSet)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        backwardTime = time.perf_counter()
        if self.showInfo:
            print("\tloss\t", format(loss.item(), ".9f"), "\tforward\t", format(outputedTime-startTime, ".7f"), "\tlosstime\t", format(backwardTime-outputedTime, ".7f"))
        return loss

    def evalResult(self, validDataset, validsetLengths, labels, context):
        evalWindowSize= self.windowSize
        dataProcessor = SlidingWindowStepDataProcessor(evalWindowSize, 1)
        
        toEvalSplitedDataTensorList = list()
        for idx in range(len(validsetLengths)):
            toEvalDataSplitedTensor, toEvalDataLength, toEvalContext = dataProcessor.process(validDataset[idx:idx+1], [validsetLengths[idx]], context)
            toEvalOutputTensor = self.mlModel(toEvalDataSplitedTensor, toEvalDataLength, toEvalContext)
            diff = torch.abs(toEvalOutputTensor - toEvalDataSplitedTensor)
            toEvalSplitedDataTensorList.append({"data":toEvalDataSplitedTensor, "length":toEvalDataLength, "output": toEvalOutputTensor})
        
        maxF1 = 0
        maxPr = 0
        maxRc = 0
        self.bestThreshold = 0
        for threshold in [0.3, 0.2, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001, 0.0005, 0.0001, 1e-5, 1e-6, 1e-7,0]:
            truePositive = 0
            falsePostive = 0
            falseNegative = 0
            step = 20
            for evalIdx in range(0, len(validsetLengths)):
                evalOutput = toEvalSplitedDataTensorList[evalIdx]["output"]
                detectResult = evalOutput > threshold
                for labelIdx in range(0, validsetLengths[evalIdx].int().item(), evalWindowSize):
                    realPosCount = 0
                    predPosCount = 0

                    labelSegement = labels[evalIdx, labelIdx:labelIdx+evalWindowSize]
                    realPosCount = torch.sum(labelSegement)

                    knownDataLabel = labels[:, evalIdx:evalIdx + evalWindowSize]
                    evalBeginIdx = evalIdx + step - evalWindowSize
                    evalEndIdx = evalIdx + evalWindowSize - step

                    for rangeIdx in range(evalBeginIdx, evalEndIdx, step):
                        if rangeIdx >= 0 and rangeIdx < evalEndIdx:
                            diff = detectResult[rangeIdx]
                            predPosCount += torch.sum(diff).int().item()

                    # If a known anomalous window overlaps any predicted windows, a TP is recorded.
                    if realPosCount != 0 and predPosCount != 0:
                        truePositive += 1

                    # If a known anomalous window does not overlap any predicted windows, a FN is recorded.
                    elif realPosCount != 0 and predPosCount == 0:
                        falseNegative += 1

                for predIdx in range(0, evalOutput.shape[0], evalWindowSize):
                    realPosCount = 0
                    predPosCount = 0
                    diff = detectResult[predIdx]
                    predPosCount = torch.sum(diff).int().item()
                    evalBeginIdx = evalIdx + step - evalWindowSize
                    evalEndIdx = evalIdx + evalWindowSize - step

                    for rangeIdx in range(evalBeginIdx, evalEndIdx, step):
                        if rangeIdx >= 0 and rangeIdx < evalEndIdx:
                            realPosCount += torch.sum(labels[evalIdx, rangeIdx:rangeIdx+evalWindowSize]).int().item()
                    
                    # If a predicted window does not overlap any labeled anomalous region, a FP is recorded.
                    if predPosCount != 0 and realPosCount == 0:
                        falsePostive += 1

            precision = truePositive
            recall = truePositive
            f1 = -1
            if truePositive != 0:
                precision = truePositive / (truePositive + falsePostive)
                recall = truePositive / (truePositive + falseNegative)
                f1 = 2*(recall * precision) / (recall + precision)
                if f1 >= maxF1:
                    maxF1 = f1
                    maxPr = precision
                    maxRc = recall
                    self.bestThreshold = threshold
            print('\tth\t', format(threshold, '.5f'), '\tprecision\t', format(precision, '.5f'), '\trecall\t', format(recall, '.3f'), '\tf1\t', format(f1, '.5f'))    

    def recordResult(self, dataset, lengths, context, storeNames):
        self.mlModel.eval()
        lengths = lengths.int()
        for validIdx in range(len(lengths)):
            validOutput = self.reconstruct(self.mlModel, dataset, lengths, context)
            for featIdx in range(dataset.shape[2]):
                tl = validOutput[validIdx,0:lengths[validIdx],featIdx]
                t = dataset[validIdx,0:lengths[validIdx],featIdx]
                thresholdList = torch.zeros(t.shape)
                thresholdList[:] = self.bestThreshold
                thresholdList = thresholdList.reshape([-1]).tolist()
                ts = t - tl
                tlList = tl.reshape([-1]).tolist()
                tList = t.reshape([-1]).tolist()
                tsList = ts.reshape([-1]).abs().tolist()
                self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.experimentName + '-' + storeNames[validIdx] + '-feat' + str(featIdx))
                self.logger.logResults([tsList, thresholdList], ["error", "threshold"], self.experimentName + '-' + storeNames[validIdx] + '-err-feat' + str(featIdx))
    def save(self):
        filename = self.experimentName + ".pt"
        torch.save(self.mlModel.state_dict(), path.join(globalConfig.getModelPath(), filename))

    def load(self):
        filename = self.experimentName + ".pt"
        self.mlModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), filename)))

    def reconstruct(self, mlModel, validDataset, validsetLength, context):
        reconstructSeqs = torch.zeros(validDataset.shape, device=torch.device('cuda'))
        for idx in range(0, validDataset.shape[1]-self.windowSize+1, self.windowSize):
            curInput = validDataset[:,idx:idx+self.windowSize,:]
            lengths = torch.tensor(curInput.shape[1]).repeat(curInput.shape[0])
            output = mlModel(validDataset[:,idx:idx+self.windowSize,:], lengths, context)
            reconstructSeqs[:,idx:idx+self.windowSize,:] = output
        return reconstructSeqs