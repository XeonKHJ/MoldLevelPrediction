import torch

class DynamicThreshold(object):
    def __init__(self, meanRate=0.3, stdRate=0.5, windowSize=100) -> None:
        self.meanRate = meanRate
        self.stdRate = stdRate
        self.windowSize = windowSize

    def getThreshold(self, tensor, lengths):
        threasholds = torch.zeros(tensor.shape)
        lengths = lengths.int().tolist()
        for dataIdx in range(len(lengths)):
            for idx in range(0, lengths[dataIdx], self.windowSize):
                curWindowData = tensor[dataIdx, idx:idx+self.windowSize, :]
                std, mean = torch.std_mean(curWindowData)
                threshold = self.stdRate * std + self.meanRate * mean + 1e-2
                threasholds[dataIdx, idx:idx+self.windowSize, :] = threshold
        return threasholds

    def getDiffs(self, originalTensor, evalTensor, lengths):
        diffs = torch.zeros(originalTensor.shape)
        lengths = lengths.int().tolist()
        for dataIdx in range(len(lengths)):
            for idx in range(0, lengths[dataIdx], self.windowSize):
                windowedOgData = originalTensor[dataIdx, idx:idx+self.windowSize, :]
                windowedEvalData = evalTensor[dataIdx, idx:idx+self.windowSize, :]
                diff = torch.sqrt(torch.square(windowedOgData - windowedEvalData).sum()/self.windowSize)
                diffs[dataIdx, idx:idx+self.windowSize, :] = diff
        return diffs        

    def compare(self, thresholds, originalTensor, evalTensor, lengths):
        diffs = torch.zeros(originalTensor.shape)
        lengths = lengths.int().tolist()
        for dataIdx in range(len(lengths)):
            for idx in range(0, lengths[dataIdx], self.windowSize):
                windowedOgData = originalTensor[dataIdx, idx:idx+self.windowSize, :]
                windowedEvalData = evalTensor[dataIdx, idx:idx+self.windowSize, :]
                diff = torch.sqrt(torch.square(windowedOgData - windowedEvalData).sum()/self.windowSize)
                diffs[dataIdx, idx:idx+self.windowSize, :] = diff
        return diffs > thresholds
    

class PaperDynamicThreshold(object):
    def __init__(self, rate, windowSize) -> None:
        self.rate = rate
        self.windowSize = windowSize

    def getThreshold(self, tensor):


        threasholds = torch.zeros(tensor.shape)
        for idx in range(0, tensor.shape[1], self.windowSize):
            curWindowData = tensor[:, idx:idx+self.windowSize, :]
            std, mean = torch.std_mean(curWindowData, 1)
            threshold = std + self.rate * mean
            threasholds[:, idx:idx+self.windowSize, :] = threshold
        return threasholds

    def compare(self, thresholds, originalTensor, evalTensor):
        diffs = torch.zeros(originalTensor.shape)
        for idx in range(0, originalTensor.shape[1], self.windowSize):
            windowedOgData = originalTensor[:, idx:idx+self.windowSize, :]
            windowedEvalData = evalTensor[:, idx:idx+self.windowSize, :]
            diff = torch.sqrt(torch.square(windowedOgData - windowedEvalData).sum(1)/self.windowSize)
            diffs[:, idx:idx+self.windowSize, :] = diff
        return diffs > thresholds