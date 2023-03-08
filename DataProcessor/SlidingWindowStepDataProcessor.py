import torch

# Need to make sure dataset is sorted by data lengthes from long to short.
class SlidingWindowStepDataProcessor:
    def __init__(self, windowSize, step, padRemainWindow=False):
        self.windowSize = windowSize
        self.step = step
        self.padRemainWindow = padRemainWindow

    def process(self, data, lengths, context):
        newData = []
        lengthsTensor = torch.tensor(lengths)
        newContext = []
        for idx in range(0, data.shape[1], self.step):
            inRangeTensor = idx+self.windowSize <= lengthsTensor
            inRangeCount = torch.sum(inRangeTensor).int().item()
            if inRangeCount > 0:
                segementedData = data[0:inRangeCount, idx:idx+self.windowSize,:]
                newData.append(segementedData)
                newContext.append(context[0:inRangeCount])
        newData = torch.cat(newData, 0)
        newContext = torch.cat(newContext, 0)
        newLengths = torch.tensor(newData.shape[1]).repeat(newData.shape[0])
        if torch.cuda.is_available():
            newLengths = newLengths.cuda()
        return newData, newLengths, newContext