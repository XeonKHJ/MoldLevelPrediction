import torch

# Need to make sure dataset is sorted by data lengthes from long to short.
class ExtractValidProcessor:
    def __init__(self):
        pass

    def process(self, dataset, lengths):
        newDataset = list()
        for i, data in enumerate(dataset):
            startIdx = 0
            endIdx = 0
            for idx, j in enumerate(data):
                if j[1] > 1 and startIdx == 0:
                    startIdx = idx
                if j[2] > 0 and endIdx == 0:
                    endIdx = idx
            newDataset.append(data[startIdx: endIdx])
                    

        offsetedDataset = torch.zeros(dataset.shape)
        offsetedDataset[:, 1:offsetedDataset.shape[1], :] = dataset[:, 0:dataset.shape[1]-1,:]
        resultSet = dataset[:,:,1] - offsetedDataset[:,:,1]
        resultSet = resultSet[:, 1:resultSet.shape[1], :]
        lengths = lengths - 1
        return resultSet, lengths