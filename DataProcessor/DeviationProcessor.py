import torch

# Need to make sure dataset is sorted by data lengthes from long to short.
class DeviationProcessor:
    def __init__(self):
        pass

    def process(self, dataset, lengths, context):
        if torch.has_cuda:
            offsetedDataset = torch.zeros([dataset.shape[0], dataset.shape[1]-1, dataset.shape[2]], device=torch.device('cuda'))
        else:
            offsetedDataset = torch.zeros([dataset.shape[0], dataset.shape[1]-1, dataset.shape[2]])
        offsetedDataset[:, :, :] = dataset[:, 0:dataset.shape[1]-1,:]
        offsetedDataset[:,:,1] = dataset[:,1:dataset.shape[1],1] - offsetedDataset[:,:,1]
        offsetedDataset[:,:,0] = dataset[:,0:dataset.shape[1]-1,0] 
        # offsetedDataset[:,:,2] = dataset[:,0:dataset.shape[1]-1,2] 
        lengths = lengths - 1
        return offsetedDataset, lengths, context