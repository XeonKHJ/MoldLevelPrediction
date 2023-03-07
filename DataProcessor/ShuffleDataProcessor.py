import torch

# Need to make sure dataset is sorted by data lengthes from long to short.
class ShuffleDataProcessor:
    def __init__(self):
        pass

    def process(self, dataset, lengths):
        shuffleIdx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffleIdx, :, :]
        lengths = lengths[shuffleIdx]
        return dataset, lengths