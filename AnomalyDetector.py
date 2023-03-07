import torch
import torch.nn
from torch.utils.data import DataLoader
from AnomalyDetectDataset import AnomalyDetectDataset
from DatasetReader.CastingDataReader import CastingDataReader

datasetReader = CastingDataReader("./datasets")

if __name__ == '__main__':
    data, lenghts, context = datasetReader.read()
    model = Lstm()
    dataset = AnomalyDetectDataset(data, lengths)
    dataLoader = DataLoader(data, shuffle=False, batch_size = data.shape[0])
    while True:
        for data, label in dataLoader:
            output = model(data)