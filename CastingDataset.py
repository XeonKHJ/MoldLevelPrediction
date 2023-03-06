import torch.utils.data.dataset
import datetime, os, re, json, pandas, torch

class CastingDataset(object):
    def __init__(self, data, length):
        self.data = data[:, :, 0:data.shape[2]-1]
        self.labels = data[:, :, data.shape[2]-1]
        self.labels = torch.cat((self.labels, length.reshape(-1, 1)),1)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]