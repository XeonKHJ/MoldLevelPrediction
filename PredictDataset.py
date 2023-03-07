import torch.utils.data.dataset
import datetime, os, re, json, pandas, torch

class CastingDataset(object):
    def __init__(self, data, length):
        self.data = data[:, :, 0:1]
        self.labels = data[:, :, 1:2]
        self.lengths = length[:]
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.lengths[idx]