import torch.utils.data.dataset
import datetime, os, re, json, pandas, torch

class AnomalyDetectDataset(object):
    def __init__(self, data, labels, contexts, length):
        self.data = data[:, :, 1:2]
        self.labels = labels
        self.length = length
        self.context = contexts
        
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.context[idx], self.labels[idx], self.length[idx]