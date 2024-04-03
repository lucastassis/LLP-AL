import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageLLPDataset(Dataset):
    def __init__(self, data, targets, bags):
        self.data = data
        self.targets = targets
        self.bags = bags
        self.pd_targets = pd.Series(targets, dtype="category")
    def __len__(self):
        return len(np.unique(self.bags))

    def __getitem__(self, idx):
        bag_data = np.array(self.data[self.bags == idx], dtype=np.float64)
        bag_bags = self.bags[self.bags == idx]
        bag_targets = self.targets[self.bags == idx]
        bag_prop = np.array(self.pd_targets[self.bags == idx].value_counts().sort_index().to_numpy() / len(bag_bags))
        return bag_data, bag_targets, bag_bags, bag_prop

class ImageDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.y = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]



