import pandas as pd
import torch as torch
import numpy as np

class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator

class SERSDataset(torch.utils.data.Dataset):
    def __init__(self,file_name):
        file_out = pd.read_csv(file_name)
        # x is the first 500 columns
        X = file_out.iloc[:,0:500].values
        # y is the last column
        y = file_out.iloc[:,-1].values

        self.X_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class SERSDatasetNPY(torch.utils.data.Dataset):
    def __init__(self,file_name, file_name_labels=None):
        X = np.load(file_name)
        self.X = torch.tensor(X, dtype=torch.float32)

        y = np.load(file_name_labels)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[:,idx] 