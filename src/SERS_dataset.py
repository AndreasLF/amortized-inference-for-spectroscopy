import pandas as pd
import torch as torch
import os
from generate_data import SERS_generator_function

# Get working directory, parent directoy, data and results directory
cwd = os.getcwd()
curr_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
results_dir = os.path.join(parent_dir, 'results')
data_dir = os.path.join(parent_dir, 'data')


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


if __name__ == "__main__":
    # dataset = SERSDataset(f"{data_dir}/SERS_data/1000_SERS_data_2023-03-15.csv")
    IterDataset(SERS_generator_function(single_spectrum = True))