import torch
from torch.utils.data import Dataset
import numpy as np


class ECGDataset(Dataset):

    def __init__(self, X_path, y_path):

        self.X = np.load(X_path)
        self.y = np.load(y_path)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]