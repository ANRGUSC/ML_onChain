import torch
import pandas as pd
import ml_models
from torch import nn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MyDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def data_import(filename:str):
    # Load your dataset
    df = pd.read_csv(filename)
    # Drop the 'id' column
    df = df.drop(["id"],axis=1)
    # Map diagnosis values to 0 (negative) and 1 (positive)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    return df