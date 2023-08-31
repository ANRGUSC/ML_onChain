import pandas as pd
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def data_import(filename: str):
    # Load your dataset
    df = pd.read_csv(filename)
    return df
