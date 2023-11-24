import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class MyDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def data_import_and_process(filename: str):
    # Load your dataset
    df = pd.read_csv(filename)

    # Convert diagnosis to binary
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

    # Separate labels and features
    features = df.columns[2:]  # All columns except id and diagnosis

    # Normalize the data (using Min-Max normalization for simplicity)
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    return df
