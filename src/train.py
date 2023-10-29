from data_import import *
import models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json


# convert tensor into list since tensor objects are not json serializable
def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()


# convert the weights_biases to json
def state_dict_to_json(state_dict):
    state_dict_serializable = {name: tensor_to_list(param) for name, param in state_dict.items()}
    return json.dumps(state_dict_serializable)


def train_MLP_1():
    df = data_import_and_process('data/binary_classification.csv')
    # Since diagnosis is already binary and data is normalized,
    # we can directly split them
    features = df.columns[2:]  # All columns except id and diagnosis
    data = torch.tensor(df[features].values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=57)

    # Split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = models.MLP_1L_1n(data_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.unsqueeze(1).float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model
    with torch.no_grad():
        test_predictions = model(data_test.float())
        test_predictions = (test_predictions > 0.5).float()
        test_accuracy = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]

    print(f'1 layer MLP Model Accuracy: {test_accuracy:.2%}')

    # Save model weights to a json file
    state_dict = model.state_dict()
    state_dict_json = state_dict_to_json(state_dict)
    with open('weights_biases/MLP_dict_1.json', 'w') as f:
        f.write(state_dict_json)
'''
def train_MLP_2():
    df = data_import('binary_classification.csv')

    # Convert diagnosis to binary
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

    # Separate labels and features
    features = df.columns[2:]  # All columns except id and diagnosis
    data = torch.tensor(df[features].values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=57)

    # Split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = models.TwoLayerMLP(data_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.unsqueeze(1).float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model
    with torch.no_grad():
        test_predictions = model(data_test.float())
        test_predictions = (test_predictions > 0.5).float()
        test_accuracy = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]

    print(f'2 layer MLP Model Accuracy: {test_accuracy:.2%}')

    # Save model weights
    state_dict = model.state_dict()
    state_dict_json = state_dict_to_json(state_dict)
    with open('./weights_biases/MLP_dict_2.json', 'w') as f:
        f.write(state_dict_json)

def train_MLP_3():
    df = data_import('binary_classification.csv')

    # Convert diagnosis to binary
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

    # Separate labels and features
    features = df.columns[2:]  # All columns except id and diagnosis
    data = torch.tensor(df[features].values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=57)

    # Split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = models.ThreeLayerMLP(data_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.unsqueeze(1).float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model
    with torch.no_grad():
        test_predictions = model(data_test.float())
        test_predictions = (test_predictions > 0.5).float()
        test_accuracy = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]

    print(f'3 layer MLP Model Accuracy: {test_accuracy:.2%}')

    # Save model weights
    state_dict = model.state_dict()
    state_dict_json = state_dict_to_json(state_dict)
    with open('./weights_biases/MLP_dict_3.json', 'w') as f:
        f.write(state_dict_json)


def train_logisticRegression():
    # Read the data
    df = pd.read_csv('binary_classification.csv')

    # Convert diagnosis to binary
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

    # Separate labels and features
    features = df.columns[2:]  # All columns except id and diagnosis
    data = torch.tensor(df[features].values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=57)

    # Split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = models.LogisticRegressionModel(data_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 500
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.unsqueeze(1).float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the model
    with torch.no_grad():
        test_predictions = model(data_test.float())
        test_predictions = (test_predictions > 0.5)
        test_accuracy2 = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]

    print(f'Logistic Regression Model Accuracy: {test_accuracy2:.2%}')

    state_dict = model.state_dict()
    state_dict_json = state_dict_to_json(state_dict)
    with open('./weights_biases/logRegression_dict.json', 'w') as f:
        f.write(state_dict_json)
'''

# train logisticRegression and train perceptron
train_MLP_1()

#train_logisticRegression()
