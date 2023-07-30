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


# convert the dict to json
def state_dict_to_json(state_dict):
    state_dict_serializable = {name: tensor_to_list(param) for name, param in state_dict.items()}
    return json.dumps(state_dict_serializable)


def train_perceptron():
    df = data_import('synthetic_data.csv', False)

    # separate labels
    data = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float)
    labels = torch.tensor(df['label'].values, dtype=torch.float)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = models.Perceptron(data_train.shape[1])
    criterion = nn.MSELoss()
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
        test_accuracy = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]

    print(f'Perceptron Model Accuracy: {test_accuracy}')

    state_dict = model.state_dict()
    state_dict_json = state_dict_to_json(state_dict)
    with open('Perceptron_dict.json', 'w') as f:
        f.write(state_dict_json)


def train_logisticRegression():
    df = data_import('binary_classification.csv', True)

    # seperate labels
    data = torch.tensor(df.drop('diagnosis', axis=1).values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # split into train and test
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
        test_accuracy = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]

    print(f'Logistic Regression Model Accuracy: {test_accuracy}')

    state_dict = model.state_dict()
    state_dict_json = state_dict_to_json(state_dict)
    with open('logRegression_dict.json', 'w') as f:
        f.write(state_dict_json)


# train logisticRegression and train perceptron
train_perceptron()
train_logisticRegression()
"""
def train_NN():
    df = data_import('binary_classification.csv')

    # seperate labels
    data = torch.tensor(df.drop('diagnosis', axis=1).values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = models.SimpleNN(data_train.shape[1], 64)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 200
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
        test_accuracy = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]
    print(f'Neural Network Model Accuracy: {test_accuracy}')

train_logisticRegression()

def train_RNN():
    df = data_import('binary_classification.csv')

    # seperate labels
    data = torch.tensor(df.drop('diagnosis', axis=1).values, dtype=torch.float).unsqueeze(2)  # Add input_size dimension
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.long)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = models.SimpleRNN(1, 64, 2, 2)  # Input size is 1 because we have 1 feature per time step
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 200
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.float()

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
        _, predicted = torch.max(test_predictions.data, 1)  # Convert the output to class indices
        test_accuracy = (predicted == labels_test).sum().item() / labels_test.shape[0]  # Calculate accuracy

    print(f'RNN Model Accuracy: {test_accuracy}')

def train_CNN():
    df = data_import('binary_classification.csv')

    # separate labels
    data = torch.tensor(df.drop('diagnosis', axis=1).values, dtype=torch.float).unsqueeze(1)  # Added unsqueeze here
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.long)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = models.SimpleCNN(in_channels=1, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 200
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.float()

            # Forward pass
            outputs = model(inputs)  # No need to unsqueeze here anymore
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the model
    with torch.no_grad():
        test_predictions = model(data_test.float())
        _, predicted = torch.max(test_predictions.data, 1) # Convert the output to class indices
        test_accuracy = (predicted == labels_test).sum().item() / labels_test.shape[0] # Calculate accuracy

    print(f'CNN Model Accuracy: {test_accuracy}')
"""
