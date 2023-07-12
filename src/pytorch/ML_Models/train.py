from data_import import *
import ml_models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def train_logisticRegression():
    df = data_import('data.csv')

    # seperate labels
    data = torch.tensor(df.drop('diagnosis', axis=1).values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # split into train and test
    train_dataset =MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = ml_models.LogisticRegressionModel(data_train.shape[1])
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

def train_MLP():
    df = data_import('data.csv')

    # seperate labels
    data = torch.tensor(df.drop('diagnosis', axis=1).values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.long)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = ml_models.MLP(data_train.shape[1], 64, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 200
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.float()
            #labels = labels.unsqueeze(1).float()

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
        _, predicted = torch.max(test_predictions.data, 1) # Convert the output to class indices
        test_accuracy = (predicted == labels_test).sum().item() / labels_test.shape[0] # Calculate accuracy

    print(f'MLP Regression Model Accuracy: {test_accuracy}')

def train_NN():
    df = data_import('data.csv')

    # seperate labels
    data = torch.tensor(df.drop('diagnosis', axis=1).values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = ml_models.SimpleNN(data_train.shape[1], 64)
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

def train_RNN():
    df = data_import('data.csv')

    # seperate labels
    data = torch.tensor(df.drop('diagnosis', axis=1).values, dtype=torch.float).unsqueeze(2)  # Add input_size dimension
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.long)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = ml_models.SimpleRNN(1, 64, 2, 2)  # Input size is 1 because we have 1 feature per time step
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
    df = data_import('data.csv')

    # separate labels
    data = torch.tensor(df.drop('diagnosis', axis=1).values, dtype=torch.float).unsqueeze(1)  # Added unsqueeze here
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.long)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # split into train and test
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    model = ml_models.SimpleCNN(in_channels=1, num_classes=2)
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




#train_logisticRegression()
#train_MLP()
#train_NN()
#train_RNN()
train_CNN()