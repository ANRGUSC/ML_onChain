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


# Assuming the previously provided code ...

def prepare_dataset(filename):
    df = data_import_and_process(filename)
    features = df.columns[2:]
    data = torch.tensor(df[features].values, dtype=torch.float)
    labels = torch.tensor(df['diagnosis'].values, dtype=torch.float)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=57)
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    return data_train, data_test, labels_train, labels_test, train_loader


def train_model(train_loader, model, num_epochs=100):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.unsqueeze(1).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate_and_save(model, data_test, labels_test, filename):
    with torch.no_grad():
        test_predictions = model(data_test.float())
        test_predictions = (test_predictions > 0.5).float()
        test_accuracy = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]
    print(f'Model Accuracy: {test_accuracy:.2%} \n')
    state_dict = model.state_dict()
    state_dict_json = state_dict_to_json(state_dict)
    with open(filename, 'w') as f:
        f.write(state_dict_json)


# Now, train your models:

def train_all():
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    data_train, data_test, labels_train, labels_test, train_loader = prepare_dataset('data/binary_classification.csv')

    # MLP 1-layer
    model_1L = models.MLP_1L_1n(data_train.shape[1])
    trained_model_1L = train_model(train_loader, model_1L)
    print("MLP 1-layer 1 neuron")
    evaluate_and_save(trained_model_1L, data_test, labels_test, 'weights_biases/MLP_1L1.json')

    # MLP 2-layer 1 neuron
    model_2L1 = models.MLP_2L_1n(data_train.shape[1])
    trained_model_2L1 = train_model(train_loader, model_2L1)
    print("MLP 2-layer 1 neurons")
    evaluate_and_save(trained_model_2L1, data_test, labels_test, 'weights_biases/MLP_2L1.json')

    # MLP 2-layer 2 neurons
    model_2L2 = models.MLP_2L_2n(data_train.shape[1])
    trained_model_2L2 = train_model(train_loader, model_2L2)
    print("MLP 2-layer 2 neurons")
    evaluate_and_save(trained_model_2L2, data_test, labels_test, 'weights_biases/MLP_2L2.json')

    # MLP 2-layer 3 neurons
    model_2L3 = models.MLP_2L_3n(data_train.shape[1])
    trained_model_2L3 = train_model(train_loader, model_2L3)
    print("MLP 2-layer 3 neurons")
    evaluate_and_save(trained_model_2L3, data_test, labels_test, 'weights_biases/MLP_2L3.json')

    # MLP 2-layer 4 neurons
    model_2L4 = models.MLP_2L_4n(data_train.shape[1])
    trained_model_2L4 = train_model(train_loader, model_2L4)
    print("MLP 2-layer 4 neurons")
    evaluate_and_save(trained_model_2L4, data_test, labels_test, 'weights_biases/MLP_2L4.json')

    # MLP 2-layer 5 neurons
    model_2L5 = models.MLP_2L_5n(data_train.shape[1])
    trained_model_2L5 = train_model(train_loader, model_2L5)
    print("MLP 2-layer 5 neurons")
    evaluate_and_save(trained_model_2L5, data_test, labels_test, 'weights_biases/MLP_2L5.json')


train_all()

