from data_import import *
import models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import json
import sys


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
    '''
    # Split the data: use 20% as test data, and the rest as training data
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=57)
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    # Create a DataFrame for the test set and save to a file
    test_df = pd.DataFrame(data_test.numpy(), columns=features)
    test_df.insert(0, 'diagnosis', labels_test.numpy())  # Insert the labels column at the beginning
    test_df.to_csv('./data/test_data.csv', index=False)

    '''
    # For debug 
    # Split the data: use the last 100 samples as test data, and the rest as training data
    data_train = data[50:]
    labels_train = labels[50:]
    data_test = data[:50]
    labels_test = labels[:50]
    train_dataset = MyDataSet(data_train, labels_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    return data_train, data_test, labels_train, labels_test, train_loader


def train_model(train_loader, model, num_epochs=200):
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


def evaluate_and_save(model, data_test, labels_test, filename, debug=False):
    with torch.no_grad():
        if debug:
            raw_outputs, test_predictions = model(data_test.float(), debug=True)
            custom_print("Raw outputs:", raw_outputs)  # This will print the raw outputs for debugging
        else:
            test_predictions = model(data_test.float())

        test_predictions = (test_predictions > 0.5).float()
        test_accuracy = (test_predictions == labels_test.unsqueeze(1)).sum().item() / labels_test.shape[0]
    custom_print(f'Model Accuracy: {test_accuracy:.2%} \n')
    state_dict = model.state_dict()
    state_dict_json = state_dict_to_json(state_dict)
    with open(filename, 'w') as f:
        f.write(state_dict_json)

def custom_print(output):
    with open('../results/Local_accuracy', 'a') as f:
        print(output)  # Print to the terminal
        f.write(output+'\n') # Write to the file
        f.close()

def train_all():

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    data_train, data_test, labels_train, labels_test, train_loader = prepare_dataset('data/binary_classification.csv')

    # MLP 1-layer
    model_1L = models.MLP_1L_1n(data_train.shape[1])
    trained_model_1L = train_model(train_loader, model_1L)
    custom_print("MLP_1L_1n")
    evaluate_and_save(trained_model_1L, data_test, labels_test, 'weights_biases/MLP_1L1.json')

    # MLP 2-layer 1 neuron
    model_2L1 = models.MLP_2L_1n(data_train.shape[1])
    trained_model_2L1 = train_model(train_loader, model_2L1)
    custom_print("MLP_2L_1n")
    evaluate_and_save(trained_model_2L1, data_test, labels_test, 'weights_biases/MLP_2L1.json')

    # MLP 2-layer 2 neurons
    model_2L2 = models.MLP_2L_2n(data_train.shape[1])
    trained_model_2L2 = train_model(train_loader, model_2L2)
    custom_print("MLP_2L_2n")
    evaluate_and_save(trained_model_2L2, data_test, labels_test, 'weights_biases/MLP_2L2.json')

    # MLP 2-layer 3 neurons
    model_2L3 = models.MLP_2L_3n(data_train.shape[1])
    trained_model_2L3 = train_model(train_loader, model_2L3)
    custom_print("MLP_2L_3n")
    evaluate_and_save(trained_model_2L3, data_test, labels_test, 'weights_biases/MLP_2L3.json')

    # MLP 2-layer 4 neurons
    model_2L4 = models.MLP_2L_4n(data_train.shape[1])
    trained_model_2L4 = train_model(train_loader, model_2L4)
    custom_print("MLP_2L_4n")
    evaluate_and_save(trained_model_2L4, data_test, labels_test, 'weights_biases/MLP_2L4.json')

    # MLP 3-layer 1 neurons
    model_3L1 = models.MLP_3L_1n1n(data_train.shape[1])
    trained_model_3L1 = train_model(train_loader, model_3L1)
    custom_print("MLP_3L_1n")
    evaluate_and_save(trained_model_3L1, data_test, labels_test, 'weights_biases/MLP_3L1.json')

    # MLP 3-layer 2 neurons
    model_3L2 = models.MLP_3L_2n1n(data_train.shape[1])
    trained_model_3L2 = train_model(train_loader, model_3L2)
    custom_print("MLP_3L_2n")
    evaluate_and_save(trained_model_3L2, data_test, labels_test, 'weights_biases/MLP_3L2.json')

    # MLP 3-layer 3 neurons
    model_3L3 = models.MLP_3L_3n1n(data_train.shape[1])
    trained_model_3L3 = train_model(train_loader, model_3L3)
    custom_print("MLP_3L_3n")
    evaluate_and_save(trained_model_3L3, data_test, labels_test, 'weights_biases/MLP_3L3.json')

    # MLP 3-layer 4 neurons
    model_3L4 = models.MLP_3L_4n1n(data_train.shape[1])
    trained_model_3L4 = train_model(train_loader, model_3L4)
    custom_print("MLP_3L_4n")
    evaluate_and_save(trained_model_3L4, data_test, labels_test, 'weights_biases/MLP_3L4.json')

train_all()
