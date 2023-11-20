import torch
from torch import nn
import torch.nn.functional as F


# Define Multilayer Perceptron architecture
class MLP_1L_1n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_1L_1n, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x))


# Define variants for 2-layer MLP
class MLP_2L_1n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_2L_1n, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)  # Input to Hidden Layer with 1 neuron
        self.fc2 = nn.Linear(1, 1)  # Hidden Layer to Output

    def forward(self, x):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))


class MLP_2L_2n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_2L_2n, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)  # Input to Hidden Layer with 2 neurons
        self.fc2 = nn.Linear(2, 1)  # Hidden Layer to Output

    def forward(self, x):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))


class MLP_2L_3n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_2L_3n, self).__init__()
        self.fc1 = nn.Linear(input_dim, 3)  # Input to Hidden Layer with 3 neurons
        self.fc2 = nn.Linear(3, 1)  # Hidden Layer to Output

    def forward(self, x):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))


class MLP_2L_4n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_2L_4n, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4)  # Input to Hidden Layer with 4 neurons
        self.fc2 = nn.Linear(4, 1)  # Hidden Layer to Output

    def forward(self, x):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))


# Define a 3-layer MLP with 1 neuron in each hidden layer
class MLP_3L_1n1n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_3L_1n1n, self).__init__()
        # Input to first Hidden Layer
        self.fc1 = nn.Linear(input_dim, 1)
        # Second Hidden Layer
        self.fc2 = nn.Linear(1, 1)
        # Third Hidden Layer to Output
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))


class MLP_3L_2n1n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_3L_2n1n, self).__init__()
        # Input to first Hidden Layer
        self.fc1 = nn.Linear(input_dim, 2)
        # Second Hidden Layer
        self.fc2 = nn.Linear(2, 1)
        # Third Hidden Layer to Output
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))


class MLP_3L_3n1n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_3L_3n1n, self).__init__()
        # Input to first Hidden Layer
        self.fc1 = nn.Linear(input_dim, 3)
        # Second Hidden Layer
        self.fc2 = nn.Linear(3, 1)
        # Third Hidden Layer to Output
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))


class MLP_3L_4n1n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_3L_4n1n, self).__init__()
        # Input to first Hidden Layer
        self.fc1 = nn.Linear(input_dim, 4)
        # Second Hidden Layer
        self.fc2 = nn.Linear(4, 1)
        # Third Hidden Layer to Output
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
