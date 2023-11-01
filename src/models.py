import torch
from torch import nn
import torch.nn.functional as F


# Define Perceptron architecture
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Single fully connected layer        

    def forward(self, x):
        return torch.sign(self.fc(x))  # Forward pass


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
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class MLP_2L_2n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_2L_2n, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)  # Input to Hidden Layer with 2 neurons
        self.fc2 = nn.Linear(2, 1)  # Hidden Layer to Output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class MLP_2L_3n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_2L_3n, self).__init__()
        self.fc1 = nn.Linear(input_dim, 3)  # Input to Hidden Layer with 3 neurons
        self.fc2 = nn.Linear(3, 1)  # Hidden Layer to Output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class MLP_2L_4n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_2L_4n, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4)  # Input to Hidden Layer with 3 neurons
        self.fc2 = nn.Linear(4, 1)  # Hidden Layer to Output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class MLP_2L_5n(nn.Module):
    def __init__(self, input_dim):
        super(MLP_2L_5n, self).__init__()
        self.fc1 = nn.Linear(input_dim, 5)  # Input to Hidden Layer with 3 neurons
        self.fc2 = nn.Linear(5, 1)  # Hidden Layer to Output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
