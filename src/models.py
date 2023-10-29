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


class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim):  # Set default input_dim to 5
        super(ThreeLayerMLP, self).__init__()

        # Input to Hidden Layer 1
        self.fc1 = nn.Linear(input_dim, 16)

        # Hidden Layer 1 to Hidden Layer 2
        self.fc2 = nn.Linear(16, 8)

        # Hidden Layer 2 to Output
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification


# Define Logistic Regression architecture
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # Linear layer

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Forward pass with sigmoid for probability


class SVM(torch.nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


'''
# Define Neural Network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(hidden_size, 1)  # Second fully connected layer

    def forward(self, x):
        out = self.relu(self.fc1(x))  # Apply ReLU after first layer
        return torch.sigmoid(self.fc2(out))  # Apply sigmoid for probability

# Define Convolutional Neural Network architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # First convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5)  # Second convolutional layer
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # First fully connected layer
        self.fc2 = nn.Linear(120, 84)  # Second fully connected layer
        self.fc3 = nn.Linear(84, 10)  # Third fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply first convolutional layer, ReLU, and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolutional layer, ReLU, and pooling
        x = x.view(-1, 16 * 4 * 4)  # Flatten tensor
        x = F.relu(self.fc1(x))  # Apply first fully connected layer and ReLU
        x = F.relu(self.fc2(x))  # Apply second fully connected layer and ReLU
        return self.fc3(x)  # Return output from third fully connected layer
'''
