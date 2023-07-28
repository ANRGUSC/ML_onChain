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
class MultiPerceptron(nn.Module):
    def __init__(self, input_dim):
        super(MultiPerceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Single fully connected layer
        self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sign(self.fc2(self.fc(x)))  # Forward pass

# Define Logistic Regression architecture
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # Linear layer

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Forward pass with sigmoid for probability


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
