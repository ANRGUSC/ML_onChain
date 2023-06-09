import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import astpretty


class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sign(self.fc(x))


if __name__ == '__main__':
    m = nn.Linear(3,2)
    input = torch.rand(1,3)
    output = m(input)
    print('Output', output)
    print('Weights', m.weight)




    with open("perceptron.py") as source:
        tree = ast.parse(source.read())
    
    # astpretty.pprint(tree)
