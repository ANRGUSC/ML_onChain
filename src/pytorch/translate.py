from interpreter import *

torch_code_log = """
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))"""
torch_code_per = """
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  

    def forward(self, x):
        return torch.sign(self.fc(x))  
"""


# Testing the converter
tree = get_ast(torch_code_log)

path = "../classifiers/log_regression.sol"
output = "// SPDX-License-Identifier: UNLICENSED\npragma solidity >=0.4.22 <0.9.0;\n\n" + py_to_solidity(torch_code_log)
print(output)
f = open(path, "w")
f.write(output)
f.close()

#perceptron generation
tree = get_ast(torch_code_per)

path = "../classifiers/perceptron.sol"
output = "// SPDX-License-Identifier: UNLICENSED\npragma solidity >=0.4.22 <0.9.0;\n\n" + py_to_solidity(torch_code_per)
print(output)
f = open(path, "w")
f.write(output)
f.close()