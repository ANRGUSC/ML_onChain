"""
import sys
sys.path.append('../')
from interpreter.solGenerator import code_gen

# Single Layer Perceptron Test
def main():
    # Generate Perceptron contract
    num_weights = 5
    code_gen(model_type='perceptron', num_weights=num_weights)

    # Call Perceptron contract and compare output py-solc + web3.py



if __name__ == '__main__':
    main()
"""
import sys
sys.path.append('../../')
from src.pytorch.models import Perceptron
import torch
# # Instantiate the Perceptron with 3 as theimension
perceptron = Perceptron(3)

# Create a 4x3 tensor as input to the Perceptron
input_tensor = torch.tensor([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0],
                             [10.0, 11.0, 12.0]])

# Forward pass through the Perceptron
output = perceptron(input_tensor)

print(output)
