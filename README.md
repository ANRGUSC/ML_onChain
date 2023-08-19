# MLPINT

## About
MLPINT is a primitive pytorch to solidity interpreter to automate the transcription of machine learning classifiers on the Ethereum Blockchain.

## Pytorch Support
As of now MLPINT supports translating the following pytorch syntax

- classes inheriting the nn.Module
- __init__(*) and forward(*) member functions
- super(class_name, self).__init() constructor
- declaring nn.Linear layers in __init__(*)
- passing input along nn.Linear layers in forward(*)
- Applying an activation function on layer outputs in forward(*)
- Returning the activation output in forward(*)


## Getting Started
To use MLPINT, clone the repository and define your pytorch models in src/models.py. Add translate_model() calls in src/translate.py for your defined models then run src/translate.py. This will use our interpreter module to generate solidity contract code in contracts/classifiers/. 
TODO: Give running/deploying instructions

## Example
Here is a sample input/output pair for interpreting a single layer perceptron.

### Input 
```python
import torch
from torch import nn

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Single fully connected layer        

    def forward(self, x):
        return torch.sign(self.fc(x))  # Forward pass
```
### Output
```solidity
// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.22 <0.9.0;

contract Perceptron {
	int[] public fc;

	
    function setfc(int[] memory value) public {
        for (uint256 i = 0; i < value.length; ++i) {
            fc[i] = value[i];
        }
    }

	constructor(uint256 input_dim) {
 		fc = new int[](input_dim);
	}

	function predict(int[] memory x) public view returns (int[] memory) {	
        int[] memory res1 = new int[](1);
        int c = 0;
        for (uint256 i = 0; i < i; ++i) {
            c += fc[i] * x[i];
        }
        res1[0] = c;
        for (uint256 i = 0; i < res1.length; ++i) {
            res1[i] = ((res1[i] >= 0) ? ((res1[i] == 0) ? int(0) : int(1)) : -1);
        }
        return res1;
        
	}
 }
```