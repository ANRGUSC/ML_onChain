// SPDX-License-Identifier: UNLICENSED 
pragma solidity ^0.8.0;

contract SimpleMLP {

    //relu fuction
    function relu(int x) public pure returns (int) {
        if (x>0) return x;
        else return 0;
    }


	function simpleMlp(int[2] memory Input, int[4][5] memory W) public pure returns (int) {
    	// Input to first layer
    	int neuron_2_input = Input[0]*W[0][2] + Input[1]*W[1][2];
    	int neuron_3_input = Input[0]*W[0][3] + Input[1]*W[1][3];

    	// Apply ReLU activation function
    	int neuron_2_output = relu(neuron_2_input);
    	int neuron_3_output = relu(neuron_3_input);

    	// Output of first layer (also input to second layer)
    	int[2] memory first_layer_output = [neuron_2_output, neuron_3_output];

    	// Second layer processing
    	int neuron_4_input = first_layer_output[0]* W[2][3] + first_layer_output[1]*W[3][3];

    	// Apply ReLU activation function
    	int neuron_4_output = relu(neuron_4_input);
    	
    	return neuron_4_output;
	}
}

