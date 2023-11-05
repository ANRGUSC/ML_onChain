//SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;
import { SD59x18 , convert, sd} from "../lib/prb-math/src/SD59x18.sol";


contract MLP_1L_1N {

    int256[][] public weights;  // 2D array for weights
    int256[] public biases;     // 1D array for biases
    int256[][] public training_data;      // 1D array for inputs
    int public correct_Count;

    constructor(uint256 neurons) {
        biases = new int256[](neurons);
    }

    function set_Biases(int256[] calldata b) external {
        require(b.length == biases.length, "Size of input biases does not match neuron number");
        biases = b;
    }

    function set_Weights(int256[] calldata w) external {
        int256[] memory temp_w = new int256[](w.length);
        for (uint256 i = 0; i < w.length; i++) {
            temp_w[i] = w[i];
        }
        weights.push(temp_w);
    }

    function view_dataset_size() external view returns(uint256 size){
        size = training_data.length;
    }

    function set_TrainingData(int256[] calldata d) external {
        int256[] memory temp_d= new int256[](d.length);
        for (uint256 i = 0; i < d.length; i++) {
            temp_d[i] = d[i];
        }
        training_data.push(temp_d);
    }

    // variable with _cvt mean it is converted from int256 to SD59x18

    // calculate the sigmoid of x
    function sigmoid(SD59x18 x) public pure returns (SD59x18) {
        int256 one = 1;
        SD59x18 one_cvt = convert(one);
        return (one_cvt).div(one_cvt.add((-x).exp()));
    }

    //relu activation function
    function relu(SD59x18 x) public pure returns (SD59x18) {
        int256 zero = 0;
        SD59x18 zero_cvt = convert(zero);
        if (x.gte(zero_cvt)){
            return x;
        }
        return zero_cvt;
    }

    function classify() public view returns (int){
        int correct = 0;

        for (uint256 j = 0; j < 50; j++) {
            // get each data item and its label
            int256[] memory data = training_data[j];
            int256 label = data[0];

            SD59x18 neuronResults; // one neuron

            //---------------------------------------------------
            // The first hidden layer with one neuron
            //---------------------------------------------------
            for (uint256 n = 0; n < 1; n++) {
                neuronResults = SD59x18.wrap(biases[n]);
                // each neuron in the first hidden layer
                for (uint256 i = 1; i < data.length; i++) {
                    SD59x18 a = SD59x18.wrap(data[i]);
                    SD59x18 b = SD59x18.wrap(weights[n][i-1]);
                    neuronResults = neuronResults.add(a.mul(b));
                }
                neuronResults = sigmoid(neuronResults);
            }
            //---------------------------------------------------
            // the output layer, since we are doing binary classification, we only need one neuron
            //---------------------------------------------------
            int256 classification;
            SD59x18 point_five = sd(0.5e18);
            if (neuronResults.gte(point_five)) {
                classification = int256(1e18);
            } else {
                classification = int256(0e18);
            }
            // count the number of correct classification
            if (label == classification) {
                correct++;
            }
        }
        return correct;
    }
}