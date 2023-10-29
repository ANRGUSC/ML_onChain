//SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;
import { SD59x18 , convert, sd} from "../../lib/prb-math/src/SD59x18.sol";


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

    function view_Weights() external view returns(int256[][] memory weight){
        weight = weights;
    }

    function view_Biases() external view returns(int256[] memory bias){
        bias = biases;
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

    event deBugEvent(SD59x18 x);

    function classify() public view returns (int){
        int correct = 0;
        for (uint256 j = 0; j < 100; j++) {
            // get each data item and its label
            int256[] memory data = training_data[j];
            int256 label = data[0];
            SD59x18 result = SD59x18.wrap(biases[0]);

            // Start from index 1 as 0 is the label
            for (uint256 i = 1; i < data.length; i++) {
                SD59x18 a = SD59x18.wrap(data[i]);
                SD59x18 b = SD59x18.wrap(weights[0][i-1]); // Subtract 1 from i to match weights index
                result = result.add(a.mul(b)); // Subtract 1 from i to match weights index
            }

            int256 classification;

            SD59x18 point_five = sd(0.5e18);
            if (sigmoid(result).gte(point_five)) { // If sigmoid(result) >= 0.5
                classification = int256(1e18);
            } else {
                classification = int256(0e18);
            }

            if (label == classification) {
               correct++;  // Correctly classified
            }
        }
        return correct;
    }
}