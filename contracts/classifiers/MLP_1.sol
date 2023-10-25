//SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;
import { SD59x18 } from "../../lib/prb-math/src/SD59x18.sol";
contract MLP_1 {

    int256[][] public weights;  // 2D array for weights
    int256[] public biases;     // 1D array for biases
    int256[][] public training_data;      // 1D array for inputs
    int[] public classifiedResults;

    constructor(uint256 input_dim, uint256 neurons) {
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

    function set_TrainingData(int256[] calldata d) external {
        int256[] memory temp_d= new int256[](d.length);
        for (uint256 i = 0; i < d.length; i++) {
            temp_d[i] = d[i];
        }
        training_data.push(temp_d);
    }

}