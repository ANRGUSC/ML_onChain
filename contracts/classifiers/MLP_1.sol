// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.22 <0.9.0;

import "../libraries/ABDKMath64x64.sol";

contract MLP_1 {
    int[][] public fc_weights;  // 2D array for weights
    int[] public fc_biases;     // 1D array for biases

    int[] public inputData;
    int[] public classifiedResults;

    address public owner;

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    constructor(uint256 input_dim, uint256 neurons) {
        owner = msg.sender;
        fc_weights = new int[][](neurons);
        for (uint256 i = 0; i < neurons; i++) {
            fc_weights[i] = new int[](input_dim);
        }
        fc_biases = new int[](neurons);
    }

    function inputDataAndLabels(int[] memory data) public onlyOwner {
        inputData = data;
    }

    function setfc(int[][] memory weights, int[] memory biases) public onlyOwner {
        //require(weights.length == biases.length, "Weights and biases dimensions do not match");
        for (uint256 i = 0; i < weights.length; ++i) {
            for (uint256 j = 0; j < weights[i].length; ++j) {
                fc_weights[i][j] = weights[i][j];
            }
            fc_biases[i] = biases[i];
        }
    }

    function predict(int x) public view returns (int) {
        int res = 0;
        for (uint256 i = 0; i < fc_weights.length; ++i) {
            res += fc_weights[i][0] * x + fc_biases[i];
        }
        res = res > 0 ? res : int256(0);  // Activation (ReLu)
        return res;
    }

    function classifyAndStore() public {
        require(inputData.length > 0, "No input data provided");
        classifiedResults = new int[](inputData.length);
        for (uint i = 0; i < inputData.length; i++) {
            classifiedResults[i] = predict(inputData[i]);
        }
    }

    function getClassifiedResults() public view returns (int[] memory) {
        return classifiedResults;
    }
}
