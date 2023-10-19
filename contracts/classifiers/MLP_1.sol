// SPDX-License-Identifier: UNLICENSED
pragma solidity >=0.4.22 <0.9.0;

import "../libraries/ABDKMath64x64.sol";

contract MLP_1 {
	int[][] public fc;

    int[] public inputData;
    int[] public classifiedResults;

    function inputDataAndLabels(int[] memory data) public {
        inputData = data;
    }

    function setfc(int[][] memory value) public {
        for (uint256 i = 0; i < value.length; ++i) {
            for (uint256 j = 0; j < value[0].length; ++j) {
                fc[i][j] = value[i][j];
            }
        }
    }

    constructor(uint256 input_dim) {
        fc = new int[][](input_dim);
        for (uint256 i = 0; i < input_dim; i++) {
            fc[i] = new int[](2);
        }
    }

    function predict(int x) public view returns (int) {
        int res = 0;
        for (uint256 i = 0; i < 2; ++i) {
            res += fc[i][0] * x;  // Adjusted this line
        }
        res = res > 0 ? res : int256(0);  // Activation (ReLu)
        return res;
    }


    function classifyAndStore() public {
        require(inputData.length > 0, "No input data provided");

        classifiedResults = new int[](inputData.length);

        for (uint i = 0; i < inputData.length; i++) {
            classifiedResults[i] = predict(inputData[i]); // Fixed this line
        }
    }


    function getClassifiedResults() public view returns (int[] memory) {
        return classifiedResults;
    }
}







