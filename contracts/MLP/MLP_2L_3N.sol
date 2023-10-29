// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;
import { SD59x18 , convert, sd } from "../../lib/prb-math/src/SD59x18.sol";

contract MLP_2L_3N {
    int256[][] public weights;
    int256[] public biases1;
    int256 public bias2;

    int256[][] public training_data;
    int public correct_Count;

    constructor() {
        biases1 = new int256[](3);
    }

    function set_Biases1(int256[] calldata b) external {
        require(b.length == biases1.length, "Size of input biases does not match neuron number");
        biases1 = b;
    }

    function set_Bias2(int256 b) external {
        bias2 = b;
    }

    function set_Weights(int256[][] calldata w) external {
        weights = w;
    }

    function set_TrainingData(int256[] calldata d) external {
        training_data.push(d);
    }

    function sigmoid(SD59x18 x) public pure returns (SD59x18) {
        return (convert(1)).div((convert(1)).add((-x).exp()));
    }


    function classify() public view returns (int) {
        int correct = 0;
        for (uint256 j = 0; j < 100; j++) {
            int256[] memory data = training_data[j];

            SD59x18 intermediateOutput1 = sigmoid(SD59x18.wrap(biases1[0] + data[1]*weights[0][0]));
            SD59x18 intermediateOutput2 = sigmoid(SD59x18.wrap(biases1[1] + data[1]*weights[0][1]));
            SD59x18 intermediateOutput3 = sigmoid(SD59x18.wrap(biases1[2] + data[1]*weights[0][2]));

            SD59x18 result = SD59x18.wrap(bias2)
                .add(intermediateOutput1.mul(SD59x18.wrap(weights[1][0])))
                .add(intermediateOutput2.mul(SD59x18.wrap(weights[1][1])))
                .add(intermediateOutput3.mul(SD59x18.wrap(weights[1][2])));

            if ((result.gte(sd(0.5e18)) && data[0] == int256(1e18)) || (!result.gte(sd(0.5e18)) && data[0] == int256(0e18))) {
               correct++;
            }
        }
        return correct;
    }
}
